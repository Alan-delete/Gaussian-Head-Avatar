import torch
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset
import numpy as np
import glob
import math
import os
import random
import cv2
from skimage import io
from PIL import Image
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.transforms import so3_exponential_map

from lib.utils.graphics_utils import getWorld2View2, getProjectionMatrix


def CropImage(left_up, crop_size, image=None, K=None):
    crop_size = np.array(crop_size).astype(np.int32)
    left_up = np.array(left_up).astype(np.int32)

    if not K is None:
        K[0:2,2] = K[0:2,2] - np.array(left_up)

    if not image is None:
        if left_up[0] < 0:
            image_left = np.zeros([image.shape[0], -left_up[0], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image_left, image])
            left_up[0] = 0
        if left_up[1] < 0:
            image_up = np.zeros([-left_up[1], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image_up, image])
            left_up[1] = 0
        if crop_size[0] + left_up[0] > image.shape[1]:
            image_right = np.zeros([image.shape[0], crop_size[0] + left_up[0] - image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image, image_right])
        if crop_size[1] + left_up[1] > image.shape[0]:
            image_down = np.zeros([crop_size[1] + left_up[1] - image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image, image_down])

        image = image[left_up[1]:left_up[1]+crop_size[1], left_up[0]:left_up[0]+crop_size[0], :]

    return image, K


def ResizeImage(target_size, source_size, image=None, K=None):
    if not K is None:
        K[0,:] = (target_size[0] / source_size[0]) * K[0,:]
        K[1,:] = (target_size[1] / source_size[1]) * K[1,:]

    if not image is None:
        image = cv2.resize(image, dsize=target_size)
    return image, K


class MeshDataset(Dataset):

    def __init__(self, cfg):
        super(MeshDataset, self).__init__()

        self.dataroot = cfg.dataroot
        self.camera_ids = cfg.camera_ids
        self.selected_frames = cfg.selected_frames

        if len(self.camera_ids) == 0:
            image_paths = sorted(glob.glob(os.path.join(self.dataroot, 'images', '*', 'image_[0-9]*.jpg')))
            self.camera_ids = set([os.path.basename(image_path).split('_')[1].split('.')[0] for image_path in image_paths])
        self.original_resolution = cfg.original_resolution
        self.resolution = cfg.resolution
        self.num_sample_view = cfg.num_sample_view
        self.samples = []

        image_folder = os.path.join(self.dataroot, 'images')
        mask_folder = os.path.join(self.dataroot, 'NeuralHaircut_masks')
        param_folder = os.path.join(self.dataroot, 'params')
        camera_folder = os.path.join(self.dataroot, 'cameras')
        # frames = os.listdir(image_folder)
        frames = [ frame for frame in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, frame))] if len(self.selected_frames) == 0 else self.selected_frames
        frames = sorted(frames)
        
        self.num_exp_id = 0
        for frame in frames:
            image_paths = [os.path.join(image_folder, frame, 'image_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            # mask_paths = [os.path.join(image_folder, frame, 'mask_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            mask_paths = [os.path.join(mask_folder, 'body', frame, 'image_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            visible_paths = [os.path.join(image_folder, frame, 'visible_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            camera_paths = [os.path.join(camera_folder, frame, 'camera_%s.npz' % camera_id) for camera_id in self.camera_ids]
            param_path = os.path.join(param_folder, frame, 'params.npz')
            landmarks_3d_path = os.path.join(param_folder, frame, 'lmk_3d.npy')
            vertices_path = os.path.join(param_folder, frame, 'vertices.npy')

            sample = (image_paths, mask_paths, visible_paths, camera_paths, param_path, landmarks_3d_path, vertices_path, self.num_exp_id)
            self.samples.append(sample)
            self.num_exp_id += 1
                                  
        init_landmarks_3d = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'lmk_3d.npy'))).float()
        init_vertices = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'vertices.npy'))).float()
        init_landmarks_3d = torch.cat([init_landmarks_3d, init_vertices[::100]], 0)

        param = np.load(os.path.join(param_folder, frames[0], 'params.npz'))
        pose = torch.from_numpy(param['pose'][0]).float()
        R = so3_exponential_map(pose[None, :3])[0]
        T = pose[None, 3:]
        S = torch.from_numpy(param['scale']).float()
        self.init_landmarks_3d_neutral = (torch.matmul(init_landmarks_3d- T, R)) / S


    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        index = index % len(self.samples)
        sample = self.samples[index]
        
        images = []
        masks = []
        visibles = []
        views = random.sample(range(len(self.camera_ids)), self.num_sample_view)
        for view in views:
            image_path = sample[0][view]
            image = cv2.resize(io.imread(image_path), (self.resolution, self.resolution))
            image = torch.from_numpy(image / 255).permute(2, 0, 1).float()
            images.append(image)

            mask_path = sample[1][view]
            mask = cv2.resize(io.imread(mask_path), (self.resolution, self.resolution))
            mask = mask[:, :, 0:1] if len(mask.shape) == 3 else mask[:, :, None]
            mask = torch.from_numpy(mask / 255).permute(2, 0, 1).float()
            masks.append(mask)

            visible_path = sample[2][view]
            if os.path.exists(visible_path):
                visible = cv2.resize(io.imread(visible_path), (self.resolution, self.resolution))[:, :, 0:1]
                visible = torch.from_numpy(visible / 255).permute(2, 0, 1).float()
            else:
                visible = torch.ones_like(image)
            visibles.append(visible)

        images = torch.stack(images)
        masks = torch.stack(masks)
        images = images * masks
        visibles = torch.stack(visibles)

        cameras = [np.load(sample[3][view]) for view in views]
        intrinsics = torch.stack([torch.from_numpy(camera['intrinsic']).float() for camera in cameras])
        extrinsics = torch.stack([torch.from_numpy(camera['extrinsic']).float() for camera in cameras])
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * 2 / self.original_resolution
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * 2 / self.original_resolution - 1
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * 2 / self.original_resolution
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * 2 / self.original_resolution - 1

        param_path = sample[4]
        param = np.load(param_path)
        pose = torch.from_numpy(param['pose'][0]).float()
        scale = torch.from_numpy(param['scale']).float()
        # unify shape for BFM and FLAME
        scale = scale.view(-1)
        exp_coeff = torch.from_numpy(param['exp_coeff'][0]).float()
        
        landmarks_3d_path = sample[5]
        landmarks_3d = torch.from_numpy(np.load(landmarks_3d_path)).float()
        vertices_path = sample[6]
        vertices = torch.from_numpy(np.load(vertices_path)).float()
        landmarks_3d = torch.cat([landmarks_3d, vertices[::100]], 0)

        exp_id = sample[7]

        return {
                'images': images,
                'masks': masks,
                'visibles': visibles,
                'pose': pose,
                'scale': scale,
                'exp_coeff': exp_coeff,
                'landmarks_3d': landmarks_3d,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'exp_id': exp_id}

    def __len__(self):
        # return len(self.samples)
        return max(len(self.samples), 128)


from torch.utils.data import Dataset

class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = [0] + list(torch.cumsum(torch.tensor([len(d) for d in datasets]), dim=0))

    def __len__(self):
        return self.cumulative_sizes[-1].item()

    def __getitem__(self, idx, view=None):
        for i in range(len(self.datasets)):
            if idx < self.cumulative_sizes[i + 1]:
                local_idx = idx - self.cumulative_sizes[i]
                if view is not None:
                    data = self.datasets[i].__getitem__(local_idx, view=view)
                else:
                    data = self.datasets[i].__getitem__(local_idx)
                data['timestep'] = idx
                return data


# TODO: use same model to predict mask and hair mask, and put them in unified folder.
class GaussianDataset(Dataset):

    def __init__(self, cfg, split_strategy = "train", debug_select_frames = []):
        super(GaussianDataset, self).__init__()

        self.dataroot = cfg.dataroot
        self.camera_ids = cfg.camera_ids
        self.selected_frames = cfg.selected_frames
        
        
        if len(self.camera_ids) == 0:
            image_paths = sorted(glob.glob(os.path.join(self.dataroot, 'images', '*', 'image_[0-9]*.jpg')))
            self.camera_ids = set([os.path.basename(image_path).split('_')[1].split('.')[0] for image_path in image_paths])
        
        # if train:
        #     self.camera_ids =[camera_id for camera_id in self.camera_ids if camera_id not in cfg.test_camera_ids]
        # else:
        #     self.camera_ids = cfg.test_camera_ids
        
        self.camera_ids = sorted(self.camera_ids)

        self.original_resolution = cfg.original_resolution
        self.resolution = cfg.resolution
        self.coarse_scale_factor = cfg.coarse_scale_factor 

        self.samples = []

        image_folder = os.path.join(self.dataroot, 'images')
        param_folder = os.path.join(self.dataroot, 'params')
        camera_folder = os.path.join(self.dataroot, 'cameras')
        

        mask_folder = os.path.join(self.dataroot, 'NeuralHaircut_masks')
        if not os.path.exists(mask_folder):
            mask_folder = os.path.join(self.dataroot, 'masks')
        # mask_folder = os.path.join(self.dataroot, 'masks')

        # as tested, face_parsing is more robust than matte anything, but less accurate
        # if os.path.exists(os.path.join(self.dataroot, 'face-parsing', 'hair')):
        #     hair_mask_folder = os.path.join(self.dataroot, 'face-parsing', 'hair')
        # else:
        #     hair_mask_folder = os.path.join(self.dataroot, 'masks', 'hair')
        # hair_mask_folder = os.path.join(self.dataroot, 'face-parsing', 'hair')

        depth_folder = os.path.join(self.dataroot, 'depths')
        flame_param_folder = os.path.join(self.dataroot, 'FLAME_params')
        flame_param_VHAP_folder = os.path.join(self.dataroot, 'FLAME_params_VHAP')
        optical_flow_folder = os.path.join(self.dataroot, 'optical_flow')
        orientation_folder = os.path.join(self.dataroot, 'orientation_maps')
        orientation_confidence_folder = os.path.join(self.dataroot, 'orientation_confidence_maps')
        frames = [ frame for frame in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, frame))] if len(self.selected_frames) == 0 else self.selected_frames
        frames = sorted(frames)
        # split the frames into training and testing
        # decrease the number of frames for debugging 
        if split_strategy == "train":
            # frames = frames[::2] if len(self.selected_frames) == 0 else frames
            
            # exclude 0.6 to 0.75 frames for training
            # frames = frames[: int(len(frames) * 0.6)] \
            #         + [frames[int(len(frames) * 0.6)]] * 4 + [frames[int(len(frames) * 0.75)]] * 4\
            #         + frames[int(len(frames) * 0.75):] if len(self.selected_frames) == 0 else frames

            frames = frames[:int(len(frames) * 0.6)] if len(self.selected_frames) == 0 else frames
        
        elif split_strategy == "test":
            # frames = frames[1::2] if len(self.selected_frames) == 0 else frames
            # from 0.6 to 0.75
            # frames = frames[ int(len(frames) * 0.6): int(len(frames) * 0.75)] if len(self.selected_frames) == 0 else frames
            frames = frames[ int(len(frames) * 0.6): ] if len(self.selected_frames) == 0 else frames
        elif split_strategy == "all":
            frames = frames 
        else:
            raise ValueError('Unknown split strategy: %s, need to be one of string : train, test or all' % split_strategy)

        self.num_exp_id = 0
        params_path_history = []
        self.poses_history = []
        self.flame_mesh_path = os.path.join(flame_param_folder, '0000', 'mesh_0.obj')
        optical_flow_path = ['void_path' for _ in self.camera_ids]
        VHAP_flame_param_paths = []
        for frame in frames:
            image_paths = [os.path.join(image_folder, frame, 'image_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            depth_paths = [os.path.join(depth_folder, frame, 'image_%s.npy' % camera_id) for camera_id in self.camera_ids]
            # mask_paths = [os.path.join(image_folder, frame, 'mask_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            if os.path.exists(os.path.join(mask_folder, 'body', frame, 'image_%s.jpg' % self.camera_ids[0])):
                mask_format = 'jpg'
            else:
                mask_format = 'png'
            # mask_paths = [os.path.join(mask_folder, 'body', frame, 'image_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            mask_paths = [os.path.join(mask_folder, 'body', frame, 'image_%s.%s' % (camera_id, mask_format)) for camera_id in self.camera_ids]
            # mask_paths = [os.path.join(image_folder, frame, 'mask_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]

            hair_mask_path = [os.path.join(mask_folder,'hair', frame, 'image_%s.%s' % (camera_id, mask_format)) for camera_id in self.camera_ids]
            # hair_mask_path = [os.path.join(hair_mask_folder, frame, 'image_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]

            # head_mask_path = [os.path.join(mask_folder, 'head', frame, 'image_%s.%s' % (camera_id, mask_format)) for camera_id in self.camera_ids]
            head_mask_path = [ '' for camera_id in self.camera_ids]
            
            
            # visible_paths = [os.path.join(flame_param_folder, frame, 'visible_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            visible_paths = [os.path.join(image_folder, frame, 'visible_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            camera_paths = [os.path.join(camera_folder, frame, 'camera_%s.npz' % camera_id) for camera_id in self.camera_ids]
            param_path = os.path.join(param_folder, frame, 'params.npz')
            flame_param_path = os.path.join(flame_param_folder, frame, 'params.npz')
            landmarks_3d_path = os.path.join(param_folder, frame, 'lmk_3d.npy')
            vertices_path = os.path.join(param_folder, frame, 'vertices.npy')
            orientation_path = [os.path.join(orientation_folder, frame, 'image_%s.png' % camera_id) for camera_id in self.camera_ids]
            orientation_confidence_path = [os.path.join(orientation_confidence_folder, frame, 'image_%s.npy' % camera_id) for camera_id in self.camera_ids]

            # TODO: use dict instead of tuple
            # sample = (image_paths, mask_paths, visible_paths, camera_paths, 
            #           param_path, landmarks_3d_path, vertices_path, self.num_exp_id, 
            #           params_path_history, hair_mask_path, optical_flow_path, orientation_path, orientation_confidence_path, flame_param_path, head_mask_path, depth_paths)
            sample = {
                'image_paths': image_paths,
                'mask_paths': mask_paths,
                'visible_paths': visible_paths,
                'camera_paths': camera_paths,
                'param_path': param_path,
                'landmarks_3d_path': landmarks_3d_path,
                'vertices_path': vertices_path,
                'exp_id': self.num_exp_id,
                'params_path_history': params_path_history,
                'hair_mask_path': hair_mask_path,
                'optical_flow_path': optical_flow_path,
                'orientation_path': orientation_path,
                'orientation_confidence_path': orientation_confidence_path,
                'flame_param_path': flame_param_path,
                'head_mask_path': head_mask_path,
                'depth_paths': depth_paths
            }

            # the optical flow of t_1 is (position_1 - position_0), which is stored in the folder of t_0
            optical_flow_path = [os.path.join(optical_flow_folder, frame, 'image_%s.npy' % camera_id) for camera_id in self.camera_ids]
            
            self.samples.append(sample)
            # params_path_history.append(param_path)
            params_path_history.append(flame_param_path)
            self.num_exp_id += 1

            VHAP_flame_param_path = os.path.join(flame_param_VHAP_folder, frame, 'params.npz')
            VHAP_flame_param_paths.append(VHAP_flame_param_path)

        for flame_param_path in params_path_history:
            if os.path.exists(flame_param_path):
                pre_param = np.load(flame_param_path)
                pre_pose = torch.from_numpy(pre_param['pose'][0]).float()
            else:
                pre_pose = torch.zeros(6, dtype=torch.float32)
            self.poses_history.append(pre_pose)
        self.poses_history = torch.stack(self.poses_history)


        train_meshes = {}

        self.shape_dims = 100
        self.exp_dims = 50
        
        # for i, flame_param_path in enumerate(VHAP_flame_param_paths):
        for i, flame_param_path in enumerate(params_path_history):
            mesh = {}
            if os.path.exists(flame_param_path):
                flame_param = np.load(flame_param_path)

                # scale is around 0.995
                scale = flame_param['scale']
                # (1, 100)
                shape = flame_param['id_coeff']
                # (1, 3)
                # the rotation in GHA is not the same one in FLAME, so we just set it to 0
                rotation = flame_param['pose'][:, :3]
                # (1, 3)
                translation = flame_param['pose'][:, 3:]
                # (1, 59)
                exp_coeff = flame_param['exp_coeff']
                # (1, 3)
                jaw_pose = exp_coeff[:, self.exp_dims: self.exp_dims + 3]
                # (1, 6)
                neck_pose = np.zeros((1, 3))
                # (1, 3)
                eyes_pose = exp_coeff[:, self.exp_dims + 3: self.exp_dims + 9]
                expr = exp_coeff[:, :self.exp_dims]

                # the difference between FLAME and GHA is that the rotation in GHA is not the same one in FLAME
                # Flame is with respect to the root joint, while GHA is with respect to world origin
                rotation_mat = so3_exponential_map(torch.from_numpy(rotation)).detach().numpy()
                # hard code the root joint position, usually it won't change much
                J_0 = np.array([-0.0013, -0.1479, -0.0829], dtype=np.float32).reshape(1, 1, 3)
                translation = translation - J_0 + np.matmul(rotation_mat, J_0.transpose(0, 2, 1)).transpose(0, 2, 1)

                
                # expr = flame_param['expr'][:, :self.exp_dims]
                # rotation = flame_param['rotation']
                # translation = flame_param['translation']
                # jaw_pose = flame_param['jaw_pose']
                # neck_pose = flame_param['neck_pose']
                # eyes_pose = flame_param['eyes_pose']
                # shape = flame_param['shape'][:self.shape_dims]

                mesh['expr'] = expr
                mesh['rotation'] = rotation
                mesh['translation'] = translation
                mesh['jaw_pose'] = jaw_pose
                mesh['neck_pose'] = neck_pose
                mesh['eyes_pose'] = eyes_pose
                mesh['shape'] = shape
                train_meshes[i] = mesh

        self.train_meshes = train_meshes


        param_folder = os.path.join(self.dataroot, 'FLAME_params')
        
        if os.path.exists(params_path_history[0]):
            param = np.load(params_path_history[0])

            pose = torch.from_numpy(param['pose'][0]).float()
            self.R = so3_exponential_map(pose[None, :3])[0]
            self.T = pose[None, 3:]
            self.S = torch.from_numpy(param['scale']).float()
            self.pose = pose

            R = so3_exponential_map(pose[None, :3])[0]
            T = pose[None, 3:]
            S = torch.from_numpy(param['scale']).float()

        
        if os.path.exists(os.path.join(param_folder, frames[0], 'lmk_3d.npy')) and os.path.exists(os.path.join(param_folder, frames[0], 'vertices.npy')):
            init_landmarks_3d = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'lmk_3d.npy'))).float()
            init_vertices = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'vertices.npy'))).float()

            self.init_landmarks_3d_neutral = (torch.matmul(init_landmarks_3d- T, R)) / S
            self.init_flame_model = (torch.matmul(init_vertices- T, R)) / S

        # color -- random color from red, green, blue, white and black
        self.random_color = [torch.as_tensor([1.0, 0.0, 0.0]), torch.as_tensor([0.0, 1.0, 0.0]), torch.as_tensor([0.0, 0.0, 1.0]), torch.as_tensor([1.0, 1.0, 1.0]), torch.as_tensor([0.0, 0.0, 0.0])]
        self.bg_rgb_color = [self.random_color[np.random.randint(0, 5)] for _ in range(len(frames))]

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index, view=None):
        index = index % len(self.samples)

        sample = self.samples[index]
        # randomly pick a view
        view = random.sample(range(len(self.camera_ids)), 1)[0] if view is None else view % len(self.camera_ids)

        image_path = sample['image_paths'][view]
        image = cv2.resize(io.imread(image_path), (self.original_resolution, self.original_resolution)) / 255
        mask_path = sample['mask_paths'][view]
        mask = cv2.resize(io.imread(mask_path), (self.original_resolution, self.original_resolution)) / 255
        mask = mask[:, :, 0:1] if len(mask.shape) == 3 else mask[:, :, None]

        bg_rgb_color = self.random_color[np.random.randint(0, 5)]
        image = image * mask + (1 - mask) * bg_rgb_color.numpy()

        hair_mask_path = sample['hair_mask_path'][view]
        if os.path.exists(hair_mask_path):
            hair_mask = cv2.resize(io.imread(hair_mask_path), (self.original_resolution, self.original_resolution))[:, :, None] / 255
        else:
            raise ValueError('Hair mask not found')

        head_mask_path = sample['head_mask_path'][view]
        if os.path.exists(head_mask_path):
            head_mask = cv2.resize(io.imread(head_mask_path), (self.original_resolution, self.original_resolution))[:, :, None] / 255
        else:
            head_mask = np.ones_like(mask)

        visible_path = sample['visible_paths'][view]
        if os.path.exists(visible_path):
            visible = cv2.resize(io.imread(visible_path), (self.original_resolution, self.original_resolution))[:, :, 0:1] / 255
        else:
            visible = np.ones_like(mask)
        visible = visible * head_mask

        camera = np.load(sample['camera_paths'][view])
        extrinsic = torch.from_numpy(camera['extrinsic']).float()
        R = extrinsic[:3, :3].t()
        T = extrinsic[:3, 3]

        intrinsic = camera['intrinsic']
        intrinsic[0, 0] = intrinsic[0, 0] * 2 / self.original_resolution
        intrinsic[0, 2] = intrinsic[0, 2] * 2 / self.original_resolution - 1
        intrinsic[1, 1] = intrinsic[1, 1] * 2 / self.original_resolution
        intrinsic[1, 2] = intrinsic[1, 2] * 2 / self.original_resolution - 1
        intrinsic = torch.from_numpy(intrinsic).float()

        image = torch.from_numpy(cv2.resize(image, (self.resolution, self.resolution))).permute(2, 0, 1).float()
        mask = torch.from_numpy(cv2.resize(mask, (self.resolution, self.resolution)))[None].float()
        hair_mask = torch.from_numpy(cv2.resize(hair_mask, (self.resolution, self.resolution)))[None].float()
        visible = torch.from_numpy(cv2.resize(visible, (self.resolution, self.resolution)))[None].float()
        image_coarse = F.interpolate(image[None], scale_factor=self.coarse_scale_factor)[0]
        mask_coarse = F.interpolate(mask[None], scale_factor=self.coarse_scale_factor)[0]
        hair_mask_coarse = F.interpolate(hair_mask[None], scale_factor=self.coarse_scale_factor)[0]
        visible_coarse = F.interpolate(visible[None], scale_factor=self.coarse_scale_factor)[0]

        fovx = 2 * math.atan(1 / intrinsic[0, 0])
        fovy = 2 * math.atan(1 / intrinsic[1, 1])

        world_view_transform = torch.tensor(getWorld2View2(R.numpy(), T.numpy())).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fovx, fovY=fovy).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        param = np.load(sample['param_path'])
        pose = torch.from_numpy(param['pose'][0]).float()
        scale = torch.from_numpy(param['scale']).float().view(-1)
        exp_coeff = torch.from_numpy(param['exp_coeff'][0]).float()

        landmarks_3d_path = sample['landmarks_3d_path']
        if os.path.exists(landmarks_3d_path):
            landmarks_3d = torch.from_numpy(np.load(landmarks_3d_path)).float()
        else:
            landmarks_3d = torch.zeros(0, 3, dtype=torch.float32)

        exp_id = torch.tensor(sample['exp_id']).long()
        poses_history = self.poses_history[:index + 1]



        flame_param_path = sample['flame_param_path']
        # self.id_coeff = nn.Parameter(torch.zeros(1, self.shape_dims, dtype=torch.float32))
        # self.exp_coeff = nn.Parameter(torch.zeros(self.batch_size, self.exp_dims + 9, dtype=torch.float32)) # include expression_params, jaw_pose, eye_pose
        # self.pose = nn.Parameter(torch.zeros(batch_size, 6, dtype=torch.float32))
        # self.scale = nn.Parameter(torch.ones(1, 1, dtype=torch.float32))
        # ['id_coeff', 'exp_coeff', 'scale', 'pose']
        # for key in flame_param.files:
        if os.path.exists(flame_param_path):
            flame_param = np.load(flame_param_path)
            flame_pose = torch.from_numpy(flame_param['pose'][0]).float()
            flame_scale = torch.from_numpy(flame_param['scale']).float().view(-1)
            flame_exp_coeff = torch.from_numpy(flame_param['exp_coeff'][0]).float()
            flame_id_coeff = torch.from_numpy(flame_param['id_coeff'][0]).float()

            # expression_params = exp_coeff[:, : exp_dims]
            # jaw_rotation = exp_coeff[:, exp_dims: exp_dims + 3]
            # neck_pose = torch.zeros(exp_coeff.shape[0], 3, dtype=torch.float32)  # neck pose is not used in GHA
            # eye_pose = exp_coeff[:, exp_dims + 3: exp_dims + 9]

            # pose_params = torch.cat([self.global_rotation, jaw_rotation], 1)
            # shape_params = self.id_coeff.repeat(self.batch_size, 1)
            # vertices, landmarks = self.flame(shape_params, 
            #                                 expression_params, 
            #                                 pose_params, 
            #                                 neck_pose, 
            #                                 eye_pose)
        else:
            flame_pose = torch.zeros(6, dtype=torch.float32)
            flame_scale = torch.ones(1, dtype=torch.float32)


        # # DEBUG: fix the pose to the first frame
        # flame_pose = self.pose 

        # shape of 59
        # flame_exp_coeff = torch.from_numpy(flame_param['exp_coeff'][0]).float()
        # from lib.face_models.FLAMEModule import FLAMEModule
        # breakpoint()
        # flame_model = FLAMEModule(batch_size=1)
        data = {
                'timestep': index,
                'images': image,
                'masks': mask,
                'hair_masks': hair_mask,
                'visibles': visible,
                'images_coarse': image_coarse,
                'masks_coarse': mask_coarse,
                'hair_masks_coarse': hair_mask_coarse,
                'visibles_coarse': visible_coarse,
                # 'orient_angle': orient_angle,
                # 'orient_conf': orient_conf,
                # 'orient_angle_coarse': orient_angle_coarse,
                'pose': pose,
                'scale': scale,
                'exp_coeff': exp_coeff,
                'landmarks_3d': landmarks_3d,
                'exp_id': exp_id,
                'extrinsics': extrinsic,
                'intrinsics': intrinsic,
                'fovx': fovx,
                'fovy': fovy,
                'world_view_transform': world_view_transform,
                'projection_matrix': projection_matrix,
                'full_proj_transform': full_proj_transform,
                'camera_center': camera_center,
                'poses_history': poses_history,
                'flame_pose': flame_pose,
                'flame_scale': flame_scale,
                'flame_exp_coeff': flame_exp_coeff,
                'flame_id_coeff': flame_id_coeff,
                # 'optical_flow': optical_flow,
                # 'optical_flow_confidence': optical_flow_confidence,
                # 'optical_flow_coarse': optical_flow_coarse,
                # 'optical_flow_confidence_coarse': optical_flow_confidence_coarse,
                # 'bg_rgb_color': self.bg_rgb_color[index],
                'bg_rgb_color': bg_rgb_color,
                }

        orientation_path = sample['orientation_path'][view]
        orientation_confidence_path = sample['orientation_confidence_path'][view]
        if os.path.exists(orientation_path):
            resized_orient_angle = cv2.resize(io.imread(orientation_path), (self.resolution, self.resolution)) 
            resized_orient_angle = torch.from_numpy(resized_orient_angle).float() / 180.0
            resized_orient_angle = resized_orient_angle.view(self.resolution, self.resolution, 1).permute(2, 0, 1)
            orient_angle = resized_orient_angle[:1, ...] * hair_mask
            orient_angle_coarse = F.interpolate(orient_angle[None], scale_factor=self.coarse_scale_factor)[0]
            data['orient_angle'] = orient_angle
            data['orient_angle_coarse'] = orient_angle_coarse


        if False and os.path.exists(orientation_confidence_path):
            resized_orient_var = F.interpolate(torch.from_numpy(np.load(orientation_confidence_path)).float()[None, None], size=(self.resolution, self.resolution), mode='bilinear')[0] / math.pi**2
            resized_orient_conf = 1 / (resized_orient_var ** 2 + 1e-7)
            orient_conf = resized_orient_conf[:1, ...] * hair_mask
        else:
            orient_conf = torch.ones(1, self.resolution, self.resolution)


        depth_path = sample['depth_paths'][view]
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
            depth = cv2.resize(depth, (self.resolution, self.resolution))
            depth = torch.from_numpy(depth).float()[None]
            depth_coarse = F.interpolate(depth[None], scale_factor=self.coarse_scale_factor)[0]
            data['depth'] = depth
        
        optical_flow_path = sample['optical_flow_path'][view]
        optical_flow_confidence_path = optical_flow_path.replace('.npy', '_confidence_map.npy')
        optical_flow = torch.zeros(2, 128, 128)
        # optical_flow = torch.zeros(2, self.resolution, self.resolution)

        # if os.path.exists(optical_flow_path):
        #     data = np.load(optical_flow_path, allow_pickle=True)
        #     if data.dtype == object:
        #         data_dict = data.item()
        #         # N, 2
        #         coord = torch.from_numpy(data_dict['coord'])
        #         # N, 2
        #         flow = torch.from_numpy(data_dict['flow'])
        #         # N
        #         visibility = torch.from_numpy(data_dict['visibility']).bool()
        #         coord = coord[visibility]
        #         flow = flow[visibility]
        #         x = coord[:, 0]
        #         y = coord[:, 1]
        #         optical_flow[0, y, x] = flow[:,0]
        #         optical_flow[1, y, x] = flow[:,1]
        #     elif data.dtype == np.float32:
        #         # (2, H, W)
        #         optical_flow = torch.from_numpy(data)
        #         optical_flow = optical_flow.squeeze(0)
        #         # dense matching estimate from t+1 to t, need to invert it
        #         optical_flow = - optical_flow
        #         # TODO: opencv/numpy's orgin at top left, now should invert y axis of optical flow? 
        #         optical_flow[1, ...] = -optical_flow[1, ...]
        #     else:
        #         raise ValueError('Unknown optical flow data type')

        optical_flow_coarse = F.interpolate(optical_flow[None], scale_factor = self.coarse_scale_factor)[0]
        
        # if os.path.exists(optical_flow_confidence_path):
        #     # (1, H, W)
        #     optical_flow_confidence = torch.from_numpy(np.load(optical_flow_confidence_path)).float()
        #     # (H, W) -> (1, H, W)
        #     if len(optical_flow_confidence.shape) == 2:
        #         optical_flow_confidence = optical_flow_confidence.unsqueeze(0)
        #     optical_flow_confidence = F.interpolate(optical_flow_confidence[None], size=(self.resolution, self.resolution), mode='bilinear')[0]
        # else:
        #     optical_flow_confidence = torch.ones(1, self.resolution, self.resolution)
        optical_flow_confidence = torch.ones(1, self.resolution, self.resolution)
        optical_flow_confidence_coarse = F.interpolate(optical_flow_confidence[None], scale_factor=self.coarse_scale_factor)[0]
        
        return data

    def __len__(self):
        # return max(len(self.samples), 128)
        return len(self.samples)
    


class ReenactmentDataset(Dataset):

    def __init__(self, cfg):
        super(ReenactmentDataset, self).__init__()

        self.dataroot = cfg.dataroot
        self.original_resolution = cfg.original_resolution
        self.resolution = cfg.resolution
        self.freeview = cfg.freeview

        self.Rot_z = torch.eye(3)
        self.Rot_z[0,0] = -1.0
        self.Rot_z[1,1] = -1.0

        self.samples = []
        image_paths = sorted(glob.glob(os.path.join(self.dataroot, cfg.image_files)))
        param_paths = sorted(glob.glob(os.path.join(self.dataroot, cfg.param_files)))
        assert len(image_paths) == len(param_paths)
        
        self.samples = []
        for i, image_path in enumerate(image_paths):
            param_path = param_paths[i]
            if os.path.exists(image_path) and os.path.exists(param_path):
                sample = (image_path, param_path)
                self.samples.append(sample)

        if os.path.exists(cfg.pose_code_path):
            self.pose_code = torch.from_numpy(np.load(cfg.pose_code_path)['pose'][0]).float()
        else:
            self.pose_code = None


        self.extrinsic = torch.tensor([[1.0000,  0.0000,  0.0000,  0.0000],
                                       [0.0000, -1.0000,  0.0000,  0.0000],
                                       [0.0000,  0.0000, -1.0000,  1.0000]]).float()
        self.intrinsic = torch.tensor([[self.original_resolution * 3.5,   0.0000e+00,                     self.original_resolution / 2],
                                       [0.0000e+00,                     self.original_resolution * 3.5,   self.original_resolution / 2],
                                       [0.0000e+00,                     0.0000e+00,                       1.0000e+00]]).float()
        if os.path.exists(cfg.camera_path):
            camera = np.load(cfg.camera_path)
            self.extrinsic = torch.from_numpy(camera['extrinsic']).float()
            if not self.freeview:
                self.intrinsic = torch.from_numpy(camera['intrinsic']).float()
            
        self.R = self.extrinsic[:3,:3].t()
        self.T = self.extrinsic[:3, 3]

        self.intrinsic[0, 0] = self.intrinsic[0, 0] * 2 / self.original_resolution
        self.intrinsic[0, 2] = self.intrinsic[0, 2] * 2 / self.original_resolution - 1
        self.intrinsic[1, 1] = self.intrinsic[1, 1] * 2 / self.original_resolution
        self.intrinsic[1, 2] = self.intrinsic[1, 2] * 2 / self.original_resolution - 1

        self.fovx = 2 * math.atan(1 / self.intrinsic[0, 0])
        self.fovy = 2 * math.atan(1 / self.intrinsic[1, 1])

        self.world_view_transform = torch.tensor(getWorld2View2(self.R.numpy(), self.T.numpy())).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=self.fovx, fovY=self.fovy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update_camera(self, index):
        elev = math.sin(index / 20) * 8 + 0
        azim = math.cos(index / 20) * 45 - 0
        R, T = look_at_view_transform(dist=1.2, elev=elev, azim=azim, at=((0.0, 0.0, 0.05),))
        R = torch.matmul(self.Rot_z, R[0].t())
        self.extrinsic = torch.cat([R, T.t()], -1)

        self.R = self.extrinsic[:3,:3].t()
        self.T = self.extrinsic[:3, 3]

        self.world_view_transform = torch.tensor(getWorld2View2(self.R.numpy(), self.T.numpy())).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        if self.freeview:
            self.update_camera(index)

        sample = self.samples[index]
        
        image_path = sample[0]
        image = torch.from_numpy(cv2.resize(io.imread(image_path), (self.resolution, self.resolution)) / 255).permute(2, 0, 1).float()

        param_path = sample[1]
        param = np.load(param_path)
        pose = torch.from_numpy(param['pose'][0]).float()
        scale = torch.from_numpy(param['scale']).float()
        exp_coeff = torch.from_numpy(param['exp_coeff'][0]).float()

        if self.pose_code is not None:
            pose_code = self.pose_code
        else:
            pose_code = pose
        
        return {
                'images': image,
                'pose': pose,
                'scale': scale,
                'exp_coeff': exp_coeff,
                'pose_code': pose_code,
                'extrinsics': self.extrinsic,
                'intrinsics': self.intrinsic,
                'fovx': self.fovx,
                'fovy': self.fovy,
                'world_view_transform': self.world_view_transform,
                'projection_matrix': self.projection_matrix,
                'full_proj_transform': self.full_proj_transform,
                'camera_center': self.camera_center}

    def __len__(self):
        return len(self.samples)