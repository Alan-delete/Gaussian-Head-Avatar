import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class Reenactment_hair():
    def __init__(self, dataloader, gaussianhead, gaussianhair,supres, camera, recorder, gpu_id, freeview, camera_id=23):
        self.dataloader = dataloader
        self.gaussianhead = gaussianhead
        self.gaussianhair = gaussianhair
        self.supres = supres
        self.camera = camera
        self.camera_id = camera_id
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.freeview = freeview


    def run(self):

        to_cuda = ['images', 'masks', 'hair_masks','visibles', 'images_coarse', 'masks_coarse','hair_masks_coarse', 'visibles_coarse', 
                    'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                    'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id', 'fovx', 'fovy', 'orient_angle', 'flame_pose', 'flame_scale','poses_history', 'optical_flow', 'optical_flow_confidence']
        iteration = -1
        dataset = self.dataloader.dataset
        
        # set all parameters of gaussianhair to be not trainable
        for param in self.gaussianhair.parameters():
            param.requires_grad = False
        for param in self.gaussianhead.parameters():
            param.requires_grad = False

        video = []
        gt_video = []
        
        frame_num = len(dataset.samples)

        head_vertices = []
        hair_strand_points = []
        hair_strand_points_posed = []
        hair_color = None

        # for i in tqdm(range(frame_num, 0, -1)):
        for i in tqdm(range(frame_num)):
            
            torch.cuda.empty_cache()
            
            iteration += 1
            data = dataset.__getitem__(i, self.camera_id)

            # prepare data
            for data_item in to_cuda:
                data[data_item] = torch.tensor(data[data_item], device=self.device)
                data[data_item] = data[data_item].unsqueeze(0)
                data[data_item].requires_grad = False

            with torch.no_grad():
                head_data = self.gaussianhead.generate(data)

                self.gaussianhair.generate_hair_gaussians(poses_history = data['poses_history'][0], 
                                                          global_pose = data['flame_pose'][0],
                                                          global_scale = data['flame_scale'][0])
                hair_data = self.gaussianhair.generate(data)
                # combine head and hair data
                for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                    # first dimension is batch size, concat along the second dimension
                    data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

                data = self.camera.render_gaussian(data, 512)
                render_images = data['render_images']
                gt_images = data['images']
                gt_video.append(gt_images[0].permute(1,2,0).clamp(0,1).cpu().numpy())
                video.append(render_images[0].permute(1,2,0).clamp(0,1).cpu().numpy())

                hair_strand_points_world_per_frame = self.gaussianhair.get_strand_points_world
                hair_strand_points_posed_per_frame = self.gaussianhair.get_strand_points_posed
                head_vertices_world_per_frame = self.gaussianhead.verts
                hair_strand_points.append(hair_strand_points_world_per_frame.cpu().numpy())
                hair_strand_points_posed.append(hair_strand_points_posed_per_frame.cpu().numpy())
                head_vertices.append(head_vertices_world_per_frame.squeeze(0).cpu().numpy())
                hair_color = data['color'].view(-1, 3).mean(dim=0).cpu().numpy()
        
        # save video
        output_path = os.path.join("{}/{}/test_{}.mp4".format(self.recorder.checkpoint_path, self.recorder.name, self.camera_id))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video[0].shape[1], video[0].shape[0]))
        for frame in video:
            frame = (frame*255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()
        print('Saved video to %s' % output_path)

        output_path = os.path.join("{}/{}/gt_{}.mp4".format(self.recorder.checkpoint_path, self.recorder.name, self.camera_id))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (gt_video[0].shape[1], gt_video[0].shape[0]))
        for frame in gt_video:
            frame = (frame*255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print('Saved video to %s' % output_path)

        # save head vertices
        head_vertices = np.stack(head_vertices, axis=0)
        faces = self.gaussianhead.faces.cpu().numpy()
        np.savez(os.path.join("{}/{}/head_vertices_{}.npz".format(self.recorder.checkpoint_path ,self.recorder.name, self.camera_id)), vertices=head_vertices, faces=faces)
        
        # save hair strand points
        hair_strand_points = np.stack(hair_strand_points, axis=0)
        hair_strand_points = hair_strand_points.reshape(hair_strand_points.shape[0], -1, 3)
        np.savez(os.path.join("{}/{}/hair_strand_points_{}.npz".format(self.recorder.checkpoint_path, self.recorder.name, self.camera_id)), points=hair_strand_points, color = hair_color)

        # save hair strand points posed
        hair_strand_points_posed = np.stack(hair_strand_points_posed, axis=0)
        hair_strand_points_posed = hair_strand_points_posed.reshape(hair_strand_points_posed.shape[0], -1, 3)
        np.savez(os.path.join("{}/{}/hair_strand_points_posed_{}.npz".format(self.recorder.checkpoint_path, self.recorder.name,self.camera_id)), points=hair_strand_points_posed, color = hair_color)

        print('Saved head vertices to %s' % os.path.join(self.recorder.checkpoint_path , self.recorder.name, 'head_vertices_{}.npz'.format(self.camera_id)))
        print('Saved hair strand points to %s' % os.path.join(self.recorder.checkpoint_path,self.recorder.name, 'hair_strand_points_{}.npz'.format(self.camera_id)))
        