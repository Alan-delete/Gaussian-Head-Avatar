import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import sys
import torch
import argparse
import numpy as np
import cv2
import trimesh
import pyrender
import glob
from tqdm import tqdm

import torch
import numpy as np
import glob
import os
import random
import cv2
from skimage import io

import torch
import numpy as np
from einops import rearrange


import torch
import pyrender
import cv2
import numpy as np

import os
from yacs.config import CfgNode as CN
 
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, RasterizationSettings, 
    MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights, TexturesVertex, SoftSilhouetteShader 
)
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.transforms import Rotate, Translate 

sys.path.append('../')
# from lib.Recorder import Recorder
from lib.face_models import get_face_model

class config():

    def __init__(self):

        self.cfg = CN()
        self.cfg.image_folder = ''
        self.cfg.camera_folder = ''
        self.cfg.landmark_folder = ''
        self.cfg.param_folder = ''
        self.cfg.gpu_id = 0
        self.cfg.camera_ids = []
        self.cfg.image_size = 512
        self.cfg.face_model = 'BFM'
        self.cfg.reg_id_weight = 1e-6
        self.cfg.reg_exp_weight = 1e-6
        self.cfg.visualize = False
        self.cfg.save_vertices = False


    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self, config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()


class Camera():
    def __init__(self, image_size):
        self.image_size = image_size

        self.lights = []
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=60.0)
        for x in [-2, 2]:
            y = 2
            for z in [-2, 2]:
                light_pose = np.array([[1.0,  0.0,  0.0,   x],
                                        [0.0,  1.0,  0.0,   y],
                                        [0.0,  0.0,  1.0,   z],
                                        [0.0,  0.0,  0.0,   1.0]])
                self.lights.append((light, light_pose))

    def project(self, query_pts, calibrations):
        query_pts = torch.bmm(calibrations[:, :3, :3], query_pts)
        query_pts = query_pts + calibrations[:, :3, 3:4]
        query_pts_xy = query_pts[:, :2, :] / query_pts[:, 2:, :]
        query_pts_xy = query_pts_xy
        return query_pts_xy

    def init_renderer(self, intrinsic, extrinsic):
        self.R = extrinsic[0:3, 0:3]
        self.T = extrinsic[0:3, 3:4]
        self.K = intrinsic

        Rotate_y_180 = torch.eye(3).to(self.R.device)
        Rotate_y_180[0,0] = -1.0
        Rotate_y_180[2,2] = -1.0
        R_pyrender = torch.matmul(torch.inverse(self.R), Rotate_y_180).float()
        T_pyrender = -torch.matmul(torch.inverse(self.R), self.T)[:,0].float()

        self.renderer = pyrender.IntrinsicsCamera(self.K[0,0], self.K[1,1], self.image_size - self.K[0,2], self.image_size - self.K[1,2])
        self.camera_pose = np.eye(4)
        self.camera_pose[0:3,0:3] = R_pyrender.cpu().numpy()
        self.camera_pose[0:3,3] = T_pyrender.cpu().numpy()
    
    def render(self, mesh, return_mask=False):
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[0.0, 0.0, 0.0])
        scene.add(mesh)
        for light in self.lights:
            scene.add(light[0], pose=light[1])
        scene.add(self.renderer, pose=self.camera_pose)
        osr = pyrender.OffscreenRenderer(self.image_size, self.image_size)
        color, depth = osr.render(scene)
        color = cv2.flip(color, -1)
        depth = cv2.flip(depth, -1)
        if return_mask:
            return color, (depth > 0).astype(np.uint8) * 255
        else:
            return color


class Fitter():
    def __init__(self, cfg, dataset, face_model, camera, recorder, device):
        self.cfg = cfg
        self.dataset = dataset
        self.face_model = face_model
        self.camera = camera
        self.recorder = recorder
        self.device = device

        # self.optimizers = [torch.optim.Adam([{'params' : self.face_model.scale, 'lr' : 1e-3},
        #                                      {'params' : self.face_model.global_translation, 'lr' : 1e-2},
        #                                      {'params' : self.face_model.global_rotation, 'lr' : 1e-2},]),
        #                    torch.optim.Adam([{'params' : self.face_model.parameters(), 'lr' : 1e-3}])]

        self.optimizers = [torch.optim.Adam([{'params' : self.face_model.scale, 'lr' : 1e-3},
                                            #  {'params':  self.face_model.neck_pose, 'lr' : 1e-2},
                                             {'params' : self.face_model.pose, 'lr' : 1e-2},]),
                           torch.optim.Adam([{'params' : self.face_model.parameters(), 'lr' : 1e-3}])]
    
    def run(self):
        landmarks_gt, extrinsics0, intrinsics0, frames = self.dataset.get_item()
        landmarks_gt = torch.from_numpy(landmarks_gt).float().to(self.device)
        extrinsics0 = torch.from_numpy(extrinsics0).float().to(self.device)
        intrinsics0 = torch.from_numpy(intrinsics0).float().to(self.device)
        extrinsics = rearrange(extrinsics0, 'b v x y -> (b v) x y')
        intrinsics = rearrange(intrinsics0, 'b v x y -> (b v) x y')
        
        for optimizer in self.optimizers:
            pprev_loss = 1e8
            prev_loss = 1e8

            for i in tqdm(range(int(1e10))):
                _, landmarks_3d = self.face_model()
                landmarks_3d = landmarks_3d.unsqueeze(1).repeat(1, landmarks_gt.shape[1], 1, 1)
                landmarks_3d = rearrange(landmarks_3d, 'b v x y -> (b v) x y')

                landmarks_2d = self.project(landmarks_3d, intrinsics, extrinsics)
                landmarks_2d = rearrange(landmarks_2d, '(b v) x y -> b v x y', b=landmarks_gt.shape[0])

                pro_loss = (((landmarks_2d / self.cfg.image_size - landmarks_gt[:, :, :, 0:2] / self.cfg.image_size) * landmarks_gt[:, :, :, 2:3]) ** 2).sum(-1).sum(-2).mean()
                reg_loss = self.face_model.reg_loss(self.cfg.reg_id_weight, self.cfg.reg_exp_weight)
                loss = pro_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if abs(loss.item() - prev_loss) < 1e-11 and abs(loss.item() - pprev_loss) < 1e-10:
                    print('iter: %d, loss: %.4f, pro_loss: %.4f, reg_loss: %.4f' % (i, loss.item(), pro_loss.item(), reg_loss.item()))
                    break
                else:
                    pprev_loss = prev_loss
                    prev_loss = loss.item()
                if i % 500 == 0:
                    print('iter: %d, loss: %.4f, pro_loss: %.4f, reg_loss: %.4f' % (i, loss.item(), pro_loss.item(), reg_loss.item()))
                
                # if i == 10000 :
                #     log = {
                #         'frames': frames,
                #         'landmarks_gt': landmarks_gt,
                #         'landmarks_2d': landmarks_2d.detach(),
                #         'face_model': self.face_model,
                #         'intrinsics': intrinsics0,
                #         'extrinsics': extrinsics0,
                #         'valid_cameras': self.dataset.valid_cameras,
                #     }
                #     self.recorder.log(log)

        log = {
            'fitter': self,
            'frames': frames,
            'landmarks_gt': landmarks_gt,
            'landmarks_2d': landmarks_2d.detach(),
            'face_model': self.face_model,
            'intrinsics': intrinsics0,
            'extrinsics': extrinsics0,
            'valid_cameras': self.dataset.valid_cameras,
            'all_cameras': self.dataset.all_cameras,
            'all_extrinsics': torch.from_numpy(self.dataset.all_extrinsics).float().to(self.device),
            'all_intrinsics': torch.from_numpy(self.dataset.all_intrinsics).float().to(self.device),
        }
        self.recorder.log(log)


    def project(self, points_3d, intrinsic, extrinsic):
        points_3d = points_3d.permute(0,2,1)
        calibrations = torch.bmm(intrinsic, extrinsic)
        points_2d = self.camera.project(points_3d, calibrations)
        points_2d = points_2d.permute(0,2,1)
        return points_2d

class LandmarkDataset():

    def __init__(self, landmark_folder, camera_folder):

        self.frames = sorted(os.listdir(landmark_folder))
        self.frames = [frame for frame in self.frames if os.path.isdir(os.path.join(landmark_folder, frame))]
        self.landmark_folder = landmark_folder
        self.camera_folder = camera_folder
        self.valid_cameras = []

    def get_item(self):
        landmarks = []
        extrinsics = []
        intrinsics = []

        #  (((landmarks_2d / self.cfg.image_size - landmarks_gt[:, :, :, 0:2] / self.cfg.image_size) * landmarks_gt[:, :, :, 2:3]) ** 2).sum(-1).sum(-2).mean()
        camera_ids = [item.split('_')[-1][:-4] for item in sorted(os.listdir(os.path.join(self.landmark_folder, self.frames[0])))]
        camera_ids = list(set(camera_ids))
        selected_cameras = []
        for v in range(len(camera_ids)):
            conf = 0
            for frame in self.frames:
                if os.path.exists(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v])):
                    landmark = np.load(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v]))
                    conf += np.mean(landmark[:, 2])
            selected_cameras.append(( conf, camera_ids[v]))
        selected_cameras = sorted(selected_cameras, key=lambda x: x[0], reverse=True)
        selected_cameras = [item[1] for item in selected_cameras]
        self.all_cameras = selected_cameras
        
        num_cameras = min(32, len(selected_cameras))
        selected_cameras = selected_cameras[:num_cameras]
        selected_cameras = sorted(selected_cameras)
        self.valid_cameras = selected_cameras
                    

        for frame in tqdm(self.frames):
            landmarks_ = []
            extrinsics_ = []
            intrinsics_ = []
            # camera_ids = [item.split('_')[-1][:-4] for item in sorted(os.listdir(os.path.join(self.landmark_folder, frame)))]
            for camera_id in self.all_cameras:
                if os.path.exists(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_id)):
                    landmark = np.load(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_id))
                    landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
                    extrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_id))['extrinsic']
                    intrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_id))['intrinsic']
                else:
                    landmark = np.zeros([66, 3], dtype=np.float32)
                    extrinsic = np.ones([3, 4], dtype=np.float32)
                    intrinsic = np.ones([3, 3], dtype=np.float32)   
                landmarks_.append(landmark)
                extrinsics_.append(extrinsic)
                intrinsics_.append(intrinsic)
            landmarks_ = np.stack(landmarks_)
            extrinsics_ = np.stack(extrinsics_)
            intrinsics_ = np.stack(intrinsics_)
            landmarks.append(landmarks_)
            extrinsics.append(extrinsics_)
            intrinsics.append(intrinsics_)

        self.all_landmarks = np.stack(landmarks)
        self.all_extrinsics = np.stack(extrinsics)
        self.all_intrinsics = np.stack(intrinsics)

        landmarks = self.all_landmarks[:, 0:num_cameras, :, :]
        extrinsics = self.all_extrinsics[:, 0:num_cameras, :, :]
        intrinsics = self.all_intrinsics[:, 0:num_cameras, :, :]

        return landmarks, extrinsics, intrinsics, self.frames
    
    def __len__(self):
        return len(self.frames)
    
    


class Recorder():
    def __init__(self, save_folder, camera, visualize=False, save_vertices=False, img_folder = None):

        self.save_folder = save_folder
        self.img_folder = img_folder
        os.makedirs(self.save_folder, exist_ok=True)

        self.camera = camera

        self.visualize = visualize
        self.save_vertices = save_vertices

    def log(self, log_data):
        frames = log_data['frames']
        face_model = log_data['face_model']
        intrinsics = log_data['all_intrinsics']
        extrinsics = log_data['all_extrinsics']
     
        with torch.no_grad():
            vertices, landmarks = log_data['face_model']()
        
            vertices_3d = vertices.unsqueeze(1).repeat(1, len(log_data['all_cameras']), 1, 1)
            vertices_3d = rearrange(vertices_3d, 'b v x y -> (b v) x y')

            extrinsics_merged = rearrange(extrinsics, 'b v x y -> (b v) x y')
            intrinsics_merged = rearrange(intrinsics, 'b v x y -> (b v) x y')
            vertices_2d =  log_data['fitter'].project(vertices_3d, intrinsics_merged, extrinsics_merged)
            # frame_num, cam_num, h, w
            vertices_2d = rearrange(vertices_2d, '(b v) x y -> b v x y', b=landmarks.shape[0])

        print('save per frame results to %s' % self.save_folder)
        for n, frame in tqdm(enumerate(frames)):
            os.makedirs(os.path.join(self.save_folder, frame), exist_ok=True)
            
            # face_model.save('%s/params.npz' % (os.path.join(self.save_folder, frame)), batch_id=n)
            # np.save('%s/lmk_3d.npy' % (os.path.join(self.save_folder, frame)), landmarks[n].cpu().numpy())
            # if self.save_vertices:
            #     np.save('%s/vertices.npy' % (os.path.join(self.save_folder, frame)), vertices[n].cpu().numpy())

            faces = log_data['face_model'].faces.cpu().numpy()
            mesh_trimesh = trimesh.Trimesh(vertices=vertices[n].cpu().numpy(), faces=faces)
            # # save the trimesh
            mesh_trimesh.export('%s/mesh_%d.obj' % (os.path.join(self.save_folder, frame), n))


            # valid_cameras = [28, 56, 22, 31, 25, 57, 27, 34, 35, 55, 32, 18, 19, 21]
            if self.visualize:
                # img_paths = sorted(glob.glob(os.path.join(self.img_folder, frame, 'image_*.jpg')))

                for v, camera_id in enumerate(log_data['all_cameras']):
                    img_paths = os.path.join(self.img_folder, frame, 'image_%s.jpg' % camera_id)
                    
                    origin_image = cv2.imread(img_paths)[:,:,::-1]
                    origin_image = cv2.resize(origin_image, (self.camera.image_size, self.camera.image_size))

                    # scene = trimesh.Scene(mesh_trimesh)
                    
                    # extrinsic_cv = np.eye(4)
                    # extrinsic_cv[:3, :] = extrinsics[n, v].cpu().numpy()
                    # camera_transform = np.linalg.inv(extrinsic_cv)
                    # scene.camera_transform = camera_transform

                    # K = intrinsics[n, v].cpu().numpy()
                    # image_width = image_height = self.camera.image_size
                    # fx, fy = K[0, 0], K[1, 1]
                    # fov_x = 2 * np.arctan(image_width / (2 * fx)) * 180 / np.pi
                    # fov_y = 2 * np.arctan(image_height / (2 * fy)) * 180 / np.pi
                    # scene.camera.fov = (fov_y, fov_x)

                    # # full_project = torch.bmm(intrinsic.unsqueeze(0), extrinsic.unsqueeze(0))
                    # # scene.camera_transform = full_project.cpu().numpy()
                    # scene.camera.resolution = (self.camera.image_size, self.camera.image_size)

                    # data = scene.save_image(resolution=(image_width, image_height), visible=False)
                    # render_image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


                    # mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
                    # self.camera.init_renderer(intrinsic=intrinsics[n, v], extrinsic=extrinsics[n, v])
                    # render_image = origin_image.copy()
                    # render_image = self.camera.render(mesh)


                    # Convert trimesh to pytorch3d mesh
                    verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32).unsqueeze(0).to("cuda")
                    faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64).unsqueeze(0).to("cuda")

                    # Assign white texture
                    # textures = TexturesVertex(verts_features=torch.ones_like(verts).to("cuda") * 0.5)
                    fixed_color = torch.tensor([1.0, 1.0, 1.0], device="cuda")
                    colors = fixed_color[None, None].expand_as(verts)  # (1, V, 3)
                    textures = TexturesVertex(verts_features=colors)
                    mesh = Meshes(verts=verts, faces=faces, textures=textures)

                    # Camera intrinsics
                    K = intrinsics[n, v]
                    fx, fy = K[0, 0], K[1, 1]
                    cx, cy = K[0, 2], K[1, 2]
                    H = W = self.camera.image_size

                    # # Use full intrinsics
                    # cameras = PerspectiveCameras(
                    #     focal_length=((fx, fy),),
                    #     principal_point=((cx, cy),),
                    #     image_size=((H, W),),
                    #     device="cuda",
                    #     in_ndc=False,
                    #     R=torch.tensor(extrinsics[n, v][:, :3], dtype=torch.float32).unsqueeze(0).to("cuda"),
                    #     T=torch.tensor(extrinsics[n, v][:, 3], dtype=torch.float32).unsqueeze(0).to("cuda")
                    # )

                    cameras = cameras_from_opencv_projection( R = torch.tensor(extrinsics[n, v][:, :3], dtype=torch.float32).unsqueeze(0).to("cuda"),
                                                             tvec = torch.tensor(extrinsics[n, v][:, 3], dtype=torch.float32).unsqueeze(0).to("cuda"),
                                                             camera_matrix = torch.tensor(intrinsics[n, v], dtype=torch.float32).unsqueeze(0).to("cuda"),
                                                            image_size=torch.tensor([H, W]).unsqueeze(0))


                    # scale = 1.0  # Adjust based on mesh size
                    # cameras = FoVOrthographicCameras(
                    #     device=device,
                    #     R=torch.tensor(extrinsics[n, v][:, :3].T, dtype=torch.float32).unsqueeze(0).to(device),
                    #     T=torch.tensor(extrinsics[n, v][:, 3], dtype=torch.float32).unsqueeze(0).to(device),
                    #     scale_xyz=((scale, scale, scale),),  # Uniform scaling
                    #     image_size=((H, W),)
                    # )

                    # Rasterization & Shader
                    raster_settings = RasterizationSettings(
                        image_size=H,
                        blur_radius=1e-6,
                        faces_per_pixel=10,
                    )

                    # Use full ambient lighting only
                    lights = PointLights(device=device, ambient_color=((1, 1, 1),),diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),))
                    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
                    
                    shader = SoftSilhouetteShader()

                    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),shader=shader)


                    # # DEBUG, it works
                    # from pytorch3d.utils import ico_sphere
                    # mesh = ico_sphere(level=2, device="cuda")  # Simple sphere mesh
                
                    # cameras = PerspectiveCameras(
                    #     focal_length=((fx, fy),),
                    #     principal_point=((cx, cy),),
                    #     image_size=((H, W),),
                    #     device="cuda",
                    #     in_ndc=False,
                    #     T=torch.tensor([0,0,5], dtype=torch.float32).unsqueeze(0).to("cuda") )

                    # renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),shader=shader)
                    # fixed_color = torch.tensor([1.0, 0.0, 0.0], device="cuda")
                    # colors = fixed_color[None, None].expand_as(mesh.verts_packed()[None])
                    # mesh.textures = TexturesVertex(verts_features=colors)

                    # Render
                    images = renderer(mesh)
                    render_image = images[0, ..., 3:].cpu().numpy()  # HxWx3 RGB, float [0,1]
                    # GRAY to RGB
                    render_image = cv2.cvtColor(render_image, cv2.COLOR_GRAY2RGB)

                    # render_image = images[0, ..., :3].cpu().numpy()  # HxWx3 RGB, float [0,1]
                    render_image = (render_image * 255).astype(np.uint8)

                    # Alpha blend with original
                    alpha = 0.65
                    origin_image_resized = cv2.resize(origin_image, (W, H))
                    render_image = cv2.addWeighted(origin_image_resized, 1 - alpha, render_image, alpha, 0)

                    # merge the original image and the render image
                    render_image = cv2.resize(render_image, (self.camera.image_size, self.camera.image_size))

                    # landmarks = vertices
                    N = landmarks[n].shape[0]


                    # Convert points to homogeneous coordinates [N, 4]
                    ones = torch.ones((N, 1), device=landmarks[n].device)
                    points_h = torch.cat([landmarks[n], ones], dim=1)  # Shape: [N, 4]

                    # R = extrinsics[n][v][:3, :3]  # Rotation part of the extrinsic matrix
                    # T = extrinsics[n][v][:3, 3]  # Translation part of
                    # T_ = Translate(T[None], device=R.device)
                    # R_ = Rotate(R[None], device=R.device)

                    # Transform to camera coordinates using extrinsic matrix
                    cam_coords = extrinsics[n][v] @ points_h.T  # Shape: [3, N]
                    
                    
                    # Project to image plane using intrinsic matrix
                    pixel_coords = intrinsics[n][v] @ cam_coords  # Shape: [3, N]

                    # Normalize homogeneous coordinates
                    U = pixel_coords[0, :] / pixel_coords[2, :]
                    V = pixel_coords[1, :] / pixel_coords[2, :]

                    for i in range(N):
                        u_ = int(U[i].item())
                        v_ = int(V[i].item())
                        cv2.circle(origin_image, (u_, v_), 5, (0, 255, 0), -1)


                    # vertices_2d_cur = vertices_2d[n, v].cpu().numpy()
                    # # find bounding box of the vertices
                    # min_x = int(np.min(vertices_2d_cur[:, 0]))
                    # max_x = int(np.max(vertices_2d_cur[:, 0]))
                    # min_y = int(np.min(vertices_2d_cur[:, 1]))
                    # max_y = int(np.max(vertices_2d_cur[:, 1]))
                    # # further lift the bounding box a little bit
                    # # draw bounding box
                    # cv2.rectangle(origin_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

                    # # remove the image below the bounding box
                    # render_image[max_y:, :, :] = 0

                    # mask = np.ones_like(render_image, dtype=np.uint8) * 255
                    # mask[max_y:, :, :] = 0
                    # # save the mask
                    # cv2.imwrite('%s/visible_%s.jpg' % (os.path.join(self.save_folder, frame), camera_id), mask)

                    render_image = np.concatenate([origin_image, render_image], axis=1)

                    cv2.imwrite('%s/vis_%d.jpg' % (os.path.join(self.save_folder, frame), v), render_image[:,:,::-1])


if __name__ == '__main__':

    Path_file = os.path.abspath(__file__)
    # from preprocess to the main directory, so that the assest reading is correct
    os.chdir(os.path.dirname(os.path.dirname(Path_file)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/sample_video.yaml')
    parser.add_argument('--image_folder', type=str, default='datasets/NeRSemble/031/images')
    parser.add_argument('--landmark_folder', type=str, default='datasets/NeRSemble/031/landmarks')
    parser.add_argument('--param_folder', type=str, default='datasets/NeRSemble/031/params')
    parser.add_argument('--camera_folder', type=str, default='datasets/NeRSemble/031/cameras')
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--save_vertices', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--camera_ids', type=list, default=[], help='if not set, all cameras will be processed')
    parser.add_argument('--image_size', type=int, default=2048)
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()


    arg_cfg = ['image_folder', arg.image_folder, 'landmark_folder', arg.landmark_folder, 'camera_folder', arg.camera_folder,
               'visualize', arg.visualize, 'save_vertices', arg.save_vertices, 'gpu_id', arg.gpu_id, 'image_size', arg.image_size]
    cfg.merge_from_list(arg_cfg)

    
    device = torch.device('cuda:%d' % arg.gpu_id)
    torch.cuda.set_device(arg.gpu_id)

    dataset = LandmarkDataset(landmark_folder=arg.landmark_folder, camera_folder=arg.camera_folder)
    print("dataset done")
    face_model = get_face_model(cfg.face_model, batch_size=len(dataset), device=device)
    print("face model done")
    camera = Camera(image_size=arg.image_size)
    recorder = Recorder(save_folder=arg.param_folder, camera=camera, visualize=arg.visualize, save_vertices=arg.save_vertices, img_folder=arg.image_folder)
    print("recorder done")
    fitter = Fitter(cfg, dataset, face_model, camera, recorder, device)
    print("fitter done")
    fitter.run()
