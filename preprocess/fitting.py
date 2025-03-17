import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
import torch
import argparse
import numpy as np
import cv2
import trimesh
import pyrender
import glob

sys.path.append('../ext/Multiview-3DMM-Fitting')
from config.config import config
from lib.LandmarkDataset import LandmarkDataset
# from lib.Recorder import Recorder
from lib.Fitter import Fitter
from lib.face_models import get_face_model
from lib.Camera import Camera



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
        intrinsics = log_data['intrinsics']
        extrinsics = log_data['extrinsics']
     
        with torch.no_grad():
            vertices, landmarks = log_data['face_model']()
        
        for n, frame in enumerate(frames):
            os.makedirs(os.path.join(self.save_folder, frame), exist_ok=True)
            face_model.save('%s/params.npz' % (os.path.join(self.save_folder, frame)), batch_id=n)
            np.save('%s/lmk_3d.npy' % (os.path.join(self.save_folder, frame)), landmarks[n].cpu().numpy())
            if self.save_vertices:
                np.save('%s/vertices.npy' % (os.path.join(self.save_folder, frame)), vertices[n].cpu().numpy())

            faces = log_data['face_model'].faces.cpu().numpy()
            mesh_trimesh = trimesh.Trimesh(vertices=vertices[n].cpu().numpy(), faces=faces)
            # save the trimesh
            mesh_trimesh.export('%s/mesh_%d.obj' % (os.path.join(self.save_folder, frame), 0))
            
            if self.visualize:
                img_paths = sorted(glob.glob(os.path.join(self.img_folder, frame, 'image_*.jpg')))

                for v in range(intrinsics.shape[1]):
                    
                    origin_image = cv2.imread(img_paths[v])[:,:,::-1]
                    origin_image = cv2.resize(origin_image, (self.camera.image_size, self.camera.image_size))
                    
                    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
                    self.camera.init_renderer(intrinsic=intrinsics[n, v], extrinsic=extrinsics[n, v])
                    render_image = self.camera.render(mesh)

                    # merge the original image and the render image
                    render_image = cv2.resize(render_image, (self.camera.image_size, self.camera.image_size))

                    dark_mask = render_image < 50
                    # add background from orginal image to rendered image
                    render_image[dark_mask] = origin_image[dark_mask]

                    N = landmarks[n].shape[0]

                    # Convert points to homogeneous coordinates [N, 4]
                    ones = torch.ones((N, 1), device=landmarks[n].device)
                    points_h = torch.cat([landmarks[n], ones], dim=1)  # Shape: [N, 4]

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


                    render_image = np.concatenate([origin_image, render_image], axis=1)

                    cv2.imwrite('%s/vis_%d.jpg' % (os.path.join(self.save_folder, frame), v), render_image[:,:,::-1])

if __name__ == '__main__':

    Path_file = os.path.abspath(__file__)
    # from preprocess to the main directory, so that the assest reading is correct
    os.chdir(os.path.dirname(os.path.dirname(Path_file)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/sample_video.yaml')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    dataset = LandmarkDataset(landmark_folder=cfg.landmark_folder, camera_folder=cfg.camera_folder)
    face_model = get_face_model(cfg.face_model, batch_size=len(dataset), device=device)
    camera = Camera(image_size=cfg.image_size)
    recorder = Recorder(save_folder=cfg.param_folder, camera=camera, visualize=cfg.visualize, save_vertices=cfg.save_vertices, img_folder=cfg.image_folder)

    fitter = Fitter(cfg, dataset, face_model, camera, recorder, device)
    fitter.run()
