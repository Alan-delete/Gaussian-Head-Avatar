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
            mesh_trimesh.export('%s/mesh_%d.obj' % (os.path.join(self.save_folder, frame), n))
            # valid_cameras = [28, 56, 22, 31, 25, 57, 27, 34, 35, 55, 32, 18, 19, 21]
            if self.visualize:
                # img_paths = sorted(glob.glob(os.path.join(self.img_folder, frame, 'image_*.jpg')))

                for v, camera_id in enumerate(log_data['valid_cameras']):
                    img_paths = os.path.join(self.img_folder, frame, 'image_%s.jpg' % camera_id)
                    
                    origin_image = cv2.imread(img_paths)[:,:,::-1]
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
