import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys
import torch
import argparse
import numpy as np
import cv2
import trimesh
import pyrender

sys.path.append('../ext/Multiview-3DMM-Fitting')
from config.config import config
from lib.LandmarkDataset import LandmarkDataset
# from lib.Recorder import Recorder
from lib.Fitter import Fitter
from lib.face_models import get_face_model
from lib.Camera import Camera



class Recorder():
    def __init__(self, save_folder, camera, visualize=False, save_vertices=False):

        self.save_folder = save_folder
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

            if self.visualize:
                faces = log_data['face_model'].faces.cpu().numpy()
                mesh_trimesh = trimesh.Trimesh(vertices=vertices[n].cpu().numpy(), faces=faces)
                # save the trimesh
                mesh_trimesh.export('%s/mesh_%d.obj' % (os.path.join(self.save_folder, frame), 0))
                for v in range(intrinsics.shape[1]):
                    
                    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
                    self.camera.init_renderer(intrinsic=intrinsics[n, v], extrinsic=extrinsics[n, v])
                    render_image = self.camera.render(mesh)

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
    recorder = Recorder(save_folder=cfg.param_folder, camera=camera, visualize=cfg.visualize, save_vertices=cfg.save_vertices)

    fitter = Fitter(cfg, dataset, face_model, camera, recorder, device)
    fitter.run()
