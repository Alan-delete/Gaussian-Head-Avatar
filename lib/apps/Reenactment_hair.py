import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class Reenactment_hair():
    def __init__(self, dataloader, gaussianhead, gaussianhair,supres, camera, recorder, gpu_id, freeview):
        self.dataloader = dataloader
        self.gaussianhead = gaussianhead
        self.gaussianhair = gaussianhair
        self.supres = supres
        self.camera = camera
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
        
        frame_num = len(dataset.samples)

        for i in tqdm(range(frame_num)):
            
            torch.cuda.empty_cache()
            
            # optical_flow_head = torch.zeros([1, self.gaussianhead.xyz.shape[0], 3], device=self.device, requires_grad=False)
            optical_flow_head = torch.zeros([1, self.gaussianhead.xyz.shape[0], 3], device=self.device, requires_grad=True)
            # optical_flow_hair = torch.zeros([1, self.gaussianhair.num_strands * (self.gaussianhair.strand_length - 1) , 3], device=self.device, requires_grad=False)
            optical_flow_hair = torch.zeros([1, self.gaussianhair.num_strands * (self.gaussianhair.strand_length - 1) , 3], device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([optical_flow_hair, optical_flow_head], lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
            

            iteration += 1
            data = dataset.__getitem__(i, 25)

            # prepare data
            for data_item in to_cuda:
                data[data_item] = torch.tensor(data[data_item], device=self.device)
                data[data_item] = data[data_item].unsqueeze(0)
                data[data_item].requires_grad = False

            with torch.no_grad():
                head_data = self.gaussianhead.generate(data)

                self.gaussianhair.generate_hair_gaussians(poses_history = data['poses_history'][0], 
                                                          pose = data['pose'][0],
                                                          scale = data['scale'][0])
                hair_data = self.gaussianhair.generate(data)
                # combine head and hair data
                for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                    # first dimension is batch size, concat along the second dimension
                    data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

                data = self.camera.render_gaussian(data, 512)
                render_images = data['render_images']
                video.append(render_images[0].permute(1,2,0).cpu().numpy())
        
        # save video
        # video = torch.stack(video, dim=0)
        # video = video.permute(0, 2, 3, 1).cpu().numpy()
        # save ground truth video
        output_path = os.path.join("./test.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video[0].shape[1], video[0].shape[0]))
        for frame in video:
            frame = (frame*255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()
        print('Saved video!')
