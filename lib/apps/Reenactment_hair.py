import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import trimesh

from lib.utils.general_utils import ssim, psnr
from lib.face_models.FLAMEModule import FLAMEModule

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
        self.flame_model = FLAMEModule().to(self.device)


    def run(self):

        to_cuda = ['images', 'masks', 'hair_masks','visibles', 'images_coarse', 'masks_coarse','hair_masks_coarse', 'visibles_coarse', 
                    'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                    'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id', 'fovx', 'fovy', 'orient_angle', 'flame_pose', 'flame_scale', 
                    'flame_exp_coeff', 'flame_id_coeff', 'poses_history', 'optical_flow', 'optical_flow_confidence']
        iteration = -1
        dataset = self.dataloader.dataset
        
        # set all parameters of gaussianhair to be not trainable
        if self.gaussianhair is not None:
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

        if self.gaussianhair is not None:
            highlight_strands_idx = torch.arange(0, self.gaussianhair.num_strands, 300, device= self.device)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(77)
            highlight_color = torch.rand(highlight_strands_idx.shape[0], 3, generator=gen, device=self.device).unsqueeze(1).repeat(1, 99, 1).unsqueeze(0)
            
        init_flame_pose = dataset.__getitem__(0, self.camera_id)['flame_pose']
        init_flame_pose = torch.as_tensor( init_flame_pose, device=self.device).unsqueeze(0) 

        init_poses_history = dataset.__getitem__(0, self.camera_id)['poses_history']
        init_poses_history = [torch.as_tensor(init_poses_history, device=self.device).unsqueeze(0)]

        loss_ssim_arr = []
        loss_rgb_arr = []
        psnr_test_arr = []
        ssim_test_arr = []
        hair_psnr_test_arr = []
        hair_ssim_test_arr = []

        # for i in tqdm(range(frame_num, 0, -1)):
        for i in tqdm(range(frame_num)):
            
            torch.cuda.empty_cache()
            
            data = dataset.__getitem__(i, self.camera_id)

            # prepare data
            for data_item in to_cuda:
                data[data_item] = torch.tensor(data[data_item], device=self.device)
                data[data_item] = data[data_item].unsqueeze(0)
                data[data_item].requires_grad = False
            
            # data['poses_history'] = [None]
            data['bg_rgb_color'] = torch.as_tensor([1.0, 1.0, 1.0]).cuda()
            # TODO: select a few strands, color and enlarge them. Then render them

            with torch.no_grad():
                head_data = self.gaussianhead.generate(data)

                if self.gaussianhair is not None:
                    self.gaussianhair.generate_hair_gaussians(poses_history = data['poses_history'][0], 
                                                            # global_pose = init_flame_pose[0],
                                                            global_pose = data['flame_pose'][0], 
                                                            global_scale = data['flame_scale'][0])
                    hair_data = self.gaussianhair.generate(data)

                    # combine head and hair data
                    for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                        # first dimension is batch size, concat along the second dimension
                        data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)
                        # data[key] = hair_data[key]

                gt_hair_mask = data['hair_masks_coarse']
                images_coarse = data['images_coarse']
                visibles_coarse = data['visibles_coarse']
                visibles_coarse = visibles_coarse * gt_hair_mask

                data = self.camera.render_gaussian(data, images_coarse.shape[2])
                render_images = data['render_images'][: ,:3, ...]
                gt_images = data['images'][:, :3, ...]
                gt_video.append(gt_images[0].permute(1,2,0).clamp(0,1).cpu().numpy())
                video.append(render_images[0].permute(1,2,0).clamp(0,1).cpu().numpy())

                psnr_test = psnr(render_images[:, 0:3, :, :]  * visibles_coarse, images_coarse * visibles_coarse)
                ssim_test = ssim(render_images[:, 0:3, :, :]  * visibles_coarse, images_coarse * visibles_coarse)
                loss_ssim = 1.0 - ssim(render_images[:, 0:3, :, :]  * visibles_coarse, images_coarse * visibles_coarse)

                hair_psnr_test = psnr(render_images[:, :3, :, :] * gt_hair_mask, images_coarse * gt_hair_mask)
                hair_ssim_test = ssim(render_images[:, :3, :, :] * gt_hair_mask, images_coarse * gt_hair_mask)
                # loss functions
                loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)


                vertices , _ = self.flame_model(pose=data['flame_pose'], scale=data['flame_scale'], 
                                                exp_coeff=data['flame_exp_coeff'], id_coeff=data['flame_id_coeff'])
                faces = self.flame_model.faces
                head_vertices.append(vertices.squeeze(0).cpu().numpy())



                if self.gaussianhair is not None:
                    hair_strand_points_world_per_frame = self.gaussianhair.get_strand_points_world
                    hair_strand_points_posed_per_frame = self.gaussianhair.get_strand_points_posed
                    hair_strand_points.append(hair_strand_points_world_per_frame.cpu().numpy())
                    hair_strand_points_posed.append(hair_strand_points_posed_per_frame.cpu().numpy())
                    hair_color = data['color'][...,:3].view(-1, 3).mean(dim=0).cpu().numpy()
                    self.gaussianhair.sign_distance_loss()
                    
                    strands_origins = self.gaussianhair.get_strand_points_posed
                    
                    strands_opacity = self.gaussianhair.get_opacity.view(strands_origins.shape[0], -1)
                    strands_opacity = strands_opacity.mean(dim=-1)
                    
                    strands_origins = strands_origins[strands_opacity > 0.2]

                    cols = torch.cat((torch.rand(strands_origins.shape[0], 3).unsqueeze(1).repeat(1, 100, 1), torch.ones(strands_origins.shape[0], 100, 1)), dim=-1).reshape(-1, 4).cpu()           
                    # trimesh.PointCloud(strands_origins.reshape(-1, 3).detach().cpu(), colors=cols).export('strands_points.ply')

                    trimesh.PointCloud(strands_origins.reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(self.recorder.checkpoint_path, self.recorder.name, 'strands_points.ply'))
                    if i==0:
                        trimesh.PointCloud(strands_origins.reshape(-1, 3).detach().cpu(), colors=cols).export(os.path.join(self.recorder.checkpoint_path, self.recorder.name, 'strands_points_frame0.ply'))


                # if hasattr(self.gaussianhead, 'verts'):
                #     head_vertices_world_per_frame = self.gaussianhead.verts
                #     faces = self.gaussianhead.faces
                #     head_vertices.append(head_vertices_world_per_frame.squeeze(0).cpu().numpy())

                #     xyz = self.gaussianhead.get_xyz
                #     # save head vertices to a ply file
                #     trimesh.PointCloud(xyz.reshape(-1, 3).detach().cpu()).export(os.path.join(self.recorder.checkpoint_path, self.recorder.name, 'head_gaussian_xyz.ply'))
                    
                #     # render the mesh
                #     mesh = trimesh.Trimesh(vertices= head_vertices_world_per_frame.reshape(-1, 3).detach().cpu().numpy(), faces=faces.cpu().numpy())
                #     # projec the mesh to the image according to the camera parameters
                #     # mesh.apply_transform(data['full_proj_transform'][0].cpu().numpy())

                log = {
                    'data': data,
                    'gaussianhead' : self.gaussianhead,
                    'gaussianhair' : self.gaussianhair,
                    'camera' : self.camera,
                    'loss_rgb_lr' : loss_rgb_lr,
                    'loss_ssim' : loss_ssim,
                    'psnr' : psnr_test,
                    'ssim' : ssim_test,
                    # 'iter' : idx + epoch * len(self.dataloader)
                    'iter' : iteration
                    }
                with torch.no_grad():
                    self.recorder.log(log)
                
            loss_rgb_arr.append(loss_rgb_lr.item())
            loss_ssim_arr.append(loss_ssim.item())
            psnr_test_arr.append(psnr_test.item())
            ssim_test_arr.append(ssim_test.item())
            hair_psnr_test_arr.append(hair_psnr_test.item())
            hair_ssim_test_arr.append(hair_ssim_test.item())

        non_rigid_video = []
        if self.gaussianhair is not None:
            with torch.no_grad():
                for i in tqdm(range(frame_num)):
                    
                    torch.cuda.empty_cache()
                    
                    data = dataset.__getitem__(i, self.camera_id)

                    # prepare data
                    for data_item in to_cuda:
                        data[data_item] = torch.tensor(data[data_item], device=self.device)
                        data[data_item] = data[data_item].unsqueeze(0)
                        data[data_item].requires_grad = False
                    
                    # data['poses_history'] = [None]
                    data['bg_rgb_color'] = torch.as_tensor([1.0, 1.0, 1.0]).cuda()
                    # TODO: select a few strands, color and enlarge them. Then render them

                    head_data = self.gaussianhead.generate(data)

                    self.gaussianhair.generate_hair_gaussians(poses_history = data['poses_history'][0], 
                                                            # global_pose = init_flame_pose[0],
                                                            global_pose = data['flame_pose'][0], 
                                                            global_scale = data['flame_scale'][0])
                    hair_data = self.gaussianhair.generate(data)
                    
                    color = hair_data['color'][..., :3].view(1, self.gaussianhair.num_strands, 99, 3)
                    new_color = torch.tensor([1.0, 0.0, 0.0], device=color.device).view(1, 1, 1, 3)

                    color[:, highlight_strands_idx, :, :] = highlight_color
                    hair_data['color'][..., :3] = color.view(1, -1, 3)
                    # Set every 100th strand to new_color
                    # color[:, ::100, :, :] = torch.rand(color[:, ::100, :, :].shape[1], 3).unsqueeze(1).repeat(1, 100, 1).unsqueeze(0).to(color.device)

                    scales = hair_data['scales'].view(1, self.gaussianhair.num_strands, 99, 3)
                    scales[:, highlight_strands_idx, :, 1: ] = 100 * scales[:, highlight_strands_idx, :, 1: ]
                    hair_data['scales'] = scales.view(1, -1, 3)


                    hair_data['opacity'][...] = 0.0
                    opacity = hair_data['opacity'].view(1, self.gaussianhair.num_strands, 99, 1)
                    opacity[:, highlight_strands_idx, :, :] = 1.0
                    hair_data['opacity'] = opacity.view(1, -1, 1)

                    # combine head and hair data
                    for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                        # first dimension is batch size, concat along the second dimension
                        data[key] = hair_data[key]

                data = self.camera.render_gaussian(data, 2048)
                render_images = data['render_images'][: ,:3, ...]
                non_rigid_video.append(render_images[0].permute(1,2,0).clamp(0,1).cpu().numpy())


        only_rigid_video = []
        if self.gaussianhair is not None:
            with torch.no_grad():
                for i in tqdm(range(frame_num)):
                    
                    torch.cuda.empty_cache()
                    
                    data = dataset.__getitem__(i, self.camera_id)

                    # prepare data
                    for data_item in to_cuda:
                        data[data_item] = torch.tensor(data[data_item], device=self.device)
                        data[data_item] = data[data_item].unsqueeze(0)
                        data[data_item].requires_grad = False
                    
                    # data['poses_history'] = [None]
                    data['bg_rgb_color'] = torch.as_tensor([1.0, 1.0, 1.0]).cuda()
                    # TODO: select a few strands, color and enlarge them. Then render them

                    head_data = self.gaussianhead.generate(data)

                    self.gaussianhair.generate_hair_gaussians( poses_history = None, #poses_history = data['poses_history'][0], 
                                                            global_pose = data['flame_pose'][0], 
                                                            global_scale = data['flame_scale'][0])
                    hair_data = self.gaussianhair.generate(data)
                    
                    color = hair_data['color'][..., :3].view(1, self.gaussianhair.num_strands, 99, 3)
                    new_color = torch.tensor([1.0, 0.0, 0.0], device=color.device).view(1, 1, 1, 3)

                    color[:, highlight_strands_idx, :, :] = highlight_color
                    hair_data['color'][..., :3] = color.view(1, -1, 3)
                    # Set every 100th strand to new_color
                    # color[:, ::100, :, :] = torch.rand(color[:, ::100, :, :].shape[1], 3).unsqueeze(1).repeat(1, 100, 1).unsqueeze(0).to(color.device)

                    scales = hair_data['scales'].view(1, self.gaussianhair.num_strands, 99, 3)
                    scales[:, highlight_strands_idx, :, 1: ] = 100 * scales[:, highlight_strands_idx, :, 1: ]
                    hair_data['scales'] = scales.view(1, -1, 3)


                    hair_data['opacity'][...] = 0.0
                    opacity = hair_data['opacity'].view(1, self.gaussianhair.num_strands, 99, 1)
                    opacity[:, highlight_strands_idx, :, :] = 1.0
                    hair_data['opacity'] = opacity.view(1, -1, 1)

                    # combine head and hair data
                    for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                        # first dimension is batch size, concat along the second dimension
                        data[key] = hair_data[key]

                data = self.camera.render_gaussian(data, 2048)
                render_images = data['render_images'][: ,:3, ...]
                only_rigid_video.append(render_images[0].permute(1,2,0).clamp(0,1).cpu().numpy())



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

        if self.gaussianhair is not None:
            output_path = os.path.join("{}/{}/non_rigid_{}.mp4".format(self.recorder.checkpoint_path, self.recorder.name, self.camera_id))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (non_rigid_video[0].shape[1], non_rigid_video[0].shape[0]))
            for frame in non_rigid_video:
                frame = (frame*255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()
            print('Saved non-rigid deformation video to %s' % output_path)

            output_path = os.path.join("{}/{}/only_rigid_{}.mp4".format(self.recorder.checkpoint_path, self.recorder.name, self.camera_id))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (only_rigid_video[0].shape[1], only_rigid_video[0].shape[0]))
            for frame in only_rigid_video:
                frame = (frame*255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()
            print('Saved only rigid deformation video to %s' % output_path)

        # save head vertices
        # faces = self.gaussianhead.faces.cpu().numpy()
        faces = self.flame_model.faces.cpu().numpy()
        if len(head_vertices) != 0:
            head_vertices = np.stack(head_vertices, axis=0)
            np.savez(os.path.join("{}/{}/head_vertices_{}.npz".format(self.recorder.checkpoint_path ,self.recorder.name, self.camera_id)), vertices=head_vertices, faces=faces)
        
        
        # save hair strand points
        if len(hair_strand_points) != 0:
            hair_strand_points = np.stack(hair_strand_points, axis=0)
            hair_strand_points = hair_strand_points.reshape(hair_strand_points.shape[0], -1, 3)
            np.savez(os.path.join("{}/{}/hair_strand_points_{}.npz".format(self.recorder.checkpoint_path, self.recorder.name, self.camera_id)), points=hair_strand_points, color = hair_color)

        # save hair strand points posed
        if len(hair_strand_points_posed) != 0:
            hair_strand_points_posed = np.stack(hair_strand_points_posed, axis=0)
            hair_strand_points_posed = hair_strand_points_posed.reshape(hair_strand_points_posed.shape[0], -1, 3)
            np.savez(os.path.join("{}/{}/hair_strand_points_posed_{}.npz".format(self.recorder.checkpoint_path, self.recorder.name,self.camera_id)), points=hair_strand_points_posed, color = hair_color)

        print('Average loss_rgb: %.4f' % np.mean(loss_rgb_arr))
        print('Average loss_ssim: %.4f' % np.mean(loss_ssim_arr))
        print('Average psnr: %.4f' % np.mean(psnr_test_arr))
        print('Average ssim: %.4f' % np.mean(ssim_test_arr))
        print('Average hair psnr: %.4f' % np.mean(hair_psnr_test_arr))
        print('Average hair ssim: %.4f' % np.mean(hair_ssim_test_arr))

        print('Saved head vertices to %s' % os.path.join(self.recorder.checkpoint_path , self.recorder.name, 'head_vertices_{}.npz'.format(self.camera_id)))
        print('Saved hair strand points to %s' % os.path.join(self.recorder.checkpoint_path,self.recorder.name, 'hair_strand_points_{}.npz'.format(self.camera_id)))
        