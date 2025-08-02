from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
import cv2
import wandb
import open3d as o3d
import time
from matplotlib import pyplot as plt

class MeshHeadTrainRecorder():
    def __init__(self, cfg):
        self.logdir = cfg.logdir
        self.logger = SummaryWriter(self.logdir)

        self.name = cfg.name
        self.checkpoint_path = cfg.checkpoint_path
        self.result_path = cfg.result_path
        
        self.save_freq = cfg.save_freq
        self.show_freq = cfg.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    
    def log(self, log_data):
        self.logger.add_scalar('loss_rgb', log_data['loss_rgb'], log_data['iter'])
        self.logger.add_scalar('loss_sil', log_data['loss_sil'], log_data['iter'])
        self.logger.add_scalar('loss_def', log_data['loss_def'], log_data['iter'])
        self.logger.add_scalar('loss_offset', log_data['loss_offset'], log_data['iter'])
        self.logger.add_scalar('loss_lmk', log_data['loss_lmk'], log_data['iter'])
        self.logger.add_scalar('loss_lap', log_data['loss_lap'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['meshhead'].state_dict(), '%s/%s/meshhead_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['meshhead'].state_dict(), '%s/%s/meshhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))

        if log_data['iter'] % self.show_freq == 0:
            image = log_data['data']['images'][0, 0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            render_image = log_data['data']['render_images'][0, 0, :, :, 0:3].detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]

            render_normal = log_data['data']['render_normals'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
            render_normal = (render_normal * 255).astype(np.uint8)[:,:,::-1]

            render_image = cv2.resize(render_image, (render_image.shape[0], render_image.shape[1]))
            render_normal = cv2.resize(render_normal, (render_image.shape[0], render_image.shape[1]))
            result = np.hstack((image, render_image, render_normal))
            cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), result)
            
# def vis_orientation(rad, mask):
#     red = np.clip(1 - np.abs(rad -  0.) / 45., a_min=0, a_max=1) + np.clip(1 - np.abs(rad - 180.) / 45., a_min=0, a_max=1)
#     green = np.clip(1 - np.abs(rad - 90.) / 45., a_min=0, a_max=1)
#     magenta = np.clip(1 - np.abs(rad - 45.) / 45., a_min=0, a_max=1)
#     teal = np.clip(1 - np.abs(rad - 135.) / 45., a_min=0, a_max=1)
#     rgb = (
#         np.array([0, 0, 1])[None, None] * red[..., None] +
#         np.array([0, 1, 0])[None, None] * green[..., None] +
#         np.array([1, 0, 1])[None, None] * magenta[..., None] +
#         np.array([1, 1, 0])[None, None] * teal[..., None]
#     )
#     # norm = (r + g + b) * 0.5
#     # b = np.zeros_like(r)
#     norm = np.ones_like(rgb[..., 0])

#     vis_img = np.clip(rgb / norm[..., None], a_min=0, a_max=1) * mask[..., None] * 255
#     return vis_img

def vis_orient(orient_angle, mask):
    device = orient_angle.device
    deg = orient_angle * 180
    red = torch.clamp(1 - torch.abs(deg -  0.) / 45., 0, 1) + torch.clamp(1 - torch.abs(deg - 180.) / 45., 0, 1) # vertical
    green = torch.clamp(1 - torch.abs(deg - 90.) / 45., 0, 1) # horizontal
    magenta = torch.clamp(1 - torch.abs(deg - 45.) / 45., 0, 1) # diagonal down
    teal = torch.clamp(1 - torch.abs(deg - 135.) / 45., 0, 1) # diagonal up
    bgr = (
        torch.tensor([0, 0, 1])[:, None, None].to(device) * red +
        torch.tensor([0, 1, 0])[:, None, None].to(device) * green +
        torch.tensor([1, 0, 1])[:, None, None].to(device) * magenta +
        torch.tensor([1, 1, 0])[:, None, None].to(device) * teal
    )
    rgb = torch.stack([bgr[2], bgr[1], bgr[0]], dim=0)

    return rgb * mask


def hair_strand_rendering(data, gaussianhead, gaussianhair, camera, iteration = 1e6):

    device = data['images'].device

    if gaussianhair is not None:
        # highlight_strands_idx = torch.arange(0, gaussianhair.num_strands, 300, device= device)
        highlight_strands_idx = torch.arange(0, gaussianhair.num_strands, 1, device= device)
        gen = torch.Generator(device=device)
        gen.manual_seed(77)
        highlight_color = torch.rand(highlight_strands_idx.shape[0], 3, generator=gen, device=device).unsqueeze(1).repeat(1, gaussianhair.strand_length -1, 1).unsqueeze(0)


    # data['poses_history'] = [None]
    data['bg_rgb_color'] = torch.as_tensor([1.0, 1.0, 1.0]).cuda()
    # TODO: select a few strands, color and enlarge them. Then render them
    with torch.no_grad():
        head_data = gaussianhead.generate(data)
        if gaussianhair is not None:
            backprop = iteration < 8000
            # backprop = True 
            gaussianhair.generate_hair_gaussians(poses_history = data['poses_history'][0], 
                                                    # global_pose = init_flame_pose[0],
                                                    backprop_into_prior = backprop,
                                                    global_pose = data['flame_pose'][0], 
                                                    global_scale = data['flame_scale'][0])
            hair_data = gaussianhair.generate(data)
                    
            color = hair_data['color'][..., :3].view(1, gaussianhair.num_strands, gaussianhair.strand_length -1, 3)
            new_color = torch.tensor([1.0, 0.0, 0.0], device=color.device).view(1, 1, 1, 3)

            color[:, highlight_strands_idx, :, :] = highlight_color
            hair_data['color'][..., :3] = color.view(1, -1, 3)
            # Set every 100th strand to new_color
            # color[:, ::100, :, :] = torch.rand(color[:, ::100, :, :].shape[1], 3).unsqueeze(1).repeat(1, 100, 1).unsqueeze(0).to(color.device)

            scales = hair_data['scales'].view(1, gaussianhair.num_strands, gaussianhair.strand_length -1, 3)
            scales[:, highlight_strands_idx, :, 1: ] = 10 * scales[:, highlight_strands_idx, :, 1: ]
            hair_data['scales'] = scales.view(1, -1, 3)


            hair_data['opacity'][...] = 0.0
            opacity = hair_data['opacity'].view(1, gaussianhair.num_strands, gaussianhair.strand_length -1, 1)
            opacity[:, highlight_strands_idx, :, :] = 1.0
            hair_data['opacity'] = opacity.view(1, -1, 1)

            # combine head and hair data
            for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                # first dimension is batch size, concat along the second dimension
                # data[key] = hair_data[key]
                data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

        data = camera.render_gaussian(data, 2048)
        render_images = data['render_images'][: ,:3, ...]
        return render_images[0].permute(1,2,0).clamp(0,1).cpu().numpy()




class GaussianHeadTrainRecorder():
    def __init__(self, full_cfg, test_dataloader=None):
        
        if full_cfg.recorder: 
            cfg = full_cfg.recorder
        else:
            cfg = full_cfg
        
        self.test_dataloader = test_dataloader

        self.debug_tool = cfg.debug_tool
        self.logdir = cfg.logdir

        # choose the debug tool, either tensorboard or wandb
        self.logger = SummaryWriter(self.logdir) if self.debug_tool == 'tensorboard' else None
        wandb_name = "GaussianHead_%s" % cfg.name
        wandb.init(
            mode="disabled" if self.debug_tool=='tensorboard' else None,
            name=wandb_name,
            project='Semester',
            config= full_cfg ,
            settings=wandb.Settings(start_method='fork'),
        )
        wandb.save('lib/module/*.py', policy='now')
        wandb.save('lib/dataset/*.py', policy='now')
        wandb.save('lib/recorder/*.py', policy='now')
        wandb.save('lib/trainer/*.py', policy='now')
        wandb.save('config/*.py', policy='now')
        wandb.save('config/*.yaml', policy='now')
        wandb.save('preprocess/*.py', policy='now')
        wandb.save('lib/face_models/*.py', policy='now')
        wandb.save('train*.py', policy='now')
        wandb.save('run_video.sh', policy='now')
        wandb.save('install.sh', policy='now')

        self.name = cfg.name
        self.checkpoint_path = cfg.checkpoint_path
        self.result_path = cfg.result_path
        
        self.save_freq = cfg.save_freq
        self.show_freq = cfg.show_freq

        self.cfg = full_cfg

        # camera view idx for debugging
        self.debug_view = [1,25]

        # current time as random seed
        self.random_seq = str(int(time.time())) + str(np.random.randint(0, 100))

        # save the video of rendered images
        self.images = []

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    
    def log(self, log_data):

        if self.logger:
            self.logger.add_scalar('loss_rgb_hr', log_data['loss_rgb_hr'], log_data['iter'])
            self.logger.add_scalar('loss_rgb_lr', log_data['loss_rgb_lr'], log_data['iter'])
            self.logger.add_scalar('loss_vgg', log_data['loss_vgg'], log_data['iter'])
        else:
            log_record = {key: log_data[key] for key in log_data if key.startswith('loss_')}
            log_record['psnr_train'] = log_data['psnr_train'] if "psnr_train" in log_data else 0
            log_record['ssim_train'] = log_data['ssim_train'] if "ssim_train" in log_data else 0
            log_record['points_num'] = log_data['gaussianhead'].get_xyz.shape[0] if 'gaussianhead' in log_data and log_data['gaussianhead'] is not None else 0
            wandb.log(log_record)
        
        # save more frequently for debugging
        if log_data['iter'] % 500 == 0 and 'pre_shift_head' in log_data:    
            path = '%s/%s/pre_shift_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch'])
            torch.save({'epoch': log_data['epoch'], 
                        'pre_shift_head': log_data['pre_shift_head'],
                        'pre_shift_hair': log_data['pre_shift_hair']}, path)
            print('save pre_shift to path: %s' % path)

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')

            if 'supres' in log_data and log_data['supres'] is not None:
                torch.save(log_data['supres'].state_dict(), '%s/%s/supres_latest_%s' % (self.checkpoint_path, self.name, self.random_seq))
                torch.save(log_data['supres'].state_dict(), '%s/%s/supres_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
            
            if 'delta_poses' in log_data and log_data['delta_poses'] is not None:
                torch.save(log_data['delta_poses'], '%s/%s/delta_poses_latest_%s' % (self.checkpoint_path, self.name, self.random_seq))
                torch.save(log_data['delta_poses'], '%s/%s/delta_poses_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
            

            if 'gaussianhair' in log_data and log_data['gaussianhair'] is not None:
                torch.save(log_data['gaussianhair'].state_dict(), '%s/%s/gaussianhair_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
                torch.save(log_data['gaussianhair'].state_dict(), '%s/%s/gaussianhair_latest_%s' % (self.checkpoint_path, self.name, self.random_seq))
                
                # log_data['gaussianhair'].save_ply("%s/%s/%06d_hair.ply" % (self.checkpoint_path, self.name, log_data['iter']))
                # log_data['gaussianhair'].save_ply("%s/%s/hair_latest_%s.ply" % (self.checkpoint_path, self.name, self.random_seq))
                
                print('save gaussianhair to path: %s/%s/gaussianhair_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
                print('save gaussianhair to path: %s/%s/gaussianhair_latest_%s' % (self.checkpoint_path, self.name, self.random_seq))

            # TODO: also render the flame
            if 'gaussianhead' in log_data and log_data['gaussianhead'] is not None:
                # torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
                torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
                torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_latest_%s' % (self.checkpoint_path, self.name, self.random_seq))
                
                log_data['gaussianhead'].save_ply("%s/%s/%06d_head.ply" % (self.checkpoint_path, self.name, log_data['iter']))
                log_data['gaussianhead'].save_ply("%s/%s/head_latest_%s.ply" % (self.checkpoint_path, self.name, self.random_seq))

                # print('save gaussianhead to path: %s/%s/gaussianhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
                print('save gaussianhead to path: %s/%s/gaussianhead_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
                print('save gaussianhead to path: %s/%s/gaussianhead_latest_%s' % (self.checkpoint_path, self.name, self.random_seq))
                print('save gaussianhead to path: %s/%s/head_latest_%s.ply' % (self.checkpoint_path, self.name, self.random_seq))
                
                
        if log_data['iter'] % self.show_freq == 0:

            data = log_data['data']

            if self.test_dataloader is not None:
                # already inside the torhc.no_grad() context
                dataset = self.test_dataloader.dataset
                
                data = next(iter(self.tese_dataloader)) 
                to_cuda = ['images', 'masks', 'hair_masks','visibles', 'images_coarse', 'masks_coarse','hair_masks_coarse', 'visibles_coarse', 
                           'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                           'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id', 'fovx', 'fovy', 'orient_angle']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].cuda()

                images = data['images']
                visibles = data['visibles']
                images_coarse = data['images_coarse']
                visibles_coarse = data['visibles_coarse']


                resolution_coarse = images_coarse.shape[2]
                resolution_fine = images.shape[2]

                image = images[0] * visibles[0]
                image_coarse = images_coarse[0] * visibles_coarse[0]

                # render coarse images
                head_data = log_data['gaussianhead'].generate(data)
                hair_data = log_data['gaussianhair'].generate(data)
                # combine head and hair data
                for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                    # first dimension is batch size, concat along the second dimension
                    data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

                data = log_data['camera'].render_gaussian(data, resolution_coarse)


            images = data['images']
            visibles = data['visibles']
            images_coarse = data['images_coarse']
            visibles_coarse = data['visibles_coarse']

            image = images[0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            images_coarse = images_coarse[0].permute(1, 2, 0).detach().cpu().numpy()
            images_coarse = (images_coarse * 255).astype(np.uint8)[:,:,::-1]


            render_image = data['render_images'][0, 0:3].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]
            render_image = cv2.resize(render_image, (image.shape[0], image.shape[1]))

            cropped_image = data['cropped_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            cropped_image = (cropped_image * 255).astype(np.uint8)[:,:,::-1]

            supres_image = data['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            supres_image = (supres_image * 255).astype(np.uint8)[:,:,::-1]

            
            result = np.hstack((image, render_image, cropped_image, supres_image))
            # result = np.hstack((image, render_image))


            if self.debug_tool == 'wandb':
                images = []

                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = cv2.resize(image, (render_image.shape[0], render_image.shape[1]))
                # images.append(wandb.Image(image, caption="gt_image_supres"))

                if 'gaussianhead' in log_data and log_data['gaussianhead'] is not None and hasattr(log_data['gaussianhead'], 'verts'):
                    head_vertices_world_per_frame = log_data['gaussianhead'].verts
                    faces = log_data['gaussianhead'].faces
                    
                    # # 4*4
                    # transform = data['full_proj_transform'][0] #.cpu().numpy()
                    # breakpoint()
                    # vertices_hom_world = torch.cat([head_vertices_world_per_frame, torch.ones_like(head_vertices_world_per_frame[:, :1])], dim=1)
                    # verts_ndc = vertices_hom_world @ transform.T 
                    # verts_ndc /= verts_ndc[:, 3:4]
                    # transformed_vertices = head_vertices_world_per_frame.reshape(-1, 3).detach().cpu().numpy() @ transform[:3, :3].T + transform[:3, 3]

                    # render the mesh
                    # mesh = trimesh.Trimesh(vertices= head_vertices_world_per_frame.reshape(-1, 3).detach().cpu().numpy(), faces=faces.cpu().numpy())

                    # project the mesh to the image according to the camera parameters
                    # mesh.apply_transform(data['full_proj_transform'][0].cpu().numpy())
                    # visualize the gaussian head
                    # mesh = o3d.geometry.TriangleMesh()
                    # mesh.vertices = o3d.utility.Vector3dVector(head_vertices_world_per_frame.reshape(-1, 3).detach().cpu().numpy())
                    # mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
                    # import open3d.visualization.rendering as rendering
                    # render = rendering.OffscreenRenderer(render_image.shape[0], render_image.shape[1])
                    # render.scene.add_geometry("mesh", mesh)
                    # render.setup_camera(60, 
                    #                     data['camera_center'][0].cpu().numpy(),
                    #                     data['camera_center'][0].cpu().numpy() + data['extrinsics'][0, :3, 2].cpu().numpy(),
                    #                     data['extrinsics'][0, :3, 1].cpu().numpy())
                    # image = render.render_to_image()
                    # images.append(wandb.Image(image, caption="gaussian_head"))

                images_coarse = cv2.cvtColor(images_coarse, cv2.COLOR_BGR2RGB)
                images_coarse = cv2.resize(images_coarse, (render_image.shape[0], render_image.shape[1]))
                images.append(wandb.Image(images_coarse, caption="gt_image_coarse"))

                if 'supres' in log_data and log_data['supres'] is not None:
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    cropped_image = cv2.resize(cropped_image, (render_image.shape[0], render_image.shape[1]))
                    images.append(wandb.Image(cropped_image, caption="gt_cropped_image"))

                render_image = cv2.cvtColor(render_image, cv2.COLOR_BGR2RGB)
                render_image = cv2.resize(render_image, (render_image.shape[0], render_image.shape[1]))
                images.append(wandb.Image(render_image, caption="rendered_image_coarse"))

                if 'supres' in log_data and log_data['supres'] is not None:
                    supres_image = cv2.cvtColor(supres_image, cv2.COLOR_BGR2RGB)
                    supres_image = cv2.resize(supres_image, (render_image.shape[0], render_image.shape[1]))
                    images.append(wandb.Image(supres_image, caption="rendered_supres_image"))


                if 'visibles_coarse' in data:
                    visibles = data['visibles_coarse'][0].permute(1, 2, 0).detach().cpu().numpy()
                    visibles = (visibles * 255).astype(np.uint8)
                    visibles = cv2.resize(visibles, (image.shape[0], image.shape[1]))
                    images.append(wandb.Image(visibles, caption="visibility"))


                if 'gt_segment' in data:
                    segment_vis = data['gt_segment'][0]
                    segment_vis[1] = torch.clamp(segment_vis[1] - segment_vis[2] , 0, 1)
                    segment_vis = segment_vis.permute(1, 2, 0).detach().cpu().numpy()
                    segment_vis = (segment_vis * 255).astype(np.uint8)
                    segment_vis = cv2.resize(segment_vis, (image.shape[0], image.shape[1]))
                    images.append(wandb.Image(segment_vis, caption="Segmentation"))

                    render_segment = data['render_segments'][0].permute(1, 2, 0).detach().cpu().numpy()
                    render_segment = (render_segment * 255).astype(np.uint8)
                    render_segment = cv2.resize(render_segment, (image.shape[0], image.shape[1]))
                    images.append(wandb.Image(render_segment, caption="rendered_segment"))

                if 'hair_masks' in data:
                    mask = data['masks_coarse'][0]
                    hair_mask = data['hair_masks_coarse'][0].to(mask.device) 
                    ones_mask = torch.ones_like(mask)
                    images.append(wandb.Image(hair_mask.permute(1, 2, 0).detach().cpu().numpy(), caption="hair_mask"))

                    # if 'erode_conf' in data:
                    #     erode_conf = data['erode_conf'][0].permute(1, 2, 0).detach().cpu().numpy()
                    #     erode_conf = (erode_conf * 255).astype(np.uint8)
                    #     erode_conf = cv2.resize(erode_conf, (image.shape[0], image.shape[1]))
                    #     images.append(wandb.Image(erode_conf, caption="erode_conf"))


                    if 'orient_angle' in data:
                        orientation = data['orient_angle_coarse'][0].to(mask.device)
                        images.append(wandb.Image(vis_orient(orientation, hair_mask), caption="gt_orientation"))

                        render_orientation = data['render_orient'][0]
                        images.append(wandb.Image(vis_orient(render_orientation, hair_mask), caption="rendered_orientation"))

                        # images.append(wandb.Image(vis_orient(render_orientation, ones_mask), caption="global_rendered_orientation"))

                if 'depth' in data:
                    # depth is the 10th channel of the render_images
                    depth = data['render_images'][0, 9:10].permute(1, 2, 0).detach().cpu().numpy()
                    inverse_depth = 1 / (depth + 1e-6)  
                    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
                    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
                    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
                    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                        max_invdepth_vizu - min_invdepth_vizu
                    )
                    # Save as color-mapped "turbo" jpg image.
                    cmap = plt.get_cmap("turbo")
                    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                        np.uint8
                    )
                    images.append(wandb.Image(color_depth, caption="rendered_inverse_depth"))
                    
                    gt_depth = data['depth'][0].permute(1, 2, 0).detach().cpu().numpy()
                    inverse_depth_gt = 1 / (gt_depth + 1e-6)
                    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
                    max_invdepth_gt = min(inverse_depth_gt.max(), 1 / 0.1)
                    min_invdepth_gt = max(1 / 250, inverse_depth_gt.min())
                    inverse_depth_normalized_gt = (inverse_depth_gt - min_invdepth_gt) / (
                        max_invdepth_gt - min_invdepth_gt
                    )
                    # Save as color-mapped "turbo" jpg image.
                    color_depth_gt = (cmap(inverse_depth_normalized_gt)[..., :3] * 255).astype(
                        np.uint8
                    )
                    images.append(wandb.Image(color_depth_gt, caption="gt_inverse_depth"))


                hair_strand_image = hair_strand_rendering(data, log_data['gaussianhead'], log_data['gaussianhair'], log_data['camera'], log_data['iter'])
                self.images.append(hair_strand_image)

                # # save video
                # gt_video = self.images
                # # output_path = os.path.join("{}/{}/gt_{}.mp4".format(self.recorder.checkpoint_path, self.recorder.name, self.camera_id))
                # output_path = os.path.join(self.checkpoint_path, 'training_hair_strand_video.mp4' )
                # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (gt_video[0].shape[1], gt_video[0].shape[0]))
                # for frame in gt_video:
                #     frame = (frame*255).astype(np.uint8)
                #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #     out.write(frame)
                # out.release()
                # print('Saved video to %s' % output_path)


                images.append(wandb.Image(hair_strand_image, caption="rendered_hair_strand"))

                wandb.log({"Images": images})

            else:
                cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), result)



class ReenactmentRecorder():
    def __init__(self, cfg):
        self.name = cfg.name
        self.result_path = cfg.result_path
        
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)
    
    def log(self, log_data):
            image = log_data['data']['images'][0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            supres_image = log_data['data']['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            supres_image = (supres_image * 255).astype(np.uint8)[:,:,::-1]

            image = cv2.resize(image, (supres_image.shape[0], supres_image.shape[1]))
            result = np.hstack((image, supres_image))
            cv2.imwrite('%s/%s/%06d.jpg' % (self.result_path, self.name, log_data['iter']), result)