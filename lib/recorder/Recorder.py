from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
import cv2
import wandb
import open3d as o3d
import time

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

            if 'gaussianhead' in log_data and log_data['gaussianhead'] is not None:
                # torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
                torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
                torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_latest_%s' % (self.checkpoint_path, self.name, self.random_seq))
                
                log_data['gaussianhead'].save_ply("%s/%s/%06d_head.ply" % (self.checkpoint_path, self.name, log_data['iter']))
                log_data['gaussianhead'].save_ply("%s/%s/head_latest_%s.ply" % (self.checkpoint_path, self.name, self.random_seq))

                # print('save gaussianhead to path: %s/%s/gaussianhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
                print('save gaussianhead to path: %s/%s/gaussianhead_iter_%d' % (self.checkpoint_path, self.name, log_data['iter']))
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
                images_coarse = images
                visibles_coarse = visibles

                resolution_coarse = images_coarse.shape[2]
                resolution_fine = images.shape[2]

                # render coarse images
                head_data = log_data['gaussianhead'].generate(data)
                hair_data = log_data['gaussianhair'].generate(data)
                # combine head and hair data
                for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                    # first dimension is batch size, concat along the second dimension
                    data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

                data = log_data['camera'].render_gaussian(data, resolution_coarse)



            image = data['images'][0].permute(1, 2, 0).detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            images_coarse = data['images_coarse'][0].permute(1, 2, 0).detach().cpu().numpy()
            images_coarse = (images_coarse * 255).astype(np.uint8)[:,:,::-1]

            resolution_fine = image.shape[0]
            resolution_coarse = images_coarse.shape[0]

            render_image = data['render_images'][0, 0:3].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]

            # cropped_image = data['cropped_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            # cropped_image = (cropped_image * 255).astype(np.uint8)[:,:,::-1]

            # supres_image = data['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            # supres_image = (supres_image * 255).astype(np.uint8)[:,:,::-1]

            render_image = cv2.resize(render_image, (image.shape[0], image.shape[1]))
            # result = np.hstack((image, render_image, cropped_image, supres_image))
            result = np.hstack((image, render_image))


            if self.debug_tool == 'wandb':
                images = []

                # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                # result = cv2.resize(result, (image.shape[0] * 2, image.shape[1]))
                # images.append(wandb.Image(result, caption="rendered"))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (render_image.shape[0], render_image.shape[1]))
                images.append(wandb.Image(image, caption="gt_image_supres"))

                images_coarse = cv2.cvtColor(images_coarse, cv2.COLOR_BGR2RGB)
                images_coarse = cv2.resize(images_coarse, (render_image.shape[0], render_image.shape[1]))
                images.append(wandb.Image(images_coarse, caption="gt_image_coarse"))

                render_image = cv2.cvtColor(render_image, cv2.COLOR_BGR2RGB)
                render_image = cv2.resize(render_image, (render_image.shape[0], render_image.shape[1]))
                images.append(wandb.Image(render_image, caption="rendered_image_coarse"))

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
                    images.append(wandb.Image(hair_mask.permute(1, 2, 0).detach().cpu().numpy(), caption="hair_mask"))


                    if 'orient_angle' in data:
                        orientation = data['orient_angle_coarse'][0].to(mask.device)
                        images.append(wandb.Image(vis_orient(orientation, hair_mask), caption="gt_orientation"))

                        render_orientation = data['render_orient'][0]
                        images.append(wandb.Image(vis_orient(render_orientation, hair_mask), caption="rendered_orientation"))

                    if 'optical_flow' in data:
                        # [2, resolution, resolution]
                        optical_flow = data['optical_flow_coarse'][0].to(mask.device)
                        ones_mask = torch.ones_like(mask)
                        # angle = torch.atan2(optical_flow[1], optical_flow[0]) * 180 / np.pi
                        angle = torch.atan2(optical_flow[1], optical_flow[0]) / np.pi
                        images.append(wandb.Image(vis_orient(angle, mask), caption="optical_flow"))


                        # estimated_flow_numpy = (data['optical_flow'][0] * mask).permute(1, 2, 0).detach().cpu().numpy()
                        # warped_query_image = remap_using_flow_fields(image, estimated_flow_numpy[:, :, 0],
                        #                              estimated_flow_numpy[:, :, 1]).astype(np.uint8)
                        # images.append(wandb.Image(warped_query_image, caption="warped_image"))

                    if 'render_velocity' in data:
                        # render_velocity = data['render_velocity'][0].permute(1, 2, 0).detach().cpu().numpy() 
                        # render_velocity = (render_velocity * 255).astype(np.uint8)
                        # render_velocity = cv2.resize(render_velocity, (image.shape[0], image.shape[1]))
                        # images.append(wandb.Image(render_velocity, caption="rendered_velocity"))
                        render_velocity = data['render_velocity'][0]
                        # angle = torch.atan2(render_velocity[1], render_velocity[0]) * 180 / np.pi
                        angle = torch.atan2(render_velocity[1], render_velocity[0]) / np.pi
                        images.append(wandb.Image(vis_orient(angle, mask), caption="rendered_velocity"))
                    
                    if 'optical_flow_confidence' in data:
                        optical_flow_confidence = data['optical_flow_confidence_coarse'][0].to(mask.device)
                        images.append(wandb.Image(optical_flow_confidence.permute(1, 2, 0).detach().cpu().numpy(), caption="optical_flow_confidence"))


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