from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
import cv2
import wandb
import open3d as o3d

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
            


class GaussianHeadTrainRecorder():
    def __init__(self, full_cfg):
        
        if full_cfg.recorder: 
            cfg = full_cfg.recorder
        else:
            cfg = full_cfg

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
        if self.logger:
            self.logger.add_scalar('loss_rgb_hr', log_data['loss_rgb_hr'], log_data['iter'])
            self.logger.add_scalar('loss_rgb_lr', log_data['loss_rgb_lr'], log_data['iter'])
            self.logger.add_scalar('loss_vgg', log_data['loss_vgg'], log_data['iter'])
        else:
            wandb.log({"loss_rgb_hr": log_data['loss_rgb_hr'], 
                       "loss_rgb_lr": log_data['loss_rgb_lr'], 
                       "loss_vgg": log_data['loss_vgg'],
                       "loss_segment": log_data['loss_segment'],
                       "loss_transform_reg": log_data['loss_transform_reg'],
                    #    "seg_label": log_data['gaussianhead'].seg_label.mean(),
                       "points_num": log_data['gaussianhead'].xyz.shape[0] })

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['gaussianhead'].state_dict(), '%s/%s/gaussianhead_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            torch.save(log_data['supres'].state_dict(), '%s/%s/supres_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['supres'].state_dict(), '%s/%s/supres_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            torch.save(log_data['delta_poses'], '%s/%s/delta_poses_latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['delta_poses'], '%s/%s/delta_poses_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))


        if log_data['iter'] % self.show_freq == 0:
            image = log_data['data']['images'][0].permute(1, 2, 0).detach().cpu().numpy()
            # [:,:,::-1] to convert RGB to BGR
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            render_image = log_data['data']['render_images'][0, 0:3].permute(1, 2, 0).detach().cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]

            cropped_image = log_data['data']['cropped_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            cropped_image = (cropped_image * 255).astype(np.uint8)[:,:,::-1]

            supres_image = log_data['data']['supres_images'][0].permute(1, 2, 0).detach().cpu().numpy()
            supres_image = (supres_image * 255).astype(np.uint8)[:,:,::-1]

            render_image = cv2.resize(render_image, (image.shape[0], image.shape[1]))
            result = np.hstack((image, render_image, cropped_image, supres_image))
            
            # too memory consuming, only used when reuiqring SIBR viewer
            # log_data['gaussianhair'].save_ply("%s/%s/%06d_hair.ply" % (self.checkpoint_path, self.name, log_data['iter']))
            # log_data['gaussianhead'].save_ply("%s/%s/%06d_head.ply" % (self.checkpoint_path, self.name, log_data['iter']))

            xyz = log_data['gaussianhair'].xyz.detach().cpu().numpy()
            hairmesh = o3d.geometry.TriangleMesh()
            hairmesh.vertices = o3d.utility.Vector3dVector(xyz)
            hairmesh.compute_vertex_normals() 
            o3d.io.write_triangle_mesh('%s/%s/%06d_hair.ply' % (self.result_path, self.name, log_data['iter']), hairmesh)

            xyz = log_data['gaussianhead'].xyz.detach().cpu().numpy()
            headmesh = o3d.geometry.TriangleMesh()
            headmesh.vertices = o3d.utility.Vector3dVector(xyz)
            headmesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh('%s/%s/%06d_head.ply' % (self.result_path, self.name, log_data['iter']), headmesh)


            if self.debug_tool == 'wandb':
                images = []

                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                images.append(wandb.Image(result, caption="rendered"))

                segment_vis = log_data['data']['gt_segment'][0]
                segment_vis[1] = torch.clamp(segment_vis[1] - segment_vis[2] , 0, 1)
                segment_vis = segment_vis.permute(1, 2, 0).detach().cpu().numpy()
                segment_vis = (segment_vis * 255).astype(np.uint8)
                images.append(wandb.Image(segment_vis, caption="Segmentation"))

                render_segment = log_data['data']['render_segments'][0].permute(1, 2, 0).detach().cpu().numpy()
                render_segment = (render_segment * 255).astype(np.uint8)
                images.append(wandb.Image(render_segment, caption="rendered_segment"))

                # append segment[2]
                images.append(wandb.Image(segment_vis[...,2], caption="hair mask"))

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