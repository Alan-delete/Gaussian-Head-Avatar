import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import lpips


class GaussianHeadHairTrainer():
    def __init__(self, dataloader, delta_poses, gaussianhead, gaussianhair,supres, camera, optimizer, recorder, gpu_id, cfg = None):
        self.dataloader = dataloader
        self.delta_poses = delta_poses
        self.gaussianhead = gaussianhead
        self.gaussianhair = gaussianhair
        self.supres = supres
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)
        # TODO: check which member of cfg is used in the code and only pass those
        self.cfg = cfg

    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):
                
                iteration = idx + epoch * len(self.dataloader)
                # prepare data
                to_cuda = ['images', 'masks', 'hair_masks','visibles', 'images_coarse', 'masks_coarse','hair_masks_coarse', 'visibles_coarse', 
                           'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                           'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                images = data['images']
                visibles = data['visibles']
                if self.cfg.use_supres:
                    images_coarse = data['images_coarse']
                    visibles_coarse = data['visibles_coarse']
                else:
                    images_coarse = images
                    visibles_coarse = visibles

                resolution_coarse = images_coarse.shape[2]
                resolution_fine = images.shape[2]

                data['pose'] = data['pose'] + self.delta_poses[data['exp_id'], :]

                backprop_into_prior = iteration <= self.cfg.gaussianhairmodule.strands_reset_from_iter
                self.gaussianhair.generate_hair_gaussians(skip_smpl=iteration <= self.cfg.gaussianheadmodule.densify_from_iter, backprop_into_prior=backprop_into_prior)
                # self.gaussianhair.generate_hair_gaussians(skip_smpl=iteration <= self.cfg.gaussianheadmodule.densify_from_iter, 
                #                                           backprop_into_prior=backprop_into_prior, 
                #                                           pose_params= data['pose'][0])
                self.gaussianhair.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                # if iteration % 1000 == 0:
                #     self.gaussianhair.oneupSHdegree()

                # render coarse images
                head_data = self.gaussianhead.generate(data)
                hair_data = self.gaussianhair.generate(data)
                # combine head and hair data
                for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                    # first dimension is batch size, concat along the second dimension
                    data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

                data = self.camera.render_gaussian(data, resolution_coarse)
                viewspace_point_tensor = data["viewspace_points"]
                visibility_filter = data["visibility_filter"]
                radii = data["radii"]
                render_images = data['render_images']
                # we have predicted label [background, body, hair], the gt_segment now is atcully [background, body & hair, hair]
                gt_hair_mask = data['hair_masks']
                gt_mask = data['masks']
                # TODO: don't forget to change dim if removing batch dimension
                gt_segment = torch.cat([ 1 - gt_mask, gt_mask, gt_hair_mask], dim=1) 
                data['gt_segment'] = gt_segment
                
                # B, C, H, W 
                render_segments = data["render_segments"]
                segment_clone = render_segments.clone()
                segment_clone[:,1] = render_segments[:,1] + render_segments[:,2]
                def l1_loss(a, b):
                    return (a - b).abs().mean()

                def recall_loss(gt, pred):
                    return (gt - pred).clamp(min=0).mean()
                
                def relax_recall_loss(gt, pred):
                    return (gt - pred).clamp(min=0).mean() + 0.05 * (pred - gt).clamp(min=0).mean()
                
                # find that the hair segment is not well predicted, so add extra hair segment loss
                # loss_segment = l1_loss(segment_clone, gt_segment)  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                # loss_segment =  10 * l1_loss(segment_clone[:,2], gt_segment[:,2]) if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                
                # too few positive samples, reduce the penalty of false positive(when predicted value larger than gt value)
                loss_segment =  25 * relax_recall_loss(gt_segment[:,2], segment_clone[:,2]) if self.cfg.train_segment else torch.tensor(0.0, device=self.device)

                loss_transform_reg = 0.1 * F.mse_loss(self.gaussianhair.init_transform, self.gaussianhair.transform) if iteration < 1000 else 0

                # crop images for augmentation
                scale_factor = random.random() * 0.45 + 0.8
                scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
                data['cropped_images'] = cropped_images
                
                # generate super resolution images
                supres_images = self.supres(cropped_render_images) if self.cfg.use_supres else cropped_render_images[:,:3]
                data['supres_images'] = supres_images

                # loss functions
                loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)
                loss_rgb_hr = F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
                left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
                loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                            (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()
                loss = loss_rgb_hr + loss_rgb_lr + loss_vgg * 1e-1 + loss_segment * 5e-1 + loss_transform_reg * 1e-1

                self.optimizer.zero_grad()
                loss.backward()
                # Optimizer step, for super
                for param in self.optimizer.param_groups[0]['params']:
                    if param.grad is not None and param.grad.isnan().any():
                        self.optimizer.zero_grad()
                        print(f'NaN during backprop in {iteration} was found, skipping iteration...')
                self.optimizer.step()

                # Optimizer step, for hair prior
                for k, optimizer in self.gaussianhair.prior_optimizers.items():
                    for param in optimizer.param_groups[0]['params']:
                        if param.grad is not None and param.grad.isnan().any():
                            optimizer.zero_grad()
                            print(f'NaN during backprop in {k} was found, skipping iteration...')
                    optimizer.step()
                    optimizer.zero_grad(set_to_none = True)


                with torch.no_grad():
                    # Densification
                    if self.cfg.gaussianheadmodule.densify:
                        # TODO: By printing the value of Gaussian Hair cut. Need to get this value in this project
                        cameras_extent = 4.907987451553345
                        if iteration <= self.cfg.gaussianheadmodule.densify_until_iter :
                            # Keep track of max radii in image-space for pruning
                            unstrct_gaussian_num = self.gaussianhead.xyz.shape[0]
                            # [B, total_gaussian_num] -> [unstruct_gaussian_num]
                            visibility_filter = visibility_filter[0][:unstrct_gaussian_num]
                            radii = radii[0][:unstrct_gaussian_num]
                            self.gaussianhead.max_radii2D[visibility_filter] = torch.max(self.gaussianhead.max_radii2D[visibility_filter], radii[visibility_filter])
                            self.gaussianhead.add_densification_stats(viewspace_point_tensor, visibility_filter)

                            if iteration >= self.cfg.gaussianheadmodule.densify_from_iter and iteration % self.cfg.gaussianheadmodule.densification_interval == 0:
                                size_threshold = 20 if iteration > self.cfg.gaussianheadmodule.opacity_reset_interval else None
                                self.gaussianhead.densify_and_prune(self.cfg.gaussianheadmodule.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
                            
                            # if iteration % self.cfg.gaussianheadmodule.opacity_reset_interval == 0 :
                            #     self.gaussianhead.reset_opacity()
                
                    # regenerate raw data from perm prior
                    if self.cfg.gaussianhairmodule.strands_reset_from_iter <= iteration <= self.cfg.gaussianhairmodule.strands_reset_until_iter \
                                                    and iteration % self.cfg.gaussianhairmodule.strands_reset_interval == 0: 
                        self.gaussianhair.reset_strands()

                    # Optimizer step, for gaussians
                    # unstrctured
                    nan_grad = False
                    params = [self.gaussianhead.xyz, 
                            self.gaussianhead.features_dc, 
                            self.gaussianhead.features_rest, 
                            self.gaussianhead.opacity, 
                            self.gaussianhead.label_hair, 
                            self.gaussianhead.label_body, 
                            self.gaussianhead.scales, 
                            self.gaussianhead.rotation]
                    labels = ['xyz', 'features_dc', 'features_rest', 'opacity', 'label_hair', 'label_body', 'scaling', 'rotation']
                    for param, label in zip(params, labels):
                        if param.grad is not None and param.grad.isnan().any():
                            nan_grad = True
                            print(f'NaN during backprop in {label} unstruct was found, skipping iteration')
                    if not nan_grad:
                        self.gaussianhead.optimizer.step()
                    self.gaussianhead.optimizer.zero_grad(set_to_none = True)
                        
                    # strctured
                    nan_grad = False
                    params = [self.gaussianhair.points_raw, 
                            self.gaussianhair.features_dc_raw]
                    labels = ['point', 'features_dc']
                    if self.gaussianhair.train_features_rest:
                        params.append(self.gaussianhair.features_rest_raw)
                        labels.append('features_rest')
                    if self.gaussianhair.train_opacity:
                        params.append(self.gaussianhair.opacity_raw)
                        labels.append('opacity')
                    if self.gaussianhair.train_width:
                        params.append(self.gaussianhair.width_raw)
                        labels.append('width')
                    for param, label in zip(params, labels):
                        if param.grad is not None and param.grad.isnan().any():
                            nan_grad = True
                            print(f'NaN during backprop in {label} struct was found, skipping iteration')
                    if not nan_grad:
                        self.gaussianhair.optimizer.step()
                    self.gaussianhair.optimizer.zero_grad(set_to_none = True)

                log = {
                    'data': data,
                    'delta_poses' : self.delta_poses,
                    'gaussianhead' : self.gaussianhead,
                    'gaussianhair' : self.gaussianhair,
                    'supres' : self.supres,
                    'loss_rgb_lr' : loss_rgb_lr,
                    'loss_rgb_hr' : loss_rgb_hr,
                    'loss_vgg' : loss_vgg,
                    'loss_segment' : loss_segment,
                    'loss_transform_reg' : loss_transform_reg,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)


    def random_crop(self, render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine):
        render_images_scaled = F.interpolate(render_images, scale_factor=scale_factor)
        images_scaled = F.interpolate(images, scale_factor=scale_factor)
        visibles_scaled = F.interpolate(visibles, scale_factor=scale_factor)

        if scale_factor < 1:
            render_images = torch.ones([render_images_scaled.shape[0], render_images_scaled.shape[1], resolution_coarse, resolution_coarse], device=self.device)
            left_up_coarse = (random.randint(0, resolution_coarse - render_images_scaled.shape[2]), random.randint(0, resolution_coarse - render_images_scaled.shape[3]))
            render_images[:, :, left_up_coarse[0]: left_up_coarse[0] + render_images_scaled.shape[2], left_up_coarse[1]: left_up_coarse[1] + render_images_scaled.shape[3]] = render_images_scaled

            images = torch.ones([images_scaled.shape[0], images_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            visibles = torch.ones([visibles_scaled.shape[0], visibles_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images[:, :, left_up_fine[0]: left_up_fine[0] + images_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + images_scaled.shape[3]] = images_scaled
            visibles[:, :, left_up_fine[0]: left_up_fine[0] + visibles_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + visibles_scaled.shape[3]] = visibles_scaled
        else:
            left_up_coarse = (random.randint(0, render_images_scaled.shape[2] - resolution_coarse), random.randint(0, render_images_scaled.shape[3] - resolution_coarse))
            render_images = render_images_scaled[:, :, left_up_coarse[0]: left_up_coarse[0] + resolution_coarse, left_up_coarse[1]: left_up_coarse[1] + resolution_coarse]

            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images = images_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
            visibles = visibles_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
        
        return render_images, images, visibles