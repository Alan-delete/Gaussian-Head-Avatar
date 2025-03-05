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
                to_cuda = ['images', 'masks', 'visibles', 'images_coarse', 'masks_coarse', 'visibles_coarse', 
                           'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                           'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                images = data['images']
                visibles = data['visibles']
                if self.supres is None:
                    images_coarse = images
                    visibles_coarse = visibles
                else:
                    images_coarse = data['images_coarse']
                    visibles_coarse = data['visibles_coarse']

                resolution_coarse = images_coarse.shape[2]
                resolution_fine = images.shape[2]

                data['pose'] = data['pose'] + self.delta_poses[data['exp_id'], :]

                backprop_into_prior = iteration <= self.cfg.gaussianhairmodule.strands_reset_from_iter
                self.gaussianhair.generate_hair_gaussians(skip_smpl=iteration <= self.cfg.gaussianheadmodule.densify_from_iter, backprop_into_prior=backprop_into_prior)
                self.gaussianhair.update_learning_rate(iteration)

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

                # crop images for augmentation
                scale_factor = random.random() * 0.45 + 0.8
                scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
                data['cropped_images'] = cropped_images
                
                # generate super resolution images
                supres_images = self.supres(cropped_render_images) if self.supres else cropped_render_images
                data['supres_images'] = supres_images

                # loss functions
                loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)
                loss_rgb_hr = F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
                left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
                loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                            (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()
                loss = loss_rgb_hr + loss_rgb_lr + loss_vgg * 1e-1

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

                            if iteration >= self.cfg.gaussianheadmodule.densify_from_iter and iteration % 2000 == 0:
                                size_threshold = 20 if iteration > self.cfg.gaussianheadmodule.opacity_reset_interval else None
                                self.gaussianhead.densify_and_prune(self.cfg.gaussianheadmodule.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
                            
                            if iteration % self.cfg.gaussianheadmodule.opacity_reset_interval == 0 :
                                self.gaussianhead.reset_opacity()
                
                    # regenerate raw data from perm prior
                    if self.cfg.gaussianhairmodule.strands_reset_from_iter <= iteration <= self.cfg.gaussianhairmodule.strands_reset_until_iter \
                                                    and iteration % self.cfg.gaussianhairmodule.strands_reset_interval == 0: 
                        self.gaussianshair.reset_strands()

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
                    # TODO: shouln't it be xyz_raw?
                    params = [self.gaussianhair.xyz, 
                            self.gaussianhair.features_dc]
                    labels = ['xyz', 'features_dc']
                    if self.gaussianhair.train_features_rest:
                        params.append(self.gaussianhair.features_rest)
                        labels.append('features_rest')
                    if self.gaussianhair.train_opacity:
                        params.append(self.gaussianhair.opacity)
                        labels.append('opacity')
                    if self.gaussianhair.train_width:
                        params.append(self.gaussianhair.width)
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