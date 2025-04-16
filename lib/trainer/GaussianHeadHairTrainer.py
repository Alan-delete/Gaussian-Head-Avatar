import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import lpips

import math
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average=True)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def or_loss(network_output, gt, confs = None, weight = None, mask = None):
    weight = torch.ones_like(gt[:1]) if weight is None else weight
    loss = torch.minimum(
        (network_output - gt).abs(),
        torch.minimum(
            (network_output - gt - 1).abs(), 
            (network_output - gt + 1).abs()
        ))
    if confs is not None:
        loss = loss * confs - (confs + 1e-7).log()    
    if mask is not None:
        loss = (loss * mask).sum() / mask.sum()
    return loss.mean()
    # if weight is not None:
    #     return (loss * weight).sum() / weight.sum()
    # else:
    #     return loss * weight


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
        iteration = -1
        for epoch in range(start_epoch, epochs):
            self.gaussianhair.epoch_start()
            # TODO: set time decay for frame at the begining of each epoch to learning more about the first frame
            for idx, data in tqdm(enumerate(self.dataloader)):

                iteration += 1
                iteration = idx + epoch * len(self.dataloader)

                # prepare data
                to_cuda = ['images', 'masks', 'hair_masks','visibles', 'images_coarse', 'masks_coarse','hair_masks_coarse', 'visibles_coarse', 
                           'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                           'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id', 'fovx', 'fovy', 'orient_angle', 'flame_pose', 'flame_scale','poses_history', 'optical_flow']
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
                
                if iteration == self.cfg.gaussianhairmodule.strands_reset_from_iter: 
                    self.gaussianhair.reset_strands()


                # if  4000 <= iteration <= 20000:
                #     if iteration % 2000 == 0: 
                #         self.gaussianhair.random_set_transparent(ratio=0.05)
                
                # before 4000, backprop into prior
                backprop_into_prior = iteration <= self.cfg.gaussianhairmodule.strands_reset_from_iter
               
                self.gaussianhair.update_learning_rate(iteration)
                # self.gaussianhead.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                # if iteration % 1000 == 0:
                #     self.gaussianhair.oneupSHdegree()

                # render coarse images
                head_data = self.gaussianhead.generate(data)
                self.gaussianhair.generate_hair_gaussians(skip_smpl=iteration <= self.cfg.gaussianheadmodule.densify_from_iter, 
                                                          backprop_into_prior=backprop_into_prior, 
                                                          poses_history = data['poses_history'][0], 
                                                          pose = data['pose'][0],
                                                          scale = data['scale'][0])
                hair_data = self.gaussianhair.generate(data)
                # combine head and hair data
                for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                    # first dimension is batch size, concat along the second dimension
                    data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

                    # DEBUG: only use head data
                    # data[key] = head_data[key] 

                data = self.camera.render_gaussian(data, resolution_coarse)
                
                #  default [4000, 15000], during that period, use strand raw data to rectify the prior 
                if self.cfg.gaussianhairmodule.strands_reset_from_iter <= iteration <= self.cfg.gaussianhairmodule.strands_reset_until_iter:
                    points, dirs, _, _ = self.gaussianhair.sample_strands_from_prior()
                    pred_pts = dirs if self.gaussianhair.train_directions else points

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
                gt_segment[:, 1] = torch.clamp(gt_segment[:, 1] - gt_segment[:, 2] , 0, 1)
                # gt_segment[:, 2] = torch.clamp(gt_segment[:, 2] - gt_segment[:, 1] , 0, 1)

                gt_orientation = data['orient_angle']
                pred_orientation = data['render_orient']
                # TODO: use pred_mask instead of gt_mask
                # orient_weight = torch.ones_like(gt_mask[:1]) * gt_orient_conf
                loss_orient = or_loss(pred_orientation , gt_orientation, mask=gt_hair_mask) #orient_conf, ) #, weight=orient_weight
                if torch.isnan(loss_orient).any(): loss_orient = 0.0

                # sharpness loss
                loss_opacity_reg = 0.005 * (self.gaussianhair.get_opacity * (1 - self.gaussianhair.get_opacity) ).mean()

                # B, C, H, W 
                render_segments = data["render_segments"]
                segment_clone = render_segments.clone()
                segment_clone[:,1] = render_segments[:,1] + render_segments[:,2]
                def l1_loss(a, b):
                    return (a - b).abs().mean()

                def recall_loss(gt, pred):
                    return (gt - pred).clamp(min=0).mean()
                
                def relax_recall_loss(gt, pred):
                    return (gt - pred).clamp(min=0).mean() + (pred - gt).clamp(min=0).mean() * 0.3
                
                # find that the hair segment is not well predicted, so add extra hair segment loss
                # loss_segment = l1_loss(segment_clone, gt_segment)  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                # loss_segment =  10 * l1_loss(segment_clone[:,2], gt_segment[:,2]) if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                
                # too few positive samples, reduce the penalty of false positive(when predicted value larger than gt value)
                loss_segment = (relax_recall_loss(gt_segment[:,2] * visibles_coarse, segment_clone[:,2] * visibles_coarse) )  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                # loss_segment = (relax_recall_loss(gt_segment * visibles_coarse, segment_clone * visibles_coarse) )  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                
                # step decay for segment loss
                if iteration > 5000:
                    decay_rate = 0.4 ** ( iteration // 5000)
                    decay_rate = max(decay_rate, 0.1)
                    loss_segment = loss_segment * decay_rate


                loss_transform_reg =  F.mse_loss(self.gaussianhair.init_transform, self.gaussianhair.transform) 

                
                # TODO: try mesh distance loss, also try knn color regularization
                loss_mesh_dist = self.gaussianhair.mesh_distance_loss()

                loss_knn_feature = self.gaussianhair.knn_feature_loss()

                loss_sign_distance = self.gaussianhair.sign_distance_loss()

                loss_strand_feature = self.gaussianhair.strand_feature_loss()
                
                gt_pts = self.gaussianhair.dir.detach() if self.gaussianhair.train_directions else self.gaussianhair.points.detach()
                loss_dir = l1_loss(pred_pts, gt_pts) if self.cfg.gaussianhairmodule.strands_reset_from_iter <= iteration <= self.cfg.gaussianhairmodule.strands_reset_until_iter else torch.zeros_like(loss_segment)


                # crop images for augmentation
                scale_factor = random.random() * 0.45 + 0.8
                scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
                data['cropped_images'] = cropped_images
                
                # generate super resolution images
                supres_images = self.supres(cropped_render_images) if self.cfg.use_supres else cropped_render_images[:,:3]
                data['supres_images'] = supres_images

                psnr_train = psnr(render_images * visibles, images * visibles)
                ssim_train = ssim(render_images * visibles, images * visibles)
                loss_ssim = 1.0 - ssim(render_images * visibles, images * visibles)

                # loss functions
                loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)
                loss_rgb_hr = F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
                left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
                loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                            (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()


                #  to better see the real magnitude of the loss
                loss_rgb_hr = loss_rgb_hr * self.cfg.loss_weights.rgb_hr
                loss_rgb_lr = loss_rgb_lr * self.cfg.loss_weights.rgb_lr
                loss_ssim = loss_ssim * self.cfg.loss_weights.dssim
                loss_vgg = loss_vgg * self.cfg.loss_weights.vgg
                loss_segment = loss_segment * self.cfg.loss_weights.segment
                loss_transform_reg = loss_transform_reg * self.cfg.loss_weights.transform_reg
                loss_dir = loss_dir * self.cfg.loss_weights.dir
                loss_mesh_dist = loss_mesh_dist * self.cfg.loss_weights.mesh_dist
                loss_knn_feature = loss_knn_feature * self.cfg.loss_weights.knn_feature
                loss_strand_feature = loss_strand_feature * self.cfg.loss_weights.strand_feature
                loss_sign_distance = loss_sign_distance * self.cfg.loss_weights.sign_distance
                loss = ( 
                        loss_rgb_hr +
                        loss_rgb_lr +
                        loss_ssim +
                        loss_vgg +
                        loss_segment +
                        loss_transform_reg +
                        loss_dir +
                        loss_mesh_dist +
                        loss_knn_feature +
                        loss_strand_feature +
                        loss_opacity_reg +   
                        loss_sign_distance
                )

                # breakpoint()
                # # draw renderimage
                # import cv2
                # img = render_images[0].cpu().detach().numpy()
                # img = img.transpose(1, 2, 0) * 255
                # img = img.astype('uint8')
                # cv2.imwrite(f'./debug_renderimage_{iteration}.png', img)

                loss.backward()
                # Optimizer step, for super
                # for param in self.optimizer.param_groups[0]['params']:
                #     if param.grad is not None and param.grad.isnan().any():
                #         self.optimizer.zero_grad()
                #         print(f'NaN during backprop in superres was found, skipping iteration...')
                # self.optimizer.step()

                for group in self.optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None and param.grad.isnan().any():
                            self.optimizer.zero_grad()
                            print(f'NaN during backprop in {group.get("name", "Unnamed")} was found, skipping iteration...')
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)

                # Optimizer step, for hair prior
                for k, optimizer in self.gaussianhair.prior_optimizers.items():
                    for param in optimizer.param_groups[0]['params']:
                        if param.grad is not None and param.grad.isnan().any():
                            optimizer.zero_grad()
                            print(f'NaN in prior during backprop was found, skipping iteration...')
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
                    # nan_grad = False
                    # params = [self.gaussianhead.xyz, 
                    #         self.gaussianhead.feature,
                    #         self.gaussianhead.features_dc, 
                    #         self.gaussianhead.features_rest, 
                    #         self.gaussianhead.opacity, 
                    #         self.gaussianhead.label_hair, 
                    #         self.gaussianhead.label_body, 
                    #         self.gaussianhead.scales, 
                    #         self.gaussianhead.rotation]
                    # labels = ['xyz', 'feature','features_dc', 'features_rest', 'opacity', 'label_hair', 'label_body', 'scaling', 'rotation']
                    # for param, label in zip(params, labels):
                    #     if param.grad is not None and param.grad.isnan().any():
                    #         nan_grad = True
                    #         print(f'NaN during backprop in {label} unstruct was found, skipping iteration')
                    
                    nan_grad = False
                    for group in self.gaussianhead.optimizer.param_groups:
                        for param in group['params']:
                            if param.grad is not None and param.grad.isnan().any():
                                nan_grad = True
                                print(f"[NaN Detected in Gradients] → Unstruct Param group: {group.get('name', 'Unnamed')}")
                            # if torch.isnan(param).any():
                            #     print(f"[NaN Detected in Weights] → Param group: {group.get('name', 'Unnamed')}") 
                    
                    if not nan_grad:
                        self.gaussianhead.optimizer.step()
                    self.gaussianhead.optimizer.zero_grad(set_to_none = True)
                        
                    # strctured
                    nan_grad = False
                    # params = [self.gaussianhair.points_raw, 
                    #         self.gaussianhair.features_dc_raw]
                    # labels = ['point', 'features_dc']
                    # if self.gaussianhair.train_features_rest:
                    #     params.append(self.gaussianhair.features_rest_raw)
                    #     labels.append('features_rest')
                    # if self.gaussianhair.train_opacity:
                    #     params.append(self.gaussianhair.opacity_raw)
                    #     labels.append('opacity')
                    # if self.gaussianhair.train_width:
                    #     params.append(self.gaussianhair.width_raw)
                    #     labels.append('width')

                    # for param in self.gaussianhair.optimizer.param_groups[0]['params']:
                    #     if param.grad is not None and param.grad.isnan().any():
                    #         optimizer.zero_grad()
                    #         print(f'NaN during backprop in {k} was found, skipping iteration...')
                    # optimizer.step()
                    # optimizer.zero_grad(set_to_none = True)
                    for group in self.gaussianhair.optimizer.param_groups:
                        for param in group['params']:
                            if param.grad is not None and param.grad.isnan().any():
                                nan_grad = True
                                print(f"[NaN Detected in Gradients] → Struct Param group: {group.get('name', 'Unnamed')}")
                            # if torch.isnan(param).any():
                            #     print(f"[NaN Detected in Weights] → Param group: {group.get('name', 'Unnamed')}")

                    # for param, label in zip(params, labels):
                    #     if param.grad is not None and param.grad.isnan().any():
                    #         nan_grad = True
                    #         print(f'NaN during backprop in {label} struct was found, skipping iteration')
                    if not nan_grad:
                        self.gaussianhair.optimizer.step()
                    self.gaussianhair.optimizer.zero_grad(set_to_none = True)

                log = {
                    'data': data,
                    'delta_poses' : self.delta_poses,
                    'gaussianhead' : self.gaussianhead,
                    'gaussianhair' : self.gaussianhair,
                    'camera' : self.camera,
                    'supres' : self.supres,
                    'loss_rgb_lr' : loss_rgb_lr,
                    'loss_rgb_hr' : loss_rgb_hr,
                    'loss_vgg' : loss_vgg,
                    'loss_segment' : loss_segment,
                    'loss_transform_reg' : loss_transform_reg,
                    'loss_dir' : loss_dir,
                    'loss_mesh_dist' : loss_mesh_dist,
                    'loss_knn_feature' : loss_knn_feature,
                    'loss_ssim' : loss_ssim,
                    'loss_orient' : loss_orient,
                    'psnr_train' : psnr_train,
                    'ssim_train' : ssim_train,
                    'loss_strand_feature' : loss_strand_feature,
                    'loss_sign_distance' : loss_sign_distance,
                    'epoch' : epoch,
                    # 'iter' : idx + epoch * len(self.dataloader)
                    'iter' : iteration
                }
                with torch.no_grad():
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