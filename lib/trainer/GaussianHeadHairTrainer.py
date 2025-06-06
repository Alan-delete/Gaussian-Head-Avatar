import numpy as np
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
        # iteration = start_epoch * len(self.dataloader) * 128
        # iteration = 24001
        iteration = 1
        end_iteration = iteration + 50000
        dataset = self.dataloader.dataset
        static_training_util_iter =  self.cfg.static_training_util_iter if self.cfg.static_scene_init else 0
        
        # prepare data
        to_cuda = ['images', 'masks', 'hair_masks','visibles', 'images_coarse', 'masks_coarse','hair_masks_coarse', 'visibles_coarse', 
                    'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                    'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id', 'fovx', 'fovy', 'orient_angle', 'flame_pose', 'flame_scale','poses_history', 'optical_flow','optical_flow_confidence',
                    'optical_flow_coarse','optical_flow_confidence_coarse', 'orient_angle_coarse']

        self.gaussianhair.epoch_start()
        self.gaussianhair.frame_start()

        if self.cfg.static_scene_init:

            epoch = start_epoch
            for _ in tqdm(range(iteration, static_training_util_iter)):

                # if this iteration is to save, use predefined view
                if iteration % self.recorder.show_freq == 0:
                    # view 22 or 25 are good for visualization
                    i = np.random.choice([0, 25])
                    data = dataset.__getitem__(0, i)
                else:
                    data = dataset[0]

                # prepare data
                for data_item in to_cuda:
                    data[data_item] = torch.as_tensor(data[data_item], device=self.device)
                    data[data_item] = data[data_item].unsqueeze(0)

                # # disable deformation util the last 2000 iterations to initialize deformation
                # if iteration < 10000:
                #     data['poses_history'] = [None]
                data['poses_history'] = [None]

                self.train_step(iteration, epoch, data)
                iteration += 1
                
            # disable the training of gaussian, just focus on the deformer
            if self.gaussianhair is not None:
                self.gaussianhair.disable_static_parameters()
            if self.gaussianhead is not None:
                self.gaussianhead.disable_static_parameters()
            print('Disable static training, start dynamic training')


        for epoch in range(start_epoch, epochs):
 
            self.gaussianhair.epoch_start()

            if iteration > end_iteration:
                break

            # for idx, data in tqdm(enumerate(range(len(dataset)))):
            for idx in (range(len(dataset))):

                self.gaussianhair.frame_start()
                for _ in tqdm(range(128)):

                    if iteration % self.recorder.show_freq == 0:
                        # random pick the view from back(1) and front(25)
                        i = np.random.choice([0, 25])
                        data = dataset.__getitem__(idx, i)
                        
                        for data_item in to_cuda:
                            data[data_item] = torch.as_tensor(data[data_item], device=self.device)
                            data[data_item] = data[data_item].unsqueeze(0)

                    else:
                        data = dataset[idx]

                        for data_item in to_cuda:
                            data[data_item] = torch.as_tensor(data[data_item], device=self.device)
                            data[data_item] = data[data_item].unsqueeze(0)

                    for data_item in to_cuda:
                        data[data_item] = data[data_item].to(device=self.device)
                    
                    self.train_step(iteration, epoch, data, grad_accumulation = 1)
                    iteration += 1



    def train_step(self, iteration, epoch, data, grad_accumulation = 1):


        images = data['images']
        visibles = data['visibles']
        # we have predicted label [background, body, hair], the gt_segment now is atcully [background, body & hair, hair]
        # gt_hair_mask = data['hair_masks']
        # gt_mask = data['masks']
        gt_hair_mask = data['hair_masks_coarse']
        gt_mask = data['masks_coarse']

        # if self.cfg.use_supres:
        #     images_coarse = data['images_coarse']
        #     visibles_coarse = data['visibles_coarse']
        # else:
        #     images_coarse = images
        #     visibles_coarse = visibles
        
        images_coarse = data['images_coarse']
        # only learn hair rgb
        # images_coarse = images_coarse * gt_hair_mask
        visibles_coarse = data['visibles_coarse']

        resolution_coarse = images_coarse.shape[2]
        resolution_fine = images.shape[2]

        data['pose'] = data['pose'] + self.delta_poses[data['exp_id'], :]
        
        
        B = data['pose'].shape[0]
        if self.gaussianhead is not None:
            # self.gaussianhead.update_learning_rate(iteration)
            head_data = self.gaussianhead.generate(data)

            if self.cfg.train_optical_flow and data['poses_history'].shape[1] >= 2:
                # TODO: get direction 2d from xyz and direction
                velocity2D = []
                pre_exp_coeff = data['exp_coeff'] 
                pre_pose = data['poses_history'][:, -2]
                pre_scale = data['scale']
                pre_head = self.gaussianhead.get_posed_points( exp_coeff = pre_exp_coeff, pose = pre_pose, scale = pre_scale)
                optical_flow_head = self.gaussianhead.xyz - pre_head['xyz']
                for b in range(B):
                    image_height = resolution_coarse
                    image_width = resolution_coarse
                    # TODO, velocity should not be normalized
                    velocity2D.append(self.gaussianhair.get_direction_2d(data['fovx'][b], data['fovy'][b], 
                                                            image_height, image_width,
                                                            data['world_view_transform'][b], 
                                                            head_data['xyz'][b], optical_flow_head[b], normalize=False))
                velocity2D = torch.stack(velocity2D, dim=0)
                head_data['color'][...,  9:12] = velocity2D


        if self.gaussianhair is not None:
            
            if iteration == self.cfg.gaussianhairmodule.strands_reset_from_iter: 
                self.gaussianhair.reset_strands()

            # before 4000, backprop into prior
            backprop_into_prior = iteration <= self.cfg.gaussianhairmodule.strands_reset_from_iter
        
            self.gaussianhair.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            # if iteration % 1000 == 0:
            #     self.gaussianhair.oneupSHdegree()

            self.gaussianhair.generate_hair_gaussians(skip_smpl=iteration <= self.cfg.gaussianheadmodule.densify_from_iter, 
                                                    backprop_into_prior=backprop_into_prior, 
                                                    poses_history = data['poses_history'][0], 
                                                    global_pose = data['flame_pose'][0],
                                                    global_scale = data['flame_scale'][0])
            hair_data = self.gaussianhair.generate(data)
            
            if self.cfg.train_optical_flow and data['poses_history'].shape[1] >= 2:

                # TODO: get direction 2d from xyz and direction
                velocity2D = []
                pose_history = data['poses_history'][0][:-1]
                pre_pose = pose_history[-1]
                pre_scale = data['scale'][0]
                pre_strand = self.gaussianhair.get_posed_points(poses_history = pose_history, pose = pre_pose, scale = pre_scale)
                optical_flow_hair = self.gaussianhair.points.reshape(-1,3) - pre_strand['points'].reshape(-1, 3)
                optical_flow_hair = optical_flow_hair.unsqueeze(0)
                for b in range(B):
                    image_height = resolution_coarse
                    image_width = resolution_coarse
                    # TODO, velocity should not be normalized
                    velocity2D.append(self.gaussianhair.get_direction_2d(data['fovx'][b], data['fovy'][b], 
                                                            image_height, image_width,
                                                            data['world_view_transform'][b], 
                                                            hair_data['xyz'][b], optical_flow_hair[b], normalize=False))
                velocity2D = torch.stack(velocity2D, dim=0)
                hair_data['color'][...,  9:12] = velocity2D


        if self.gaussianhair is not None and self.gaussianhead is not None:
            # combine head and hair data
            for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                # first dimension is batch size, concat along the second dimension
                data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)
        elif self.gaussianhead is not None:
            # only head data
            for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                # first dimension is batch size, concat along the second dimension
                data[key] = head_data[key]
        elif self.gaussianhair is not None:
            # only hair data
            for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                # first dimension is batch size, concat along the second dimension
                data[key] = hair_data[key]


        data = self.camera.render_gaussian(data, resolution_coarse)
        
        viewspace_point_tensor = data["viewspace_points"]
        visibility_filter = data["visibility_filter"]
        radii = data["radii"]
        render_images = data['render_images']
        

        # gt_mask = torch.maximum(gt_mask, gt_hair_mask)
        # gt_segment = torch.cat([ 1 - gt_mask, gt_mask, gt_hair_mask], dim=1) 
        # or
        # gt_segment = torch.cat([ 1 - gt_mask, gt_mask - gt_hair_mask, gt_hair_mask], dim=1) 

        # TODO: don't forget to change dim if removing batch dimension
        binary_hair_mask = gt_hair_mask > 0.1
        gt_body_mask = (~binary_hair_mask) * gt_mask
        gt_background_mask = torch.clamp(1 - gt_body_mask - gt_hair_mask, min=0, max=1)
        gt_segment = torch.cat([ gt_background_mask , gt_body_mask, gt_hair_mask], dim=1) 
        # normalize the gt_segment to [0, 1]
        gt_segment = gt_segment / (gt_segment.sum(dim=1, keepdim=True) + 1e-6)
        
        data['gt_segment'] = gt_segment


        render_segments = data["render_segments"]

        loss_segment = 0
        loss_orient = 0
        loss_optical_flow = 0
        loss_rgb_lr = 0
        loss_rgb_hr = 0
        loss_vgg = 0
        loss_ssim = 0
        loss_transform_reg = 0
        loss_dir = 0
        loss_elastic = 0
        loss_sign_distance = 0
        loss_landmarks = 0
        loss_mesh_dist = 0
        loss_knn_feature = 0
        loss_strand_feature = 0
        loss_flame_gaussian_reg = 0
        loss_deform_reg = 0


        # sgement loss
        # B, C, H, W 

        segment_clone = render_segments.clone()
        segment_clone[:,1] = render_segments[:,1] + render_segments[:,2]
        def l1_loss(a, b):
            return (a - b).abs().mean()

        def recall_loss(gt, pred):
            return (gt - pred).clamp(min=0).mean()

        def relax_recall_loss(gt, pred):
            return (gt - pred).clamp(min=0).mean() + (pred - gt).clamp(min=0).mean() * 0.7
        
        # too few positive samples, reduce the penalty of false positive(when predicted value larger than gt value)
        loss_segment = (relax_recall_loss(gt_segment[:,2] * visibles_coarse, segment_clone[:,2] * visibles_coarse))  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
        # loss_segment = (l1_loss(gt_segment * visibles_coarse, render_segments * visibles_coarse) )  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
        
        # step decay for segment loss
        if iteration > 20000:
            decay_rate = 0.6 ** ( iteration // 20000)
            decay_rate = max(decay_rate, 0.4)
            loss_segment = loss_segment * decay_rate



        intersect_body_mask = gt_mask * segment_clone[:, 1].detach()
        intersect_hair_mask = gt_hair_mask * segment_clone[:, 2].detach()



        if self.cfg.train_optical_flow and data['poses_history'].shape[1] >= 2 and iteration > 7000:
            gt_optical_flow = data['optical_flow_coarse']
            gt_optical_flow_confidence = data['optical_flow_confidence_coarse']
            pred_optical_flow = data['render_velocity']
            gt_optical_flow = gt_optical_flow * intersect_body_mask
            pred_optical_flow = pred_optical_flow * intersect_body_mask
            loss_optical_flow = ( (pred_optical_flow - gt_optical_flow) ** 2 * gt_optical_flow_confidence).mean() * 0.01
            # TODO: use 3d optical flow or 2d optical flow projection?
            loss_optical_flow_hair_reg = optical_flow_hair.norm(2).mean() * 0.1
            loss_optical_flow_head_reg = optical_flow_head.norm(2).mean() * 0.1
        else:
            loss_optical_flow = 0
            loss_optical_flow_hair_reg = 0
            loss_optical_flow_head_reg = 0



        
        # TODO: try mesh distance loss, also try knn color regularization
        if self.gaussianhair is not None:

            gt_orientation = data['orient_angle_coarse']
            pred_orientation = data['render_orient']

            intersect_hair_mask = gt_hair_mask * segment_clone[:, 2].detach()

            # TODO: use pred_mask instead of gt_mask
            # orient_weight = torch.ones_like(gt_mask[:1]) * gt_orient_conf
            loss_orient = or_loss(pred_orientation , gt_orientation, mask = intersect_hair_mask) #orient_conf, ) #, weight=orient_weight
            if torch.isnan(loss_orient).any(): loss_orient = 0.0

            loss_sign_distance = self.gaussianhair.sign_distance_loss()

            loss_transform_reg = 0 #F.mse_loss(self.gaussianhair.init_transform, self.gaussianhair.transform) 

            loss_mesh_dist = 0 #self.gaussianhair.mesh_distance_loss()

            loss_knn_feature = 0 # self.gaussianhair.knn_feature_loss()

            loss_strand_feature = 0 #self.gaussianhair.strand_feature_loss()

            loss_deform_reg = self.gaussianhair.deform_regularization_loss()


            if True or backprop_into_prior:
                loss_elastic = 0
            else:
                loss_elastic = self.gaussianhair.elastic_potential_loss(data['poses_history'][0])
        

            #  default [4000, 15000], during that period, use strand raw data to rectify the prior 
            if  self.cfg.gaussianhairmodule.strands_reset_from_iter <= iteration <= self.cfg.gaussianhairmodule.strands_reset_until_iter:
                points, dirs, _, _ = self.gaussianhair.sample_strands_from_prior()
                pred_pts = dirs if self.gaussianhair.train_directions else points
            gt_pts = self.gaussianhair.dir.detach() if self.gaussianhair.train_directions else self.gaussianhair.points.detach()
            loss_dir = l1_loss(pred_pts, gt_pts) if self.cfg.gaussianhairmodule.strands_reset_from_iter <= iteration <= self.cfg.gaussianhairmodule.strands_reset_until_iter else torch.zeros_like(loss_segment)
        

        # landmark loss
        if self.cfg.flame_gaussian_module.enable:
            gt_landmarks_3d = data['landmarks_3d']
            pred_landmarks_3d = self.gaussianhead.landmarks
            pred_landmarks_3d = torch.cat([pred_landmarks_3d[:, 0:48], pred_landmarks_3d[:, 49:54], pred_landmarks_3d[:, 55:68]], 1)
            loss_landmarks = F.mse_loss(pred_landmarks_3d, gt_landmarks_3d, reduction='mean') * 100.


        # crop images for augmentation
        scale_factor = random.random() * 0.45 + 0.8
        scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
        cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
        data['cropped_images'] = cropped_images
        
        # generate super resolution images
        supres_images = self.supres(cropped_render_images) if self.cfg.use_supres else cropped_images
        data['supres_images'] = supres_images

        # visibles_coarse = visibles_coarse * intersect_hair_mask

        psnr_train = psnr(render_images[:, 0:3, :, :]  * visibles_coarse, images_coarse * visibles_coarse)
        ssim_train = ssim(render_images[:, 0:3, :, :]  * visibles_coarse, images_coarse * visibles_coarse)
        loss_ssim = 1.0 - ssim(render_images[:, 0:3, :, :]  * visibles_coarse, images_coarse * visibles_coarse)

        # loss functions
        loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)

        if self.cfg.use_supres:
            loss_rgb_hr = F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
            left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
            loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                        (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()
        else:
            loss_rgb_hr = loss_rgb_lr
            # loss_vgg = self.fn_lpips((render_images[:,:3] * visibles_coarse), images_coarse * visibles_coarse, normalize=True).mean()


        if self.cfg.flame_gaussian_module.enable:
            unstrct_gaussian_num = self.gaussianhead.get_xyz.shape[0]
            visibility_filter_unstruct = visibility_filter[0][:unstrct_gaussian_num]
            # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
            # loss_flame_gaussian_reg += F.relu(self.gaussianhead._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz
            loss_flame_gaussian_reg += F.relu(self.gaussianhead._xyz[visibility_filter_unstruct].norm(dim=1) - self.cfg.flame_gaussian_module.threshold_xyz).mean() * self.cfg.flame_gaussian_module.lambda_xyz

            # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
            # loss_flame_gaussian_reg += F.relu(torch.exp(self.gaussianhead._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
            loss_flame_gaussian_reg += F.relu(torch.exp(self.gaussianhead._scaling[visibility_filter_unstruct]) - self.cfg.flame_gaussian_module.threshold_scale).norm(dim=1).mean() * self.cfg.flame_gaussian_module.lambda_scale

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
        loss_orient = loss_orient * self.cfg.loss_weights.orient
        loss_deform_reg = loss_deform_reg * self.cfg.loss_weights.deform_reg
        # in static training, keep the deformation small
        if iteration < self.cfg.static_training_util_iter:
            loss_deform_reg = loss_deform_reg * 50.

        # random_point = torch.as_tensor([0,0,0]).unsqueeze(0).to(self.device) 
        # strand_end_points = self.gaussianhair.get_strand_points_posed.view(-1, 100, 3)[:, -1 , :]
        # loss_debug = (strand_end_points - random_point).norm(2, dim=1).mean() * 1e3

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
                loss_orient +
                loss_sign_distance + 
                loss_landmarks +
                loss_elastic +
                loss_optical_flow + 
                loss_flame_gaussian_reg +
                loss_deform_reg 
        )

        loss = loss / grad_accumulation
        loss.backward()

        log = {
            'data': data,
            'delta_poses' : self.delta_poses if self.cfg.optimize_pose else None,
            'gaussianhead' : self.gaussianhead,
            'gaussianhair' : self.gaussianhair,
            'camera' : self.camera,
            'supres' : self.supres if self.cfg.use_supres else None,
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
            'loss_optical_flow' : loss_optical_flow,
            'loss_landmarks' : loss_landmarks,
            'loss_elastic' : loss_elastic,
            'loss_flame_gaussian_reg' : loss_flame_gaussian_reg,
            'loss_deform_reg' : loss_deform_reg,
            # 'loss_debug' : loss_debug,
            'epoch' : epoch,
            # 'iter' : idx + epoch * len(self.dataloader)
            'iter' : iteration
        }
        with torch.no_grad():
            self.recorder.log(log)


            # gradient accumulation
            if iteration % grad_accumulation != 0:
                return

            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None and param.grad.isnan().any():
                        self.optimizer.zero_grad()
                        print(f'NaN during backprop in {group.get("name", "Unnamed")} was found, skipping iteration...')
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)

            # Optimizer step, for hair prior
            if self.gaussianhair is not None:
                for k, optimizer in self.gaussianhair.prior_optimizers.items():
                    for param in optimizer.param_groups[0]['params']:
                        if param.grad is not None and param.grad.isnan().any():
                            optimizer.zero_grad()
                            print(f'NaN in prior during backprop was found, skipping iteration...')
                    optimizer.step()
                    optimizer.zero_grad(set_to_none = True)


            # unstructured
            if self.gaussianhead is not None:
                # Densification
                opt = self.cfg.flame_gaussian_module if self.cfg.flame_gaussian_module.enable else self.cfg.gaussianheadmodule
            
                if opt.densify:
                    # TODO: By printing the value of Gaussian Hair cut. Need to get this value in this project
                    cameras_extent = 4.907987451553345
                    if iteration <= opt.densify_until_iter :
                        # Keep track of max radii in image-space for pruning
                        unstrct_gaussian_num = self.gaussianhead.get_xyz.shape[0]
                        # [B, total_gaussian_num] -> [unstruct_gaussian_num]
                        visibility_filter = visibility_filter[0][:unstrct_gaussian_num]
                        radii = radii[0][:unstrct_gaussian_num]
                        self.gaussianhead.max_radii2D[visibility_filter] = torch.max(self.gaussianhead.max_radii2D[visibility_filter], radii[visibility_filter])
                        self.gaussianhead.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            self.gaussianhead.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
                        
                        if iteration % opt.opacity_reset_interval == 0 :
                            self.gaussianhead.reset_opacity()


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
            # regenerate raw data from perm prior
            if self.gaussianhair is not None:
                if self.cfg.gaussianhairmodule.strands_reset_from_iter <= iteration <= self.cfg.gaussianhairmodule.strands_reset_until_iter \
                                                and iteration % self.cfg.gaussianhairmodule.strands_reset_interval == 0: 
                    self.gaussianhair.reset_strands()
                nan_grad = False
                for group in self.gaussianhair.optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None and param.grad.isnan().any():
                            nan_grad = True
                            print(f"[NaN Detected in Gradients] → Struct Param group: {group.get('name', 'Unnamed')}")
                if not nan_grad:
                    self.gaussianhair.optimizer.step()
                self.gaussianhair.optimizer.zero_grad(set_to_none = True)


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