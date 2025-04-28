import os
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


class OpticalFlowTrainer():
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


        pre_shift_checkpoint_path = "checkpoints/gaussianhead_renderme_single/pre_shift_epoch_15"
        # pre_shift_checkpoint_path = "checkpoints/gaussianhead_renderme_single/pre_shift_epoch_version2_12"
        if False and os.path.exists(pre_shift_checkpoint_path):
            checkpoint = torch.load(pre_shift_checkpoint_path)
            start_frame = checkpoint['epoch'] + 1
            pre_shift_head = checkpoint['pre_shift_head']
            pre_shift_hair = checkpoint['pre_shift_hair']
            pre_shift_head.requires_grad = False
            pre_shift_hair.requires_grad = False
            init_data = dataset[0]
            for data_item in to_cuda:
                init_data[data_item] = torch.tensor(init_data[data_item], device=self.device)
                init_data[data_item] = init_data[data_item].unsqueeze(0)
            print("load pre_shift from checkpoint, start from ", start_frame)
        else:
            start_frame = 10
            pre_shift_head = torch.zeros([1, self.gaussianhead.xyz.shape[0], 3], device=self.device, requires_grad=False)
            pre_shift_hair = torch.zeros([1, self.gaussianhair.num_strands * (self.gaussianhair.strand_length - 1) , 3], device=self.device, requires_grad=False)
            init_data = dataset[start_frame-1]
            for data_item in to_cuda:
                init_data[data_item] = torch.tensor(init_data[data_item], device=self.device)
                init_data[data_item] = init_data[data_item].unsqueeze(0)
            print("checkpoint not found, start from ", start_frame)
        

        for i in range(start_frame, len(dataset)):
            
            torch.cuda.empty_cache()
            
            # optical_flow_head = torch.zeros([1, self.gaussianhead.xyz.shape[0], 3], device=self.device, requires_grad=False)
            optical_flow_head = torch.zeros([1, self.gaussianhead.xyz.shape[0], 3], device=self.device, requires_grad=True)
            # optical_flow_hair = torch.zeros([1, self.gaussianhair.num_strands * (self.gaussianhair.strand_length - 1) , 3], device=self.device, requires_grad=False)
            optical_flow_hair = torch.zeros([1, self.gaussianhair.num_strands * (self.gaussianhair.strand_length - 1) , 3], device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([optical_flow_hair, optical_flow_head], lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
            
            for _ in tqdm(range(3000)):
                iteration += 1

                # if this iteration is to save, use predefined view
                if iteration % self.recorder.show_freq == 0:
                    # view 22 or 25 are good for visualization
                    data = dataset.__getitem__(i, 25)
                else:
                    data = dataset[i]

                # prepare data
                for data_item in to_cuda:
                    data[data_item] = torch.tensor(data[data_item], device=self.device)
                    data[data_item] = data[data_item].unsqueeze(0)
                    data[data_item].requires_grad = False

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

                # use the first frame(init_data) to generate data

                # render coarse images

                head_data = self.gaussianhead.generate(init_data)
                # get previous accumulated shift
                head_data['xyz'] = head_data['xyz'] + pre_shift_head
                head_data['xyz'] = head_data['xyz'] + optical_flow_head

                # TODO: get direction 2d from xyz and direction
                velocity2D = []
                for b in range(1):
                    image_height = data['images'].shape[2]
                    image_width = data['images'].shape[3]
                    # TODO, velocity should not be normalized
                    velocity2D.append(self.gaussianhair.get_direction_2d(data['fovx'][b], data['fovy'][b], 
                                                            image_height, image_width,
                                                            data['world_view_transform'][b], 
                                                            head_data['xyz'][b], optical_flow_head[b], normalize=False))
                velocity2D = torch.stack(velocity2D, dim=0)
                head_data['color'][...,  9:12] = velocity2D


                self.gaussianhair.generate_hair_gaussians(skip_smpl=True, 
                                                        backprop_into_prior=False, 
                                                        poses_history = init_data['poses_history'][0], 
                                                        global_pose = init_data['pose'][0],
                                                        global_scale = init_data['scale'][0],
                                                        given_optical_flow = optical_flow_hair[0],
                                                        accumulate_optical_flow = pre_shift_hair[0])
                hair_data = self.gaussianhair.generate(init_data)

                # TODO: get direction 2d from xyz and direction

                dir = self.gaussianhair.dir.unsqueeze(0)
                dir2D = []
                velocity2D = []
                for b in range(1):
                    image_height = data['images'].shape[2]
                    image_width = data['images'].shape[3]
                    dir2D.append(self.gaussianhair.get_direction_2d(data['fovx'][b], data['fovy'][b],
                                                            image_height, image_width,
                                                            data['world_view_transform'][b],
                                                            hair_data['xyz'][b], dir[b]))
                    # TODO, velocity should not be normalized
                    velocity2D.append(self.gaussianhair.get_direction_2d(data['fovx'][b], data['fovy'][b], 
                                                            image_height, image_width,
                                                            data['world_view_transform'][b], 
                                                            hair_data['xyz'][b], optical_flow_hair[b], normalize=False))
                velocity2D = torch.stack(velocity2D, dim=0)
                dir2D = torch.stack(dir2D, dim=0)
                hair_data['color'][...,  6:9] = dir2D
                hair_data['color'][...,  9:12] = velocity2D



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
                gt_segment[:, 1] = torch.clamp(gt_segment[:, 1] - gt_segment[:, 2] , 0, 1)
                # gt_segment[:, 2] = torch.clamp(gt_segment[:, 2] - gt_segment[:, 1] , 0, 1)

                # B, C, H, W 
                render_segments = data["render_segments"]
                segment_clone = render_segments.clone()
                segment_clone[:,1] = render_segments[:,1] + render_segments[:,2]

                
                def relax_recall_loss(gt, pred):
                    return (gt - pred).clamp(min=0).mean() + (pred - gt).clamp(min=0).mean() * 0.3
                
                # find that the hair segment is not well predicted, so add extra hair segment loss
                # loss_segment = l1_loss(segment_clone, gt_segment)  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                # loss_segment =  10 * l1_loss(segment_clone[:,2], gt_segment[:,2]) if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                
                # too few positive samples, reduce the penalty of false positive(when predicted value larger than gt value)
                loss_segment = 0
                # loss_segment = (relax_recall_loss(gt_segment[:,2] * visibles_coarse, segment_clone[:,2] * visibles_coarse) )  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
                # loss_segment = (relax_recall_loss(gt_segment * visibles_coarse, segment_clone * visibles_coarse) )  if self.cfg.train_segment else torch.tensor(0.0, device=self.device)
            
                intersect_body_mask = gt_mask * segment_clone[:, 1].detach()
                intersect_hair_mask = gt_hair_mask * segment_clone[:, 2].detach()

                def optical_flow_loss(gt, pred):
                    # direction loss
                    direction_loss = 1 - F.cosine_similarity(gt, pred, dim=1).mean()
                    # magnitude loss
                    magnitude_loss = F.mse_loss(gt.norm(dim=1), pred.norm(dim=1), reduction='mean') * 0.001
                    # l1 loss
                    l1_loss = F.l1_loss(pred_optical_flow, gt_optical_flow, reduction='mean') * 10

                    return direction_loss + l1_loss


                gt_optical_flow = data['optical_flow']
                gt_optical_flow_confidence = data['optical_flow_confidence']
                pred_optical_flow = data['render_velocity']
                gt_optical_flow = gt_optical_flow * intersect_body_mask
                pred_optical_flow = pred_optical_flow * intersect_body_mask

                
                # loss_optical_flow = F.mse_loss(pred_optical_flow, gt_optical_flow, reduction='mean') 
                loss_optical_flow = ( (pred_optical_flow - gt_optical_flow) ** 2 * gt_optical_flow_confidence).mean()* 0.1 
                # loss_optical_flow = F.l1_loss(pred_optical_flow, gt_optical_flow, reduction='mean')* 0.5 
                # loss_optical_flow = optical_flow_loss(gt_optical_flow, pred_optical_flow)
                
                # TODO: use 3d optical flow or 2d optical flow projection?
                loss_optical_flow_hair_reg = optical_flow_hair.norm(2).mean() * 0.1
                loss_optical_flow_head_reg = optical_flow_head.norm(2).mean() * 0.1

# 
                gt_orientation = data['orient_angle']
                pred_orientation = data['render_orient']
                # TODO: use pred_mask instead of gt_mask
                # orient_weight = torch.ones_like(gt_mask[:1]) * gt_orient_conf
                loss_orient = or_loss(pred_orientation , gt_orientation, mask=intersect_hair_mask) #orient_conf, ) #, weight=orient_weight
                if torch.isnan(loss_orient).any(): loss_orient = 0.0


                loss_transform_reg = 0
                
                # TODO: try mesh distance loss, also try knn color regularization
                # loss_mesh_dist = self.gaussianhair.mesh_distance_loss()
                loss_mesh_dist = 0 

                # loss_knn_feature = self.gaussianhair.knn_feature_loss()
                loss_knn_feature = 0  

                # loss_strand_feature = self.gaussianhair.strand_feature_loss()
                loss_strand_feature = 0
                

                # crop images for augmentation
                scale_factor = random.random() * 0.45 + 0.8
                scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
                data['cropped_images'] = cropped_images
                
                # generate super resolution images
                supres_images = self.supres(cropped_render_images) if self.cfg.use_supres else cropped_render_images[:,:3]
                data['supres_images'] = supres_images

                psnr_train = psnr(render_images * visibles, images * visibles)
                ssim_train = 0
                # ssim_train = ssim(render_images * visibles, images * visibles)
                loss_ssim = 0
                # loss_ssim = 1.0 - ssim(render_images * visibles, images * visibles)

                # loss functions
                loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)
                loss_rgb_hr = 0
                # loss_rgb_hr = F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
                left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
                # loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                            # (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()
                loss_vgg = 0 



                #  to better see the real magnitude of the loss
                loss_rgb_hr = loss_rgb_hr * self.cfg.loss_weights.rgb_hr
                loss_rgb_lr = loss_rgb_lr * self.cfg.loss_weights.rgb_lr
                loss_ssim = loss_ssim * self.cfg.loss_weights.dssim
                loss_vgg = loss_vgg * self.cfg.loss_weights.vgg
                loss_segment = loss_segment * self.cfg.loss_weights.segment
                loss_transform_reg = loss_transform_reg * self.cfg.loss_weights.transform_reg
                loss_dir = 0
                loss_mesh_dist = loss_mesh_dist * self.cfg.loss_weights.mesh_dist
                loss_knn_feature = loss_knn_feature * self.cfg.loss_weights.knn_feature
                loss_strand_feature = loss_strand_feature * self.cfg.loss_weights.strand_feature
                loss_orient = loss_orient * self.cfg.loss_weights.orient
                loss = ( 
                        # loss_rgb_hr +
                        # loss_rgb_lr +
                        # loss_ssim +
                        # loss_vgg +
                        # loss_segment +
                        # loss_transform_reg +
                        # loss_dir +
                        # loss_mesh_dist +
                        # loss_knn_feature +
                        # loss_strand_feature +
                        loss_orient +
                        loss_optical_flow + 
                        loss_optical_flow_hair_reg +
                        loss_optical_flow_head_reg 
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
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
                    'loss_optical_flow_hair_reg': loss_optical_flow_hair_reg,
                    'loss_optical_flow_head_reg': loss_optical_flow_head_reg,
                    'loss_optical_flow' : loss_optical_flow,
                    'pre_shift_head' : pre_shift_head,
                    'pre_shift_hair' : pre_shift_hair,
                    'epoch' : i,
                    # 'iter' : idx + epoch * len(self.dataloader)
                    'iter' : iteration
                }
                with torch.no_grad():
                    self.recorder.log(log)
            
            pre_shift_head += optical_flow_head.detach()
            pre_shift_hair += optical_flow_hair.detach()
            pre_shift_head = pre_shift_head.detach()
            pre_shift_hair = pre_shift_hair.detach()
            pre_shift_hair.requires_grad = False
            pre_shift_head.requires_grad = False

            del data
            del optical_flow_hair
            del optical_flow_head
            del optimizer
            



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
