#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, log, pi

# copy from MiDaS,  of size C x W x H -> C
# def compute_scale_and_shift(prediction, target, mask):
#     # system matrix: A = [[a_00, a_01], [a_10, a_11]]
#     a_00 = torch.sum(mask * prediction * prediction, (1, 2))
#     a_01 = torch.sum(mask * prediction, (1, 2))
#     a_11 = torch.sum(mask, (1, 2))

#     # right hand side: b = [b_0, b_1]
#     b_0 = torch.sum(mask * prediction * target, (1, 2))
#     b_1 = torch.sum(mask * target, (1, 2))

#     # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
#     x_0 = torch.zeros_like(b_0)
#     x_1 = torch.zeros_like(b_1)

#     det = a_00 * a_11 - a_01 * a_01
#     valid = det.nonzero()

#     x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
#     x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

#     return x_0, x_1
# of size C * path_num_w * patch_num_h * patch_size * patch_size -> C * patch_num_w * patch_num_h
def compute_scale_and_shift(prediction, target, mask, isPatch = False):
    
    if isPatch:
        sum_dim = (3, 4)
    else:
        sum_dim = (1, 2)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, sum_dim)
    a_01 = torch.sum(mask * prediction, sum_dim)
    a_11 = torch.sum(mask, sum_dim)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, sum_dim)
    b_1 = torch.sum(mask * target, sum_dim)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # valid = det.nonzero()
    valid = det.abs() > 1e-7
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def l1_loss(network_output, gt, weight = None, mask = None):
    loss = torch.abs(network_output - gt)
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss.mean()

def cluster_loss(xyz, seg, num_clusters = 10, num_iterations = 20):
    # Initialize centroids, by randomly selecting 100 data points
    centroids = xyz[torch.randperm(xyz.shape[0])[:num_clusters]].clone()

    for _ in range(num_iterations):
        # Calculate distances from data points to centroids
        distances = torch.cdist(xyz, centroids)

        # Assign each data point to the closest centroid
        _, labels = torch.min(distances, dim=1)

        # Update centroids by taking the mean of data points assigned to each centroid
        for i in range(num_clusters):
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(xyz[labels == i], dim=0)
    
    distances = torch.cdist(xyz, centroids)
    _, labels = torch.min(distances, dim=1)
    # Compute loss, segment label in same cluster should be similar
    loss = 0
    for i in range(num_clusters):
        if torch.sum(labels == i) > 0:
            mean_seg = seg[labels == i].mean()
            loss += l1_loss(seg[labels == i], mean_seg)
    return loss


# use the soft constraint the distance of means to the centre
def dist_loss(xyz, dist_threshold = 5):
    # xyz: num_points x 3
    square_dist = torch.sum(xyz ** 2, dim = 1)
    loss = torch.relu(square_dist - dist_threshold ** 2)
    return loss.mean()

# compute loss without patch
# Ldepth = l2_depth_loss(depth, gt_depth, mask=depth_mask) if opt.train_depth else torch.zeros_like(Ll1)
def l2_depth_loss(depth, gt_depth, weight = None, mask = None):
    with torch.no_grad():
        depth_scale, depth_shift = compute_scale_and_shift(depth, gt_depth, mask, isPatch = False)
    depth = depth_scale * depth + depth_shift
    return l2_loss(depth, gt_depth, mask=mask, weight=weight)    

    
def l2_depth_patch_loss(depth, gt_depth, weight = None, mask = None):
    kernel_size = 32
    stride = 32
    # C W H -> [C, patch_num_w , patch_num_h, patch_size, patch_size]
    gt_patches = gt_depth.unfold(dimension = 2, size = kernel_size, step = stride).unfold(dimension = 1, size = kernel_size, step = stride)
    output_patches = depth.unfold(dimension = 2, size = kernel_size, step = stride).unfold(dimension = 1, size = kernel_size, step = stride) 
    if mask is not None:
        mask_patches = mask.unfold(dimension = 2, size = kernel_size, step = stride).unfold(dimension = 1, size = kernel_size, step = stride) 
    else:
        mask_patches = torch.ones_like(gt_patches)
    
    if weight is not None:
        weight_patches = weight.unfold(dimension = 2, size = kernel_size, step = stride).unfold(dimension = 1, size = kernel_size, step = stride)
    else:
        weight_patches = torch.ones_like(gt_patches)

    # for (i, j) in [(i, j) for i in range(patch_num_w) for j in range(patch_num_h)]: 
    #     with torch.no_grad():
    #         depth_scale, depth_shift = compute_scale_and_shift(output_patches[:, i, j], gt_patches[:, i, j], mask_patches[:, i, j])
    #     loss += l2_loss(output_patches[:, i, j] * depth_scale + depth_shift, gt_patches[:, i, j], weight_patches, mask_patches[:, i, j]) 
    # return loss / (patch_num_w * patch_num_h)
    with torch.no_grad():
        depth_scale, depth_shift = compute_scale_and_shift(output_patches, gt_patches, mask_patches, isPatch = True)
    loss = l2_loss(output_patches * depth_scale[..., None, None] + depth_shift[...,None, None], gt_patches, weight_patches, mask_patches)
    return loss

def l2_loss(network_output, gt, weight = None, mask = None):
    loss = (network_output - gt).pow(2)
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss.mean()

def ce_loss(network_output, gt):
    return F.binary_cross_entropy(network_output.clamp(1e-3, 1.0 - 1e-3), gt)

def or_loss(network_output, gt, confs = None, weight = None, mask = None):
    weight = torch.ones_like(gt[:1]) if weight is None else weight
    loss = torch.minimum(
        (network_output - gt).abs(),
        torch.minimum(
            (network_output - gt - 1).abs(), 
            (network_output - gt + 1).abs()
        ))
    loss = loss * pi
    if confs is not None:
        loss = loss * confs - (confs + 1e-7).log()    
    if mask is not None:
        loss = loss * mask
    if weight is not None:
        return (loss * weight).sum() / weight.sum()
    else:
        return loss * weight

def dp_loss(pred, gt, pred_mask, gt_mask, eps=0.1):
    filter_fg = torch.logical_and(gt_mask >= 1 - eps, pred_mask >= 1 - eps).detach()
    
    if (filter_fg.sum() == 0).all():
        return None, pred, gt
    
    pred_fg = pred[filter_fg]
    gt_fg = gt[filter_fg]

    with torch.no_grad():    
        # # Subsample points
        # idx_1 = torch.argsort(gt_fg).detach()
        # idx_2 = torch.randperm(gt_fg.shape[0], device='cuda').detach()
        # to_penalize = torch.logical_or(
        #     torch.logical_and(idx_1 < idx_2, pred_fg[idx_1] > pred_fg[idx_2]),
        #     torch.logical_and(idx_1 > idx_2, pred_fg[idx_1] < pred_fg[idx_2])
        # ).detach()

        pred_q2, pred_q98 = torch.quantile(pred_fg, torch.tensor([0.02, 0.98]).cuda())
        gt_q2, gt_q98 = torch.quantile(gt_fg, torch.tensor([0.02, 0.98]).cuda())

    pred_aligned = ((pred - pred_q2.detach()) / (pred_q98.detach() - gt_q2.detach())).clamp(0, 1)
    gt_aligned = ((gt - gt_q2) / (gt_q98 - gt_q2)).clamp(0, 1)

    mask = gt_mask * pred_mask.detach()
    pred_masked = pred_aligned * mask + (1 - mask)
    gt_masked = gt_aligned * mask + (1 - mask)

    loss = (pred_masked - gt_masked).abs().mean()

    return loss, pred_masked, gt_masked

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
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
        return ssim_map


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
