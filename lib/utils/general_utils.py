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
import sys
from datetime import datetime
import numpy as np
import random
import math
import torch.nn.functional as F
import os
import shutil
import glob

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def eval_sh_bases(deg, dirs):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param deg: int SH max degree. Currently, 0-4 supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., (deg+1) ** 2)
    """
    assert deg <= 4 and deg >= 0
    result = torch.empty((*dirs.shape[:-1], (deg + 1) ** 2), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = C0
    if deg > 0:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -C1 * y;
        result[..., 2] = C1 * z;
        result[..., 3] = -C1 * x;
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = C2[0] * xy;
            result[..., 5] = C2[1] * yz;
            result[..., 6] = C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = C2[3] * xz;
            result[..., 8] = C2[4] * (xx - yy);

            if deg > 2:
                result[..., 9] = C3[0] * y * (3 * xx - yy);
                result[..., 10] = C3[1] * xy * z;
                result[..., 11] = C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = C3[5] * z * (xx - yy);
                result[..., 15] = C3[6] * x * (xx - 3 * yy);

                if deg > 3:
                    result[..., 16] = C4[0] * xy * (xx - yy);
                    result[..., 17] = C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=r.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def Rotate_y_180(X, pos='right'):
    R = torch.eye(3).to(X.device)
    R[0,0] = -1.0
    R[2,2] = -1.0
    if pos == 'right':
        X = torch.matmul(X, R)
    else:
        X = torch.matmul(R, X)
    return X

def Rotate_z_180(X, pos='right'):
    R = torch.eye(3).to(X.device)
    R[0,0] = -1.0
    R[1,1] = -1.0
    if pos == 'right':
        X = torch.matmul(X, R)
    else:
        X = torch.matmul(R, X)
    return X



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


            # flame_pose = torch.from_numpy(flame_param['pose'][0]).float()
            # flame_scale = torch.from_numpy(flame_param['scale']).float().view(-1)
            # # breakpoint()
            # # exp_coeff = torch.from_numpy(flame_param['exp_coeff'][0]).float()
            # # id_coeff = torch.from_numpy(flame_param['id_coeff'][0]).float()
def GHA2FLAME(pose, scale, exp_coeff, id_coeff, exp_dims=50, id_dims=100):
    """
    Convert Gaussian Head A parameters to FLAME parameters.
    :param pose: (B, 6) tensor
    :param scale: (B, 3) tensor
    :param exp_coeff: (B, E) tensor
    :param id_coeff: (B, I) tensor
    :return: dict with FLAME parameters
    """
            # expression_params = exp_coeff[:, : exp_dims]
            # jaw_rotation = exp_coeff[:, exp_dims: exp_dims + 3]
            # neck_pose = torch.zeros(exp_coeff.shape[0], 3, dtype=torch.float32)  # neck pose is not used in GHA
            # eye_pose = exp_coeff[:, exp_dims + 3: exp_dims + 9]

            # pose_params = torch.cat([self.global_rotation, jaw_rotation], 1)
            # shape_params = self.id_coeff.repeat(self.batch_size, 1)
    expression_params = exp_coeff[:, :exp_dims]
    jaw_rotation = exp_coeff[:, exp_dims: exp_dims + 3]
    neck_pose = torch.zeros(exp_coeff.shape[0], 3, dtype=torch.float32)  # neck pose is not used in GHA
    eye_pose = exp_coeff[:, exp_dims + 3: exp_dims + 9]
    pose_params = torch.cat([pose[:, :3], jaw_rotation], 1)
    shape_params = id_coeff.repeat(pose.shape[0], 1)

    return 


def to_rgb_pca(tex):
    """ Convert texture to RGB using PCA. """
    C, H, W = tex.shape
    def normalize_per_channel(arr):
        arr = arr.astype(np.float32)
        arr_ = arr.reshape(C, -1)
        mins = arr_.min(axis=1, keepdims=True)
        maxs = arr_.max(axis=1, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-8)
        arr_norm = ((arr_ - mins) / denom).reshape(C, H, W)
        return arr_norm
    chw = normalize_per_channel(tex)
    X = chw.reshape(C, -1).T  # [H*W, C]
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:3]  # [3, C]
    proj = Xc @ comps.T  # [H*W, 3]
    # Normalize each channel to [0,1]
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    return proj.reshape(H, W, 3)


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




# used to convert the strcuture input_folder/images/frame_id/image_camera_id.jpg
# to the structure input_folder/images/camera_id/image_frame_id.jpg
# e.g. input_folder/images/00001/image_222200049.jpg
# to input_folder/images/222200049/00001.jpg
def convert_file_structure(img_dir, output_dir):
    frames = sorted(os.listdir(img_dir))
    frames = [f for f in frames if os.path.isdir(os.path.join(img_dir, f))]
    for frame in frames:
        frame_path = os.path.join(img_dir, frame)
        images = glob.glob(os.path.join(frame_path, 'image_[0-9]*.jpg'))
        for image in images:
            camera_id = image.split('_')[1].split('.')[0]  # Extract camera ID from the image name
            new_image_name = f"image_{frame}.jpg"
            new_camera_dir = os.path.join(output_dir, camera_id)
            os.makedirs(new_camera_dir, exist_ok=True)
            old_image_path = os.path.join(frame_path, image)
            new_image_path = os.path.join(new_camera_dir, new_image_name)
            # copy the image to the new location
            shutil.copy(old_image_path, new_image_path)


def dot(a, b, dim=-1, keepdim=True):
    return (a*b).sum(dim=dim, keepdim=keepdim)
        
def parallel_transport(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
            
    s = 1 + dot(a, b, dim=-1, keepdim=True)
    v = torch.cross(a, b, dim=-1)
    q = torch.cat([s, v], dim=-1)
            
    # q = F.normalize(q, dim=-1)

    return q

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


# from https://gist.github.com/dendenxu/ee5008acb5607195582e7983a384e644#file-moller_trumbore-py-L8

from typing import Tuple

def multi_indexing(index: torch.Tensor, shape: torch.Size, dim=-2):
    shape = list(shape)
    back_pad = len(shape) - index.ndim
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)



def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the index's last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(index, values.shape, dim))


def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1):
        f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3


def ray_stabbing(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, multiplier: int = 1):
    """
    Check whether a bunch of points is inside the mesh defined by verts and faces
    effectively calculating their occupancy values
    Parameters
    ----------
    ray_o : torch.Tensor(float), (n_rays, 3)
    verts : torch.Tensor(float), (n_verts, 3)
    faces : torch.Tensor(long), (n_faces, 3)
    """
    n_rays = pts.shape[0]
    pts = pts[None].expand(multiplier, n_rays, -1)
    pts = pts.reshape(-1, 3)
    ray_d = torch.rand_like(pts)  # (n_rays, 3)
    ray_d = normalize(ray_d)  # (n_rays, 3)
    u, v, t = moller_trumbore(pts, ray_d, multi_gather_tris(verts, faces))  # (n_rays, n_faces, 3)
    inside = ((t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0)).bool()  # (n_rays, n_faces)
    inside = (inside.count_nonzero(dim=-1) % 2).bool()  # if mod 2 is 0, even, outside, inside is odd
    inside = inside.view(multiplier, n_rays, -1)
    inside = inside.sum(dim=0) / multiplier  # any show inside mesh
    return inside

def moller_trumbore(ray_o: torch.Tensor, ray_d: torch.Tensor, tris: torch.Tensor, eps=1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The Moller Trumbore algorithm for fast ray triangle intersection
    Naive batch implementation (m rays and n triangles at the same time)
    O(n_rays * n_faces) memory usage, parallelized execution
    Parameters
    ----------
    ray_o : torch.Tensor, (n_rays, 3)
    ray_d : torch.Tensor, (n_rays, 3)
    tris  : torch.Tensor, (n_faces, 3, 3)
    """
    E1 = tris[:, 1] - tris[:, 0]  # vector of edge 1 on triangle (n_faces, 3)
    E2 = tris[:, 2] - tris[:, 0]  # vector of edge 2 on triangle (n_faces, 3)

    # batch cross product
    N = torch.cross(E1, E2)  # normal to E1 and E2, automatically batched to (n_faces, 3)

    invdet = 1. / -(torch.einsum('md,nd->mn', ray_d, N) + eps)  # inverse determinant (n_faces, 3)

    A0 = ray_o[:, None] - tris[None, :, 0]  # (n_rays, 3) - (n_faces, 3) -> (n_rays, n_faces, 3) automatic broadcast
    DA0 = torch.cross(A0, ray_d[:, None].expand(*A0.shape))  # (n_rays, n_faces, 3) x (n_rays, 3) -> (n_rays, n_faces, 3) no automatic broadcast

    u = torch.einsum('mnd,nd->mn', DA0, E2) * invdet
    v = -torch.einsum('mnd,nd->mn', DA0, E1) * invdet
    t = torch.einsum('mnd,nd->mn', A0, N) * invdet  # t >= 0.0 means this is a ray

    return u, v, t


# winding number way 
def winding_number(pts: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Parallel implementation of the Generalized Winding Number of points on the mesh
    O(n_points * n_faces) memory usage, parallelized execution
    1. Project tris onto the unit sphere around every points
    2. Compute the signed solid angle of the each triangle for each point
    3. Sum the solid angle of each triangle
    Parameters
    ----------
    pts    : torch.Tensor, (n_points, 3)
    verts  : torch.Tensor, (n_verts, 3)
    faces  : torch.Tensor, (n_faces, 3)
    This implementation is also able to take a/multiple batch dimension
    """
    # projection onto unit sphere: verts implementation gives a little bit more performance
    uv = verts[..., None, :, :] - pts[..., :, None, :]  # n_points, n_verts, 3
    uv = uv / uv.norm(dim=-1, keepdim=True)  # n_points, n_verts, 3

    # gather from the computed vertices (will result in a copy for sure)
    expanded_faces = faces[..., None, :, :].expand(*faces.shape[:-2], pts.shape[-2], *faces.shape[-2:])  # n_points, n_faces, 3

    u0 = multi_gather(uv, expanded_faces[..., 0])  # n, f, 3
    u1 = multi_gather(uv, expanded_faces[..., 1])  # n, f, 3
    u2 = multi_gather(uv, expanded_faces[..., 2])  # n, f, 3

    e0 = u1 - u0  # n, f, 3
    e1 = u2 - u1  # n, f, 3
    del u1

    # compute solid angle signs
    sign = (torch.cross(e0, e1) * u2).sum(dim=-1).sign()

    e2 = u0 - u2
    del u0, u2

    l0 = e0.norm(dim=-1)
    del e0

    l1 = e1.norm(dim=-1)
    del e1

    l2 = e2.norm(dim=-1)
    del e2

    # compute edge lengths: pure triangle
    l = torch.stack([l0, l1, l2], dim=-1)  # n_points, n_faces, 3

    # compute spherical edge lengths
    l = 2 * (l/2).arcsin()  # n_points, n_faces, 3

    # compute solid angle: preparing: n_points, n_faces
    s = l.sum(dim=-1) / 2
    s0 = s - l[..., 0]
    s1 = s - l[..., 1]
    s2 = s - l[..., 2]

    # compute solid angle: and generalized winding number: n_points, n_faces
    eps = 1e-10  # NOTE: will cause nan if not bigger than 1e-10
    solid = 4 * (((s/2).tan() * (s0/2).tan() * (s1/2).tan() * (s2/2).tan()).abs() + eps).sqrt().arctan()
    signed_solid = solid * sign  # n_points, n_faces

    winding = signed_solid.sum(dim=-1) / (4 * torch.pi)  # n_points

    return winding


def find_boundary_edges(faces):
    # Create all directed edges
    edges = torch.cat([faces[:, [0, 1]],
                       faces[:, [1, 2]],
                       faces[:, [2, 0]]], dim=0)

    # Sort each edge so direction is ignored
    edges_sorted, _ = edges.sort(dim=1)
    edges_sorted = edges_sorted.cpu().numpy()

    # Count occurrences of each edge
    from collections import defaultdict
    edge_count = defaultdict(int)
    for e in map(tuple, edges_sorted):
        edge_count[e] += 1

    # Edges that appear only once are boundary edges
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    return torch.tensor(boundary_edges, dtype=torch.long)

# get ordered vertices indices
def order_boundary_loop(edges):
    # Build connectivity
    from collections import defaultdict
    conn = defaultdict(list)
    for a, b in edges.tolist():
        conn[a].append(b)
        conn[b].append(a)

    # Start at any vertex
    # loop = [edges[0, 0].item()] 3260 3359
    loop = [3260]
    visited = set(loop)
    while True:
        last = loop[-1]
        next_candidates = [n for n in conn[last] if n not in visited]
        if not next_candidates:
            break
        next_v = next_candidates[0]
        loop.append(next_v)
        visited.add(next_v)
    return loop



if __name__ == "__main__":
    # Example usage
    input_folder = '/local/home/haonchen/Gaussian-Head-Avatar/datasets/NeRSemble/258/sequences/HAIR/images'
    output_folder = '/local/home/haonchen/Gaussian-Head-Avatar/datasets/NeRSemble/258/sequences/HAIR/images_converted'
    convert_file_structure(input_folder, output_folder)
    print(f"Converted file structure from {input_folder} to {output_folder}")