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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return face_normals

def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale

def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals


def hair_strand_rendering(data, gaussianhead, gaussianhair, camera, iteration = 1e6, dynamic_strands=True):

    device = data['images'].device

    # data['poses_history'] = [None]
    data['bg_rgb_color'] = torch.as_tensor([1.0, 1.0, 1.0]).cuda()
    # TODO: select a few strands, color and enlarge them. Then render them
    with torch.no_grad():
        head_data = gaussianhead.generate(data)
        if gaussianhair is not None:
            backprop = iteration < 8000
            # backprop = True 
            gaussianhair.generate_hair_gaussians(poses_history = data['poses_history'][0]  if dynamic_strands else None,
                                                    # global_pose = init_flame_pose[0],
                                                    backprop_into_prior = backprop,
                                                    global_pose = data['flame_pose'][0], 
                                                    global_scale = data['flame_scale'][0])
            hair_data = gaussianhair.generate(data)
                    
            color = hair_data['color'][..., :3].view(1, -1, gaussianhair.strand_length - 1, 3)
            # highlight_strands_idx = torch.arange(0, gaussianhair.num_strands, 300, device= device)
            highlight_strands_idx = torch.arange(0, color.shape[1], 1, device= device)
            gen = torch.Generator(device=device)
            gen.manual_seed(77)
            highlight_color = torch.rand(highlight_strands_idx.shape[0], 3, generator=gen, device=device).unsqueeze(1).repeat(1, gaussianhair.strand_length - 1, 1).unsqueeze(0)
            
            new_color = torch.tensor([1.0, 0.0, 0.0], device=color.device).view(1, 1, 1, 3)

            color[:, highlight_strands_idx, :, :] = highlight_color
            hair_data['color'][..., :3] = color.view(1, -1, 3)
            # Set every 100th strand to new_color
            # color[:, ::100, :, :] = torch.rand(color[:, ::100, :, :].shape[1], 3).unsqueeze(1).repeat(1, 100, 1).unsqueeze(0).to(color.device)

            scales = hair_data['scales'].view(1, -1, gaussianhair.strand_length - 1, 3)
            scales[:, highlight_strands_idx, :, 1: ] = 10 * scales[:, highlight_strands_idx, :, 1: ]
            hair_data['scales'] = scales.view(1, -1, 3)


            hair_data['opacity'][...] = 0.0
            opacity = hair_data['opacity'].view(1, -1, gaussianhair.strand_length - 1, 1)
            opacity[:, highlight_strands_idx, :, :] = 1.0
            hair_data['opacity'] = opacity.view(1, -1, 1)

            # combine head and hair data
            for key in ['xyz', 'color', 'scales', 'rotation', 'opacity']:
                # first dimension is batch size, concat along the second dimension
                # data[key] = hair_data[key]
                data[key] = torch.cat([head_data[key], hair_data[key]], dim=1)

        data = camera.render_gaussian(data, 512)
        render_images = data['render_images'][: ,:3, ...]
        return render_images[0].permute(1,2,0).clamp(0,1).cpu().numpy()


def hair_strand_coloring(data, gaussianhair):
    device = data['images'].device
    color = data['color'][..., :3].view(1, -1, gaussianhair.strand_length - 1, 3)
    new_color = torch.tensor([1.0, 0.0, 0.0], device=color.device).view(1, 1, 1, 3)
    
    highlight_strands_idx = torch.arange(0, color.shape[1], 1, device= device)
    gen = torch.Generator(device=device)
    gen.manual_seed(77)
    highlight_color = torch.rand(highlight_strands_idx.shape[0], 3, generator=gen, device=device).unsqueeze(1).repeat(1, 99, 1).unsqueeze(0)
         

    color[:, highlight_strands_idx, :, :] = highlight_color
    data['color'][..., :3] = color.view(1, -1, 3)
    # Set every 100th strand to new_color
    # color[:, ::100, :, :] = torch.rand(color[:, ::100, :, :].shape[1], 3).unsqueeze(1).repeat(1, 100, 1).unsqueeze(0).to(color.device)

    scales = data['scales'].view(1, -1, gaussianhair.strand_length - 1, 3)
    scales[:, highlight_strands_idx, :, 1: ] = 10 * scales[:, highlight_strands_idx, :, 1: ]
    data['scales'] = scales.view(1, -1, 3)


    data['opacity'][...] = 0.0
    opacity = data['opacity'].view(1, -1, gaussianhair.strand_length - 1, 1)
    opacity[:, highlight_strands_idx, :, :] = 1.0
    data['opacity'] = opacity.view(1, -1, 1)

    return data
