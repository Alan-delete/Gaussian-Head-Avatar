
import os
import sys
import numpy as np
import torch
from torch import nn
import tqdm
from plyfile import PlyData, PlyElement
from einops import rearrange
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops.knn import knn_gather, knn_points

from lib.utils.general_utils import inverse_sigmoid
from lib.module.GaussianBaseModule import GaussianBaseModule

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/../../ext/perm/src')
from hair.hair_models import Perm



class GaussianHairModule(GaussianBaseModule):
    def __init__(self, strands_config, optimizer=None ):
        super().__init__(optimizer)

        # Hair strands
        self.num_strands = strands_config['extra_args']['num_strands']
        self.strand_length = strands_config['extra_args']['strand_length']
        self.simplify_strands = strands_config['extra_args']['simplify_strands']
        self.aspect_ratio = strands_config['extra_args']['aspect_ratio']
        self.quantile = strands_config['extra_args']['quantile']
        self.train_features_rest = strands_config['extra_args']['train_features_rest']
        self.train_width = strands_config['extra_args']['train_width']
        self.train_opacity = strands_config['extra_args']['train_opacity']
        self.train_directions = strands_config['extra_args']['train_directions']
        # TODO: change the path to the format like {dir_perm}/checkpoints
        self.strands_generator = Perm(
            model_path=f'{dir_path}/../../ext/perm/checkpoints', 
            head_mesh=f'{dir_path}/../../ext/perm/data/head.obj',
            scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047]).eval().cuda().requires_grad_(True)
        if self.num_strands == 10_140:
            hair_roots_path = f'{dir_path}/../../ext/perm/data/roots/rootPositions_10k.txt'
        elif self.num_strands == 21_057:
            hair_roots_path = f'{dir_path}/../../ext/perm/data/roots/rootPositions_20k.txt'
        elif self.num_strands == 30_818:
            hair_roots_path = f'{dir_path}/../../ext/perm/data/roots/rootPositions_30k.txt'
        roots, _ = self.strands_generator.hair_roots.load_txt(hair_roots_path)
        self.roots = roots.cuda().unsqueeze(0)
        with torch.no_grad():
            init_theta = self.strands_generator.G_raw.mapping(torch.zeros(1, self.strands_generator.G_raw.z_dim).cuda())
        self.theta = nn.Parameter(init_theta.requires_grad_().cuda())
        self.beta = nn.Parameter(torch.zeros(1, self.strands_generator.G_res.num_ws, self.strands_generator.G_res.w_dim).requires_grad_().cuda())
        if self.train_directions:
            self._dir_struct_raw = torch.empty(0)
        else:
            self._points_struct_raw = torch.empty(0)
        self._features_dc_struct_raw = torch.empty(0)
        self._features_rest_struct_raw = torch.empty(0)
        self._width_struct_raw = torch.empty(0)
        self._opacity_struct_raw = torch.empty(0)

        self.transform = torch.eye(4).cuda()

    def get_direction_2d(self, viewpoint_camera):
        mean = self.get_xyz

        height = int(viewpoint_camera.image_height)
        width = int(viewpoint_camera.image_width)
    
        tan_fovx = torch.tan(viewpoint_camera.FoVx * 0.5)
        tan_fovy = torch.tan(viewpoint_camera.FoVy * 0.5)

        focal_y = height / (2.0 * tan_fovy)
        focal_x = width / (2.0 * tan_fovx)

        viewmatrix = viewpoint_camera.world_view_transform

        t = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        txtz = tx / tz
        tytz = ty / tz
        
        tx = torch.clamp(txtz, min=-limx, max=limx) * tz
        ty = torch.clamp(tytz, min=-limy, max=limy) * tz

        zeros = torch.zeros_like(tz)

        J = torch.stack(
            [
                torch.stack([focal_x / tz,        zeros, -(focal_x * tx) / (tz * tz)], dim=-1), # 1st column
                torch.stack([       zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1), # 2nd column
                torch.stack([       zeros,        zeros,                       zeros], dim=-1)  # 3rd column
            ],
            dim=-1 # stack columns into rows
        )

        W = viewmatrix[None, :3, :3]

        T = W @ J

        # Get structured Gaussian directions
        self._dir =  self._dir_struct

        #dir3D = F.normalize(self._dir, dim=-1)
        dir2D = (self._dir[:, None, :] @ T)[:, 0]
        dir2D = torch.nn.functional.normalize(dir2D, dim=-1)
        # make only strctured gassiuan has direction
        # dir2D[:self._dir_unstruct.shape[0]] = torch.zeros_like(dir2D[:self._dir_unstruct.shape[0]])
        return dir2D

    def sample_strands_from_prior(self, num_strands = -1):
        # Subsample Perm strands if needed
        roots = self.roots
        if num_strands < self.num_strands and num_strands != -1:
            strands_idx = torch.randperm(self.num_strands)[:num_strands]
            roots = roots[strands_idx]
        else:
            strands_idx = None

        out = self.strands_generator(roots, self.theta, self.beta)
        pts_perm = out['strands'][0].position

        # Map strands into the scene coordinates
        pts = (torch.cat([pts_perm, torch.ones_like(pts_perm[..., :1])], dim=-1) @ self.transform)[..., :3]
        dir = pts[:, 1:] - pts[:, :-1]

        return pts[:, 1:], dir.view(-1, 3), pts[:, :1], strands_idx
    

    # set gaussian representation from hair strands
    def generate_hair_gaussians(self, num_strands = -1, skip_color = False, skip_smpl = False, backprop_into_prior = False):
        # only optimize prior
        if backprop_into_prior:
            self._points_struct, self._dir_struct, self._origins_struct, strands_idx = self.sample_strands_from_prior(num_strands)
            self._points_origins_struct = torch.cat([self._origins_struct, self._points_struct], dim=1)
            if num_strands == -1:
                num_strands = self.num_strands
        # directly optimize (structured) hair strand points
        else:
            if num_strands < self.num_strands and num_strands != -1:
                strands_idx = torch.randperm(self.num_strands)[:num_strands]
                self._origins_struct = self._origins_struct_raw[strands_idx]
                if self.train_directions:
                    self._dir_struct = self._dir_struct_raw.view(self.num_strands, self.strand_length - 1, 3)[strands_idx].view(-1, 3)
                    self._points_origins_struct = torch.cumsum(torch.cat([
                        self._origins_struct, 
                        self._dir_struct.view(num_strands, self.strand_length -1, 3)
                    ], dim=1), dim=1)
                else:
                    self._points_struct = self._points_struct_raw[strands_idx]
                    self._points_origins_struct = torch.cat([self._origins_struct, self._points_struct], dim=1)
                    self._dir_struct = (self._points_origins_struct[:, 1:] - self._points_origins_struct[:, :-1]).view(-1, 3)
            else:
                num_strands = self.num_strands
                self._origins_struct = self._origins_struct_raw
                if self.train_directions:
                    self._dir_struct = self._dir_struct_raw
                    self._points_origins_struct = torch.cumsum(torch.cat([
                        self._origins_struct, 
                        self._dir_struct.view(num_strands, self.strand_length -1, 3)
                    ], dim=1), dim=1)
                else:
                    self._points_struct = self._points_struct_raw
                    self._points_origins_struct = torch.cat([self._origins_struct, self._points_struct], dim=1)
                    self._dir_struct = (self._points_origins_struct[:, 1:] - self._points_origins_struct[:, :-1]).view(-1, 3)

        if num_strands < self.num_strands and num_strands != -1:
            self._width_struct = self._width_struct_raw[strands_idx]
            self._opacity_struct = self._opacity_struct_raw.view(self.num_strands, self.strand_length - 1, 1)[strands_idx].view(-1, 1)
            if not skip_color:
                self._features_dc_struct = self._features_dc_struct_raw.view(self.num_strands, self.strand_length - 1, 1, 3)[strands_idx].view(-1, 1, 3)
                self._features_rest_struct = self._features_rest_struct_raw.view(self.num_strands, self.strand_length - 1, (self.max_sh_degree + 1) ** 2 - 1, 3).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
        else:
            self._width_struct = self._width_struct_raw
            self._opacity_struct = self._opacity_struct_raw
            if not skip_color:
                self._features_dc_struct = self._features_dc_struct_raw
                self._features_rest_struct = self._features_rest_struct_raw

        self._xyz_struct = (self._points_origins_struct[:, 1:] + self._points_origins_struct[:, :-1]).view(-1, 3) * 0.5

        self.scaling_struct = torch.ones_like(self._xyz_struct)
        self.scaling_struct[:, 0] = self._dir_struct.norm(dim=-1) * 0.66
        self.scaling_struct[:, 1:] = self._width_struct.repeat(1, self.strand_length - 1).view(-1, 1)

        self._seg_label_struct = torch.zeros_like(self._xyz_struct)

        if not skip_smpl and self.simplify_strands:
            # Run line simplification
            MAX_ITERATIONS = 4
            xyz = self._xyz_struct.view(num_strands, self.strand_length - 1, 3)
            dir = self._dir_struct.view(num_strands, self.strand_length - 1, 3)
            num_gaussians = xyz.shape[1]
            len = (dir**2).sum(-1)
            if not skip_color:
                features_dc = self._features_dc_struct.view(num_strands, self.strand_length - 1, 3)
                features_rest = self._features_rest_struct.view(num_strands, self.strand_length - 1, -1)
            scaling = self.scaling_struct.view(num_strands, self.strand_length - 1, 3)
            opacity = self._opacity_struct.view(num_strands, self.strand_length - 1, 1)
            for _ in range(MAX_ITERATIONS):
                new_num_gaussians = num_gaussians // 2
                dir_new = (dir[:, :new_num_gaussians*2:2, :] + dir[:, 1::2, :])
                len_new = (dir_new**2).sum(-1)
                err = ( len[:, :new_num_gaussians*2:2] - (dir[:, :new_num_gaussians*2:2, :] * dir_new).sum(-1)**2 / (len_new + 1e-7) )**0.5
                xyz_new = (xyz[:, :new_num_gaussians*2:2, :] + xyz[:, 1::2, :]) * 0.5
                if not skip_color:
                    features_dc_new = (features_dc[:, :new_num_gaussians*2:2, :] + features_dc[:, 1::2, :]) * 0.5
                    features_rest_new = (features_rest[:, :new_num_gaussians*2:2, :] + features_rest[:, 1::2, :]) * 0.5
                scaling_new = (scaling[:, :new_num_gaussians*2:2, :] + scaling[:, 1::2, :]) * 0.5
                opacity_new = (opacity[:, :new_num_gaussians*2:2, :] + opacity[:, 1::2, :]) * 0.5
                if (torch.quantile(err, self.quantile, dim=1) < self._width_struct * self.aspect_ratio).float().mean() > 0.5:
                    if num_gaussians % 2:
                        xyz = torch.cat([xyz_new, xyz[:, -1:]], dim=1)
                        dir = torch.cat([dir_new, dir[:, -1:]], dim=1)
                        len = torch.cat([len_new, len[:, -1:]], dim=1)
                        if not skip_color:
                            features_dc = torch.cat([features_dc_new, features_dc[:, -1:]], dim=1)
                            features_rest = torch.cat([features_rest_new, features_rest[:, -1:]], dim=1)
                        scaling = torch.cat([scaling_new, scaling[:, -1:]], dim=1)
                        opacity = torch.cat([opacity_new, opacity[:, -1:]], dim=1)
                    else:
                        xyz = xyz_new
                        dir = dir_new
                        len = len_new
                        if not skip_color:
                            features_dc = features_dc_new
                            features_rest = features_rest_new
                        scaling = scaling_new
                        opacity = opacity_new
                    num_gaussians = xyz.shape[1]
                else:
                    break
            self._xyz_struct = xyz.view(-1, 3)
            self._dir_struct = dir.view(-1, 3)
            if not skip_color:
                self._features_dc_struct = features_dc.view(-1, 1, 3)
                self._features_rest_struct = features_rest.view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
            self.scaling_struct = scaling.view(-1, 3)
            self._opacity_struct= opacity.view(-1, 1)
        
            if num_gaussians + 1 != self.prev_strand_length:
                print(f'Simplified strands from {self.prev_strand_length} to {num_gaussians + 1} points')
                self.prev_strand_length = num_gaussians + 1

        self._scaling_struct = self.scaling_inverse_activation(self.scaling_struct)


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
        # Assign geometric features        
        self._rotation_struct = parallel_transport(
            a=torch.cat(
                [
                    torch.ones_like(self._xyz_struct[:, :1]),
                    torch.zeros_like(self._xyz_struct[:, :2])
                ],
                dim=-1
            ),
            b=self._dir_struct
        ).view(-1, 4) # rotation parameters that align x-axis with the segment direction


    # initialize the hair gaussian representation
    def create_hair_gaussians(self, scaling_factor = 1e-3):
        self.prev_strand_length = self.strand_length
        self.prev_mesh_width = 1.0
        with torch.no_grad():
            points_struct_raw, dir_struct_raw, self._origins_struct_raw, _ = self.sample_strands_from_prior(self.num_strands)
        if self.train_directions:
            self._dir_struct_raw = nn.Parameter(dir_struct_raw.contiguous().requires_grad_(True))
        else:
            self._points_struct_raw = nn.Parameter(points_struct_raw.contiguous().requires_grad_(True))
        self._width_struct_raw = nn.Parameter(torch.ones(self.num_strands, 1).cuda().contiguous().requires_grad_(True))
        self._width_struct_raw.data *= scaling_factor
        self._opacity_struct_raw = nn.Parameter(inverse_sigmoid(1.0 * torch.ones(self.num_strands * (self.strand_length - 1), 1, dtype=torch.float, device="cuda")))
        self.generate_hair_gaussians(skip_color=True, skip_smpl=True)
        closest_idx = []
        with torch.no_grad():
            print('Initializing hair Gaussians color using closest COLMAP points')
            for i in tqdm(range(self._xyz_struct.shape[0] // 1_000 + (self._xyz_struct.shape[0] % 1000 > 0))):
                closest_idx.append(((self._xyz_struct[i*1_000 : (i+1)*1_000, None, :] - self.init_xyz[None, :, :])**2).sum(-1).amin(-1))
        closest_idx = torch.cat(closest_idx).long()
        features_dc = self.init_features[closest_idx].view(self.num_strands, self.strand_length - 1, 3).mean(1, keepdim=True)
        features_dc = features_dc.repeat(1, self.strand_length - 1, 1).view(self.num_strands * (self.strand_length - 1), 3)
        self._features_dc_struct_raw = nn.Parameter(features_dc[:, None, :].contiguous().cuda().requires_grad_(True))
        assert self._features_dc_struct_raw.shape[0] == self.num_strands * (self.strand_length - 1)
        self._features_rest_struct_raw = nn.Parameter(torch.zeros(self.num_strands * (self.strand_length - 1), (self.max_sh_degree + 1) ** 2 - 1, 3).cuda().requires_grad_(True))

    def generate(self,data):
        B = data['exp_coeff'].shape[0]

        xyz = self.xyz.unsqueeze(0).repeat(B, 1, 1)
        feature = torch.tanh(self.feature).unsqueeze(0).repeat(B, 1, 1)

        dists, _, _ = knn_points(xyz, self.landmarks_3d_neutral.unsqueeze(0).repeat(B, 1, 1))
        exp_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
        pose_weights = 1 - exp_weights
        exp_controlled = (dists < self.dist_threshold_far).squeeze(-1)
        pose_controlled = (dists > self.dist_threshold_near).squeeze(-1)

        color = torch.zeros([B, xyz.shape[1], self.exp_color_mlp.dims[-1]], device=xyz.device)
        # dir2d + depth + seg = 1 + 1 + 3
        extra_feature = torch.zeros([B, xyz.shape[1], 5], device=xyz.device)
        delta_xyz = torch.zeros_like(xyz, device=xyz.device)
        delta_attributes = torch.zeros([B, xyz.shape[1], self.scales.shape[1] + self.rotation.shape[1] + self.opacity.shape[1]], device=xyz.device)
        for b in range(B):
            feature_exp_controlled = feature[b, exp_controlled[b], :]
            exp_color_input = torch.cat([feature_exp_controlled.t(), 
                                         data['exp_coeff'][b].unsqueeze(-1).repeat(1, feature_exp_controlled.shape[0])], 0)[None]
            exp_color = self.exp_color_mlp(exp_color_input)[0].t()
            color[b, exp_controlled[b], :] += exp_color * exp_weights[b, exp_controlled[b], :]

            feature_pose_controlled = feature[b, pose_controlled[b], :]
            pose_color_input = torch.cat([feature_pose_controlled.t(), 
                                               self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, feature_pose_controlled.shape[0])], 0)[None]
            pose_color = self.pose_color_mlp(pose_color_input)[0].t()
            color[b, pose_controlled[b], :] += pose_color * pose_weights[b, pose_controlled[b], :]

            exp_attributes_input = exp_color_input
            exp_delta_attributes = self.exp_attributes_mlp(exp_attributes_input)[0].t()
            delta_attributes[b, exp_controlled[b], :] += exp_delta_attributes * exp_weights[b, exp_controlled[b], :]

            pose_attributes_input = pose_color_input
            pose_attributes = self.pose_attributes_mlp(pose_attributes_input)[0].t()
            delta_attributes[b, pose_controlled[b], :] += pose_attributes * pose_weights[b, pose_controlled[b], :]


            xyz_exp_controlled = xyz[b, exp_controlled[b], :]
            exp_deform_input = torch.cat([self.pos_embedding(xyz_exp_controlled).t(), 
                                          data['exp_coeff'][b].unsqueeze(-1).repeat(1, xyz_exp_controlled.shape[0])], 0)[None]
            exp_deform = self.exp_deform_mlp(exp_deform_input)[0].t()
            delta_xyz[b, exp_controlled[b], :] += exp_deform * exp_weights[b, exp_controlled[b], :]

            xyz_pose_controlled = xyz[b, pose_controlled[b], :]
            pose_deform_input = torch.cat([self.pos_embedding(xyz_pose_controlled).t(), 
                                           self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, xyz_pose_controlled.shape[0])], 0)[None]
            pose_deform = self.pose_deform_mlp(pose_deform_input)[0].t()
            delta_xyz[b, pose_controlled[b], :] += pose_deform * pose_weights[b, pose_controlled[b], :]

        xyz = xyz + delta_xyz * self.deform_scale

        delta_scales = delta_attributes[:, :, 0:3]
        scales = self.scales.unsqueeze(0).repeat(B, 1, 1) + delta_scales * self.attributes_scale
        scales = torch.exp(scales)

        delta_rotation = delta_attributes[:, :, 3:7]
        rotation = self.rotation.unsqueeze(0).repeat(B, 1, 1) + delta_rotation * self.attributes_scale
        rotation = torch.nn.functional.normalize(rotation, dim=2)

        delta_opacity = delta_attributes[:, :, 7:8]
        opacity = self.opacity.unsqueeze(0).repeat(B, 1, 1) + delta_opacity * self.attributes_scale
        opacity = torch.sigmoid(opacity)

        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, None, 3:]
            S = data['scale'][:, :, None]
            xyz = torch.bmm(xyz * S, R.permute(0, 2, 1)) + T

            rotation_matrix = quaternion_to_matrix(rotation)
            rotation_matrix = rearrange(rotation_matrix, 'b n x y -> (b n) x y')
            R = rearrange(R.unsqueeze(1).repeat(1, rotation.shape[1], 1, 1), 'b n x y -> (b n) x y')
            rotation_matrix = rearrange(torch.bmm(R, rotation_matrix), '(b n) x y -> b n x y', b=B)
            rotation = matrix_to_quaternion(rotation_matrix)

            scales = scales * S
        color = torch.cat([color, extra_feature], dim=-1)
        data['exp_deform'] = exp_deform
        data['xyz'] = xyz
        data['color'] = color
        data['scales'] = scales
        data['rotation'] = rotation
        data['opacity'] = opacity
        return data