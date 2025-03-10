
import os
import sys
import numpy as np
import torch
from torch import nn
import tqdm
from plyfile import PlyData, PlyElement
from einops import rearrange

from simple_knn._C import distCUDA2
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops.knn import knn_gather, knn_points
import open3d as o3d

from lib.utils.general_utils import inverse_sigmoid
from lib.module.GaussianBaseModule import GaussianBaseModule
from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/../../ext/perm/src')
from hair.hair_models import Perm


# TODO: move to lib/utils/general_utils.py
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


class GaussianHairModule(GaussianBaseModule):
    def __init__(self, cfg, optimizer=None ):
        super().__init__(optimizer)

        self.cfg = cfg
        # Hair strands
        self.num_strands = cfg['num_strands']
        self.strand_length = cfg['strand_length']
        self.simplify_strands = cfg['simplify_strands']
        self.aspect_ratio = cfg['aspect_ratio']
        self.quantile = cfg['quantile']
        self.train_features_rest = cfg['train_features_rest']
        self.train_width = cfg['train_width']
        self.train_opacity = cfg['train_opacity']
        self.train_directions = cfg['train_directions']
        self.max_sh_degree = cfg['sh_degree']
        self.active_sh_degree = self.max_sh_degree if self.train_features_rest else 0
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

        self.origins_raw = torch.empty(0)
        self.points_raw = torch.empty(0)
        self.dir_raw = torch.empty(0)
        self.features_dc_raw = torch.empty(0)
        self.features_rest_raw = torch.empty(0)
        self.opacity_raw = torch.empty(0)
        self.width_raw = torch.empty(0)


        self.pose_color_mlp = MLP(cfg.pose_color_mlp, last_op=None)
        self.pose_attributes_mlp = MLP(cfg.pose_attributes_mlp, last_op=None)
        self.pose_deform_mlp = MLP(cfg.pose_deform_mlp, last_op=nn.Tanh())
        self.pose_point_mlp = MLP(cfg.pose_point_mlp, last_op=None)
        self.pose_prior_mlp = MLP(cfg.pose_prior_mlp, last_op=None)
        # pose [6] -> pose embedding [54]
        self.pos_embedding, _ = get_embedder(cfg.pos_freq)

        self.transform = torch.eye(4).cuda()

        self.create_hair_gaussians(cfg.strand_scale)

        # TODO: By printing the value of Gaussian Hair cut. Need to get this value in this project
        self.cameras_extent = 4.907987451553345
        self.spatial_lr_scale = self.cameras_extent

        # TODO: Add learning rate for each parameter
        l_struct = [
            {'params': [self.features_dc_raw], 'lr': cfg.feature_lr, "name": "f_dc"},
            {'params': self.pose_prior_mlp.parameters(), 'lr': 1e-4, "name": "pose_prior"},
            {'params': self.pose_point_mlp.parameters(), 'lr': 1e-4, "name": "pose_point"},
        ]
        if self.train_directions:
            l_struct.append({'params': [self.dir_raw], 'lr': cfg.position_lr_init * 0.1 * self.spatial_lr_scale, "name": "pts"})
        else:
            l_struct.append({'params': [self.points_raw], 'lr': cfg.position_lr_init * 0.1 * self.spatial_lr_scale, "name": "pts"})
        if self.train_features_rest:
            l_struct.append({'params': [self.features_rest_raw], 'lr': cfg.feature_lr / 20.0, "name": "f_rest"})
        if self.train_width:
            l_struct.append({'params': [self.width_raw], 'lr': cfg.scaling_lr, "name": "width"})
        if self.train_opacity:
            l_struct.append({'params': [self.opacity_raw], 'lr': cfg.opacity_lr, "name": "opacity"})

        self.pts_scheduler_args = get_expon_lr_func(lr_init=cfg.position_lr_init*0.1*self.spatial_lr_scale,
                                                    lr_final=cfg.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=cfg.position_lr_delay_mult,
                                                    max_steps=cfg.position_lr_max_steps)
        self.optimizer = torch.optim.Adam(l_struct, lr=0.0, eps=1e-15)     

        self.milestones = cfg['milestones']
        self.lrs = cfg['lrs']
        self.prior_optimizers = {
            'theta': torch.optim.Adam([self.theta], self.lrs['theta']),
            'G_raw': torch.optim.Adam(self.strands_generator.G_raw.parameters(), self.lrs['G_raw']),
            'G_superres': torch.optim.Adam(self.strands_generator.G_superres.parameters(), self.lrs['G_superres']),
            'beta': torch.optim.Adam([self.beta], self.lrs['beta']),
            'G_res': torch.optim.Adam(self.strands_generator.G_res.parameters(), self.lrs['G_res']),
        }
        for k, [iter_start, _] in self.milestones.items():
            for param_group in self.prior_optimizers[k].param_groups:
                if iter_start != 0:
                    print(f'Disabling optimization of {k}')
                    param_group['lr'] = 0.0
                    param_group['opt'] = 'disabled'
                else:
                    param_group['opt'] = 'enabled'

    @property
    def get_seg_label(self):
        seg_label_struct = torch.cat([torch.zeros_like(self.opacity), torch.zeros_like(self.opacity) ,torch.ones_like(self.opacity)], dim =-1)
        return seg_label_struct
    
    @property
    def get_hair_label(self):
        return torch.ones_like(self.opacity)
    
    @property
    def get_body_label(self):
        return torch.ones_like(self.opacity)

    def update_learning_rate(self, iter):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pts":
                lr = self.pts_scheduler_args(iter)
                param_group['lr'] = lr

        for k in self.prior_optimizers.keys():
            iter_start, iter_end = self.milestones[k]
            for param_group in self.prior_optimizers[k].param_groups:
                if iter >= iter_start and iter <= iter_end and param_group['opt'] == 'disabled':
                    print(f'Starting optimization of {k}')
                    param_group['lr'] = self.lrs[k]
                    param_group['opt'] = 'enabled'
                elif iter > iter_end and param_group['opt'] == 'enabled':
                    print(f'Ending optimization of {k}')
                    param_group['lr'] = 0.0
                    param_group['opt'] = 'disabled'

    # TODO: data should provide image_height and image_width and world_view_transform
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

        #dir3D = F.normalize(self.dir, dim=-1)
        dir2D = (self.dir[:, None, :] @ T)[:, 0]
        dir2D = torch.nn.functional.normalize(dir2D, dim=-1)

        return dir2D
    # TODO: dirpath and flame_mesh_dir should be provided instead of hardcoded
    def update_mesh_alignment_transform(self, dir_path = None, flame_mesh_dir = None):
        # Estimate the transform to align Perm canonical space with the scene
        print('Updating FLAME to Pinscreen alignment transform')
        # source_mesh = o3d.io.read_triangle_mesh(f'{dir_path}/data/flame_mesh_aligned_to_pinscreen.obj')
        source_mesh = o3d.io.read_triangle_mesh(f'assets/flame_mesh_aligned_to_pinscreen.obj')
        # target_mesh = o3d.io.read_triangle_mesh(flame_mesh_dir)
        source = torch.from_numpy(np.asarray(source_mesh.vertices))
        # target = torch.from_numpy(np.asarray(target_mesh.vertices)) 
        target = torch.from_numpy(np.load('datasets/mini_demo_dataset/031/FLAME_params/0000/vertices.npy').astype(np.double))
        # target = torch.from_numpy(np.load('datasets/mini_demo_dataset/031/params/0000/vertices.npy').astype(np.double))
        source = torch.cat([source, torch.ones_like(source[:, :1])], -1)
        target = torch.cat([target, torch.ones_like(target[:, :1])], -1)
        transform = (source.transpose(0, 1) @ source).inverse() @ source.transpose(0, 1) @ target
        self.transform = transform.cuda().unsqueeze(0).float()
        mesh_width = (target[4051, :3] - target[4597, :3]).norm() # 2 x distance between the eyes
        width_raw_new = self.width_raw * mesh_width / self.prev_mesh_width
        if self.train_width:
            optimizable_tensors = self.replace_tensor_to_optimizer(self.optimizer, width_raw_new, "width")
            self.width_raw = optimizable_tensors["width"]
        else:
            self.width_raw.data = width_raw_new
        self.prev_mesh_width = mesh_width

    def reset_strands(self):
        print('Resetting strands using the current weights of the prior')
        points_raw_new, dir_raw_new, self.origins_raw, _ = self.sample_strands_from_prior(self.num_strands)
        if self.train_directions:
            optimizable_tensors = self.replace_tensor_to_optimizer(self.optimizer, dir_raw_new, "pts")
            self.dir_raw = optimizable_tensors["pts"]
        else:
            optimizable_tensors = self.replace_tensor_to_optimizer(self.optimizer, points_raw_new, "pts")
            self.points_raw = optimizable_tensors["pts"]

    def sample_strands_from_prior(self, num_strands = -1, pose_params = None):
        # Subsample Perm strands if needed
        roots = self.roots
        if num_strands < self.num_strands and num_strands != -1:
            strands_idx = torch.randperm(self.num_strands)[:num_strands]
            roots = roots[strands_idx]
        else:
            strands_idx = torch.arange(self.num_strands)
        # TODO: use pose embedding to change theta and beta
        # theta [1, 8, 512], meaning [54] -> [8, 512]
        # beta [1, 14, 512]
        # roots [1, 10_140, 3]
        if pose_params is not None:
            pose_embedding = self.pos_embedding(pose_params)
            theta = self.theta + self.pose_prior_mlp(pose_embedding)
            beta = self.beta + self.pose_prior_mlp(pose_embedding)

        out = self.strands_generator(roots, self.theta, self.beta)
        pts_perm = out['strands'][0].position

        # Map strands into the scene coordinates
        pts = (torch.cat([pts_perm, torch.ones_like(pts_perm[..., :1])], dim=-1) @ self.transform)[..., :3]
        dir = pts[:, 1:] - pts[:, :-1]

        return pts[:, 1:], dir.view(-1, 3), pts[:, :1], strands_idx
    

    # set gaussian representation from hair strands
    def generate_hair_gaussians(self, num_strands = -1, skip_color = False, skip_smpl = False, backprop_into_prior = False, pose_params = None):
        if num_strands < self.num_strands and num_strands != -1:
            strands_idx = torch.randperm(self.num_strands)[:num_strands]
        else:
            strands_idx = torch.arange(self.num_strands)
        # only optimize prior
        if backprop_into_prior:
            self.points, self.dir, self.origins, strands_idx = self.sample_strands_from_prior(num_strands)
            self.points_origins = torch.cat([self.origins, self.points], dim=1)
            if num_strands == -1:
                num_strands = self.num_strands
        # directly optimize (structured) hair strand points
        else:      
            self.origins = self.origins_raw[strands_idx]
            if self.train_directions:
                self.dir = self.dir_raw.view(self.num_strands, self.strand_length - 1, 3)[strands_idx].view(-1, 3)
                self.points_origins = torch.cumsum(torch.cat([
                    self.origins, 
                    self.dir.view(num_strands, self.strand_length -1, 3)
                ], dim=1), dim=1)
            else:
                self.points = self.points_raw[strands_idx]
                self.points_origins = torch.cat([self.origins, self.points], dim=1)
                self.dir = (self.points_origins[:, 1:] - self.points_origins[:, :-1]).view(-1, 3)
        # Add dynamics to the hair strands
        # Points shift
        if pose_params is not None:
            pose_embedding = self.pos_embedding(pose_params)
            points = self.points_origins.view(-1, 3)
            pose_deform_input = torch.cat([self.pos_embedding(points).t(), 
                                            pose_embedding.unsqueeze(-1).repeat(1, points.shape[0])], 0)[None]
            pose_deform = self.pose_point_mlp(pose_deform_input)[0].t()
            self.points_origins = self.points_origins + pose_deform.view(num_strands, self.strand_length, 3)
            self.dir = (self.points_origins[:, 1:] - self.points_origins[:, :-1]).view(-1, 3)
        
        self.width = self.width_raw[strands_idx]
        self.opacity = self.opacity_raw.view(self.num_strands, self.strand_length - 1, 1)[strands_idx].view(-1, 1)
        if not skip_color:
            self.features_dc = self.features_dc_raw.view(self.num_strands, self.strand_length - 1, 1, 3)[strands_idx].view(-1, 1, 3)
            self.features_rest = self.features_rest_raw.view(self.num_strands, self.strand_length - 1, (self.max_sh_degree + 1) ** 2 - 1, 3).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3) 

        self.xyz = (self.points_origins[:, 1:] + self.points_origins[:, :-1]).view(-1, 3) * 0.5

        self.scales = torch.ones_like(self.xyz)
        self.scales[:, 0] = self.dir.norm(dim=-1) * 0.66
        self.scales[:, 1:] = self.width.repeat(1, self.strand_length - 1).view(-1, 1)

        self.seg_label = torch.zeros_like(self.xyz)

        if not skip_smpl and self.simplify_strands:
            # Run line simplification
            MAX_ITERATIONS = 4
            xyz = self.xyz.view(num_strands, self.strand_length - 1, 3)
            dir = self.dir.view(num_strands, self.strand_length - 1, 3)
            num_gaussians = xyz.shape[1]
            len = (dir**2).sum(-1)
            if not skip_color:
                features_dc = self.features_dc.view(num_strands, self.strand_length - 1, 3)
                features_rest = self.features_rest.view(num_strands, self.strand_length - 1, -1)
            scaling = self.scales.view(num_strands, self.strand_length - 1, 3)
            opacity = self.opacity.view(num_strands, self.strand_length - 1, 1)
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
                if (torch.quantile(err, self.quantile, dim=1) < self.width * self.aspect_ratio).float().mean() > 0.5:
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
            self.xyz = xyz.view(-1, 3)
            self.dir = dir.view(-1, 3)
            if not skip_color:
                self.features_dc = features_dc.view(-1, 1, 3)
                self.features_rest = features_rest.view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
            self.scales = scaling.view(-1, 3)
            self.opacity= opacity.view(-1, 1)
        
            if num_gaussians + 1 != self.prev_strand_length:
                print(f'Simplified strands from {self.prev_strand_length} to {num_gaussians + 1} points')
                self.prev_strand_length = num_gaussians + 1

        self.scales = self.scales_inverse_activation(self.scales)

        # Assign geometric features        
        self.rotation = parallel_transport(
            a=torch.cat(
                [
                    torch.ones_like(self.xyz[:, :1]),
                    torch.zeros_like(self.xyz[:, :2])
                ],
                dim=-1
            ),
            b=self.dir
        ).view(-1, 4) # rotation parameters that align x-axis with the segment direction


    def create_from_pcd(self, pcd, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        self.init_xyz = fused_point_cloud.clone()
        self.init_features = fused_color.clone()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.scales = nn.Parameter(scales.requires_grad_(True))
        self.rotation = nn.Parameter(rots.requires_grad_(True))
        self.opacity = nn.Parameter(opacities.requires_grad_(True))
        self.label_hair = nn.Parameter(inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")).requires_grad_(True))
        self.label_body = nn.Parameter(inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")).requires_grad_(True))
        self.seg_label = nn.Parameter(inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    # initialize the hair gaussian representation
    def create_hair_gaussians(self, scaling_factor = 1e-3):
        self.prev_strand_length = self.strand_length
        self.prev_mesh_width = 1.0
        with torch.no_grad():
            points_struct_raw, dir_struct_raw, self.origins_raw, _ = self.sample_strands_from_prior(self.num_strands)
        if self.train_directions:
            self.dir_raw = nn.Parameter(dir_struct_raw.contiguous().requires_grad_(True))
        else:
            self.points_raw = nn.Parameter(points_struct_raw.contiguous().requires_grad_(True))
        self.width_raw = nn.Parameter(torch.ones(self.num_strands, 1).cuda().contiguous().requires_grad_(True))
        self.width_raw.data *= scaling_factor
        self.opacity_raw = nn.Parameter(inverse_sigmoid(1.0 * torch.ones(self.num_strands * (self.strand_length - 1), 1, dtype=torch.float, device="cuda")))
        self.generate_hair_gaussians(skip_color=True, skip_smpl=True)

        # closest_idx = []
        # with torch.no_grad():
        #     print('Initializing hair Gaussians color using closest COLMAP points')
        #     for i in tqdm(range(self.xyz.shape[0] // 1_000 + (self.xyz.shape[0] % 1000 > 0))):
        #         closest_idx.append(((self.xyz[i*1_000 : (i+1)*1_000, None, :] - self.init_xyz[None, :, :])**2).sum(-1).amin(-1))
        # closest_idx = torch.cat(closest_idx).long()
        # features_dc = self.init_features[closest_idx].view(self.num_strands, self.strand_length - 1, 3).mean(1, keepdim=True)
        # features_dc = features_dc.repeat(1, self.strand_length - 1, 1).view(self.num_strands * (self.strand_length - 1), 3)      
        features_dc = torch.zeros( self.num_strands * (self.strand_length - 1), 3)

        self.features_dc_raw = nn.Parameter(features_dc[:, None, :].contiguous().cuda().requires_grad_(True))
        assert self.features_dc_raw.shape[0] == self.num_strands * (self.strand_length - 1)
        self.features_rest_raw = nn.Parameter(torch.zeros(self.num_strands * (self.strand_length - 1), (self.max_sh_degree + 1) ** 2 - 1, 3).cuda().requires_grad_(True))

    # TODO: remove the batch dimension since the GS does not suit for batch processing very much
    def generate(self,data):
        B = data['pose'].shape[0]
        
        hair_data = {}

        xyz = self.get_xyz.unsqueeze(0).repeat(B, 1, 1)
        scales = self.get_scales.unsqueeze(0).repeat(B, 1, 1)   
        rotation = self.get_rotation.unsqueeze(0).repeat(B, 1, 1)
        
        # need 32 channels for the color
        color = torch.zeros([B, self.xyz.shape[0], 32], device=xyz.device)
        # TODO: If we decide that hair only has diffuse color, then the following direction staff is not needed
        for b in range(B):
            # view dependent/independent color
            shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            dir_pp = (self.get_xyz - data['camera_center'][b].repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            color[b,:,:3] = torch.clamp_min(sh2rgb + 0.5, 0.0) 

        

        color[...,3:6] = self.get_seg_label.unsqueeze(0).repeat(B, 1, 1)
        opacity = self.get_opacity.unsqueeze(0).repeat(B, 1, 1)

        hair_data['xyz'] = xyz
        hair_data['color'] = color
        hair_data['scales'] = scales
        hair_data['rotation'] = rotation
        hair_data['opacity'] = opacity

        return hair_data
