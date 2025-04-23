

import os
import sys
import numpy as np
import torch
from torch import nn
import tqdm
from plyfile import PlyData, PlyElement
from einops import rearrange
import pickle

from simple_knn._C import distCUDA2
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
import open3d as o3d

from lib.utils.general_utils import inverse_sigmoid, eval_sh,RGB2SH
from lib.module.GaussianBaseModule import GaussianBaseModule
from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/../../ext/perm/src')
from hair.hair_models import Perm


# # TODO: move to lib/utils/general_utils.py

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

        self.register_buffer('origins_raw', torch.empty(0))
        self.origins_raw = torch.empty(0)
        
        self.points_raw = torch.empty(0)
        self.dir_raw = torch.empty(0)
        self.features_dc_raw = torch.empty(0)
        self.features_rest_raw = torch.empty(0)
        self.opacity_raw = torch.empty(0)
        self.width_raw = torch.empty(0)

        # for dynamic hair, points deformation should be used only when directly training point_raw
        self.points_deform_accumulate = torch.empty(0)
        self.points_velocity = torch.empty(0)
        self.direct_train_optical_flow = False
        self.optical_flow_3D_lift = nn.Parameter(torch.zeros(self.num_strands, self.strand_length - 1, 3).cuda()) 
        
        # self.pose_color_mlp = MLP(cfg.pose_color_mlp, last_op=None)
        # self.pose_attributes_mlp = MLP(cfg.pose_attributes_mlp, last_op=None)
        # self.pose_deform_mlp = MLP(cfg.pose_deform_mlp, last_op=nn.Tanh())
        self.pose_point_mlp = MLP(cfg.pose_point_mlp, last_op=None)
        self.pose_prior_mlp = MLP(cfg.pose_prior_mlp, last_op=None)
        
        self.pose_query_mlp = MLP([54, 128, 128], last_op=None)
        self.pose_key_mlp = MLP([54, 128, 128], last_op=None)
        self.pose_value_mlp = MLP([54, 128, 128], last_op=None)
        self.pose_deform_attention = torch.nn.MultiheadAttention(128, 8, dropout=0.1)
    
        # pose [6] -> pose embedding [54]
        self.pos_embedding, _ = get_embedder(cfg.pos_freq)


        self.transform = nn.Parameter(torch.eye(4).cuda())

        self.create_hair_gaussians(cfg.strand_scale)

        # TODO: By printing the value of Gaussian Hair cut. Need to get this value in this project
        self.cameras_extent = 4.907987451553345
        self.spatial_lr_scale = self.cameras_extent

        # TODO: Add learning rate for each parameter
        l_struct = [
            {'params': [self.features_dc_raw], 'lr': cfg.feature_lr, "name": "f_dc"},
            {'params': self.pose_prior_mlp.parameters(), 'lr': 1e-4, "name": "pose_prior"},
            {'params': self.pose_query_mlp.parameters(), 'lr': 1e-4, "name": "pose_query"},
            {'params': self.pose_key_mlp.parameters(), 'lr': 1e-4, "name": "pose_key"},
            {'params': self.pose_value_mlp.parameters(), 'lr': 1e-4, "name": "pose_value"},
            {'params': self.pose_deform_attention.parameters(), 'lr': 1e-4, "name": "pose_deform_attention"},
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
        # TODO: transform actually should be more related to prior, since it's used by prior to generate strand 
        # better add it to prior optimizer
        # l_struct.append({'params': [self.transform], 'lr': 1e-4 , "name": "transform"})
        l_struct.append({'params': [self.transform], 'lr': 0 , "name": "transform"})

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

    def epoch_start(self):
        self.points_deform_accumulate = torch.zeros_like(self.points_raw)
        self.points_velocity = torch.zeros_like(self.points_raw)
        self.pre_pose_deform = None
        self.optical_flow_3D = torch.ones_like(self.points_raw)
        self.optical_flow_3D_lift = nn.Parameter(torch.zeros(self.num_strands, self.strand_length - 1, 3).cuda())

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
    def get_direction_2d(self, fovx, fovy, height, width, viewmatrix, xyz = None, dir = None, normalize = True):
        mean = self.get_xyz if xyz is None else xyz
    
        tan_fovx = torch.tan(fovx * 0.5)
        tan_fovy = torch.tan(fovy * 0.5)


        focal_y = height / (2.0 * tan_fovy)
        focal_x = width / (2.0 * tan_fovx)

        # viewmatrix = viewpoint_camera['world_view_transform']

        t = (mean[:, None, :] @ viewmatrix[None, :3, :3] + viewmatrix[None, [3], :3])[:, 0]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

        limx = 1.3 * tan_fovx
        limy = 1.3 * tan_fovy
        txtz = tx / tz
        tytz = ty / tz
        
        tx = torch.clamp(txtz, min=-limx, max=limx) * tz
        ty = torch.clamp(tytz, min=-limy, max=limy) * tz

        zeros = torch.zeros_like(tz)

        # remove z here 
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
        dir = self.dir if dir is None else dir
        dir2D = (dir[:, None, :] @ T)[:, 0]

        if normalize:
            dir2D = torch.nn.functional.normalize(dir2D, dim=-1)

        return dir2D
    
    # TODO: backward twice error?
    def mesh_distance_loss(self):
        return 0
        # strand_num, 3
        roots = self.origins_raw.view(-1, 3)
        vecrtices = self.FLAME_mesh.verts_packed()
        square_dist = ((roots[:, None] - vecrtices[None])**2).sum(dim=-1)
        square_dist = square_dist.min(dim=-1)[0]
        return square_dist.mean()
    
    def sign_distance_loss(self, max_num_points = 10000):
        # negative relu
        # calc distance points to mesh 
        
        # self.points_raw is of shape (strand_num, strand_length - 1, 3)
        # select finite points
        num_points = self.points_raw.shape[0] * self.points_raw.shape[1]
        if num_points > max_num_points:
            indices = torch.randperm(num_points)[:max_num_points]
        else:
            indices = torch.arange(num_points)

        points = self.points_raw.reshape(-1, 3)[indices]
        vertices = self.FLAME_mesh.verts_packed()  # (V, 3)
        faces = self.FLAME_mesh.faces_packed()  # (F, 3)
        # (N, 1), bool, true means inside
        inside = ray_stabbing(points, vertices, faces).view(-1).bool()


        # only inside points get the loss
        points = points[inside]

        # B, N, 3
        points = points.view(1, -1, 3)
        point_cloud = Pointclouds(points)
        unsigned_dist = point_mesh_face_distance(self.FLAME_mesh, point_cloud)

        return unsigned_dist 

    
    # TODO: use root distance or point distance?
    def knn_feature_loss(self):
        # strand_num, 3
        roots = self.origins_raw.view(-1, 3)
        # don't need to do it every time
        # _, indices = distCUDA2(roots, roots)
        square_dist = ((roots[:, None] - roots[None])**2).sum(dim=-1)
        _, indices = square_dist.topk(2, dim=-1, largest=False)
        indices = indices[:, 1]

        # should get idx of size [num_strands], indicating the nearest neighbor
        feature_dc = self.features_dc_raw.view(self.num_strands, self.strand_length -1, -1)
        feature_diff = (feature_dc - feature_dc[indices]) ** 2
        
        return feature_diff.mean()
        return 0
    
    # TODO: may limit the expressive capacity of the model
    # what this loss try to solve --- hair gaussian mixed with body gaussian
    # the color feature of points on the same strand are assumed to be similar
    def strand_feature_loss(self):
        feature_dc = self.features_dc_raw.view(self.num_strands, self.strand_length -1, -1)
        feature_diff = (feature_dc - feature_dc.mean(dim=1, keepdim=True)) ** 2
        
        return feature_diff.mean()

    def random_set_transparent(self, ratio = 0.2):
        if self.train_opacity:
            reset_strand_num = int(self.num_strands * ratio)
            indices = torch.randperm(self.num_strands)[:reset_strand_num]
            opacity_raw = self.opacity_raw.view(self.num_strands, self.strand_length - 1).detach().clone()
            opacity_raw[indices] = self.inverse_opacity_activation(torch.zeros_like(opacity_raw[indices]) + 1e-5)
            opacity_raw = opacity_raw.view(-1, 1)
            optimizable_tensors = self.replace_tensor_to_optimizer(self.optimizer, opacity_raw, "opacity")
            self.opacity_raw = optimizable_tensors["opacity"]
        

    # should better be called when training the raw data, instead of prior
    def remove_transparent(self):

        # Option 1, remove the whole strand according to mean opacity or root opacity

        # Option 2, from root to top, remove the transparent part and keep the rest while making the last point root
        pass


    # TODO: dirpath and flame_mesh_dir should be provided instead of hardcoded
    def update_mesh_alignment_transform(self, R, T, S, dir_path = None, flame_mesh_path = 'datasets/mini_demo_dataset/031/FLAME_params/0000/mesh_0.obj'):
        # Estimate the transform to align Perm canonical space with the scene
        print('Updating FLAME to Pinscreen alignment transform')
        # source_mesh = o3d.io.read_triangle_mesh(f'{dir_path}/data/flame_mesh_aligned_to_pinscreen.obj')
        source_mesh = o3d.io.read_triangle_mesh(f'assets/flame_mesh_aligned_to_pinscreen.obj')
        # target_mesh = o3d.io.read_triangle_mesh(flame_mesh_dir)
        source = torch.from_numpy(np.asarray(source_mesh.vertices))
        # target = torch.from_numpy(np.asarray(target_mesh.vertices))
        # already posed mesh, for hair we don't do the pose transformation again in generate() function
        # target = torch.from_numpy(np.load('datasets/mini_demo_dataset/031/FLAME_params/0000/vertices.npy').astype(np.double))
        target_mesh = o3d.io.read_triangle_mesh(flame_mesh_path)
        target = torch.from_numpy(np.asarray(target_mesh.vertices))
        
        # tensor double
        R = R.to(torch.double)
        T = T.to(torch.double) 
        S = S.to(torch.double)
        target = (torch.matmul(target- T, R)) / S 
        # shrink the mesh a little bit so that most hair roots are roughly on/inside the surface.
        # target = target * 0.95

        # pytorch3d canonical mesh
        faces = torch.from_numpy(np.asarray(target_mesh.triangles)).cuda() 
        target_mesh = Meshes(verts= target[None].cuda().to(torch.float), faces=faces[None])
        self.FLAME_mesh = target_mesh


        # hair_list_filename = "assets/FLAME/hair_list.pkl"
        # with open(hair_list_filename, 'rb') as f:
        #     scalp_indices = torch.tensor(pickle.load(f))
        #     self.scalp_indices = scalp_indices
        #     source = source[scalp_indices]  
        #     target = target[scalp_indices] 


        # # create a mesh from xyz
        hair_mesh = o3d.geometry.TriangleMesh()
        # source mesh
        source_mesh = o3d.geometry.TriangleMesh()
        source_mesh.vertices = o3d.utility.Vector3dVector(source)
        source_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh('Init_source_mesh.ply', source_mesh)

        # create a mesh from the target vertices for debugging
        target_mesh = o3d.geometry.TriangleMesh()
        target_mesh.vertices = o3d.utility.Vector3dVector(target)
        target_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh('Init_target_mesh.ply', target_mesh) 
        

        source = torch.cat([source, torch.ones_like(source[:, :1])], -1)
        target = torch.cat([target, torch.ones_like(target[:, :1])], -1)
        transform = (source.transpose(0, 1) @ source).inverse() @ source.transpose(0, 1) @ target
        self.transform.data = transform.detach().clone().cuda().float()

        # don't simply use detach here, returned value from detach shares the same memory with the original tensor
        self.init_transform = transform.detach().clone().cuda().float()
        

        # transformed hair
        transformed_hair = (torch.cat([self.xyz, torch.ones_like(self.xyz[:, :1])], -1) @ self.transform).detach().cpu().numpy()[:, :3]
        # transformed_hair = (self.xyz * s @ R.cuda().float() + t.cuda().float()).detach().cpu().numpy()
        hair_mesh.vertices = o3d.utility.Vector3dVector(transformed_hair)
        hair_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh('Init_transformed_hair.ply', hair_mesh)

        # transformed source
        transformed_source = (torch.cat([source[:, :3], torch.ones_like(source[:, :1])], -1) @ transform).detach().cpu().numpy()[:, :3]
        # transformed_source = (source[:, :3] * s @ R + t).detach().cpu().numpy()
        hair_mesh.vertices = o3d.utility.Vector3dVector(transformed_source)
        hair_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh('Init_transformed_source.ply', hair_mesh)

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
        
        # TODO: add strand length control, Later maybe adptively change the strand length on certain area
        # maybe choose the points evenly, instead of simply cut the points
        evenly_indices = torch.linspace(0, 99, self.strand_length).long()
        pts_perm = pts_perm[:, evenly_indices]
        # pts_perm = pts_perm[:, :self.strand_length]

        # Map strands into the scene coordinates
        pts = (torch.cat([pts_perm, torch.ones_like(pts_perm[..., :1])], dim=-1) @ self.transform)[..., :3]
        dir = pts[:, 1:] - pts[:, :-1]

        return pts[:, 1:], dir.view(-1, 3), pts[:, :1], strands_idx
    
    # TODO: according to split num or length threshold?
    def split_strands(self, length_threshold):
        # Split the strands into groups according the length
        # [strand_num, strand_length-1]
        long_axis = 0.66 * self.dir.norm(dim=-1).view(self.num_strands, self.strand_length-1) 
        # [strand_num, 1]
        short_axis = self.width.view(self.num_strands, -1)
        # [strand_num, strand_length-1]
        length_ratio = long_axis / short_axis
        # [strand_num]
        length_ratio = length_ratio.mean(dim=-1)
        groups = []

        while (length_threshold > 0.1):
            nxt_length_threshold = length_threshold / 2
            candidates = torch.logical_and(length_ratio <= length_threshold, length_ratio > nxt_length_threshold)
            if candidates.sum() == 0:
                break
            groups.append(candidates)
            length_threshold = nxt_length_threshold

        return groups
    
    def shorten_strands(self, groups):

        xyz_list = []
        scales_list = []
        opacity_list = []
        seg_label_list = []
        rotation_list = []
        features_dc_list = []
        features_rest_list = []

        for idx, group in enumerate(groups):
            
            target_len = max(2, int(self.strand_length / (2 ** (idx + 1))))
            
            # strand_length includes the root point, so target_length - 1 is the real strand length
            evenly_indices = torch.linspace(0, self.strand_length - 1, target_len).long()
            # create mask for indexing
            # indice_mask = torch.zeros(self.strand_length, dtype=torch.bool)
            # indice_mask[evenly_indices] = True 

            points_origin = self.points_origins[group].view(-1, self.strand_length, 3)[:, evenly_indices]
            dir = (points_origin[:, 1:] - points_origin[:, :-1]).view(-1, 3)
            
            xyz = (points_origin[:, 1:] + points_origin[:, :-1]).view(-1, 3) * 0.5
            scales = torch.ones_like(xyz)
            # breakpoint()
            scales[:, 0] = 0.66 * dir.norm(dim=-1).view(-1)
            scales[:, 1:] = self.width[group].repeat(1, target_len - 1).view(-1, 1)
            scales = self.scales_inverse_activation(scales)
            # opacity = self.opacity.view(-1, self.strand_length - 1, 1)[group, evenly_indices[:-1], :].view(-1, 1)
            opacity = self.opacity.view(-1, self.strand_length - 1, 1)[group][:, evenly_indices[:-1], :].view(-1, 1)
            seg_label = torch.zeros_like(xyz)
            rotation = parallel_transport(
            a=torch.cat(
                [
                    torch.ones_like(xyz[:, :1]),
                    torch.zeros_like(xyz[:, :2])
                ],
                dim=-1
            ),
            b=dir
            ).view(-1, 4) # rotation parameters that align x-axis with the segment direction

            features_dc = self.features_dc.view(-1, self.strand_length - 1, 3)[group][:, evenly_indices[:-1]].view(-1, 1, 3)
            features_rest = self.features_rest.view(-1, self.strand_length - 1, (self.max_sh_degree + 1) ** 2 - 1, 3)[group][:,evenly_indices[:-1]].view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)

            xyz_list.append(xyz)
            scales_list.append(scales)
            opacity_list.append(opacity)
            seg_label_list.append(seg_label)
            rotation_list.append(rotation)
            features_dc_list.append(features_dc)
            features_rest_list.append(features_rest)
        
        self.xyz = torch.cat(xyz_list, dim=0)
        self.scales = torch.cat(scales_list, dim=0)
        self.opacity = torch.cat(opacity_list, dim=0)
        self.seg_label = torch.cat(seg_label_list, dim=0)
        self.rotation = torch.cat(rotation_list, dim=0)
        self.features_dc = torch.cat(features_dc_list, dim=0)
        self.features_rest = torch.cat(features_rest_list, dim=0)

        return 


    def get_pose_deform(self, all_pose):
        if len(all_pose) == 0:
            return 0

        cur_pose = all_pose[-1]
        pose_embedding = self.pos_embedding(cur_pose[None])
        # L, 54
        all_pose_embedding = self.pos_embedding(all_pose)
        # L, 54 -> L, 54, 1 -> L, 128 , 1 -> L, 128 
        pose_query = self.pose_query_mlp(pose_embedding[..., None])[..., 0]
        pose_key = self.pose_key_mlp(all_pose_embedding[..., None])[..., 0]
        pose_value = self.pose_value_mlp(all_pose_embedding[...,None])[..., 0]
        # 1, 128 
        pose_deform_embedding, _ = self.pose_deform_attention(pose_query, pose_key, pose_value)
        # 1, 128 -> 128 -> 54
        pose_deform_embedding = pose_deform_embedding[0, :54] 
        # # [strand_num, strand_length-1, 3] -> [strand_num * (strand_length-1), 3]
        points = self.points.contiguous().view(-1, 3)
        # [27, point_num] + [54, point_num] -> [81, point_num]
        pose_deform_input = torch.cat([self.pos_embedding(points).t(),
                                        pose_deform_embedding.unsqueeze(-1).repeat(1, points.shape[0])], 0)[None]
        pose_deform = self.pose_point_mlp(pose_deform_input)[0].t()

        return pose_deform.view(self.num_strands, self.strand_length - 1, 3)

    # set gaussian representation from hair strands
    def generate_hair_gaussians(self, num_strands = -1, skip_color = False, skip_smpl = False, backprop_into_prior = False, poses_history = None, pose = None, scale = None, given_optical_flow = None, accumulate_optical_flow = None):
        # determine the number of strands to sample
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
                self.points = self.points_origins[:, 1:]
            else:
                self.points = self.points_raw[strands_idx]
                self.points_origins = torch.cat([self.origins, self.points], dim=1)
                self.dir = (self.points_origins[:, 1:] - self.points_origins[:, :-1]).view(-1, 3)
        

        # from canonical space to world space, used in orginal codes.
        if pose is not None and scale is not None:
            # add batch dimension
            R = so3_exponential_map(pose[None, :3])
            T = pose[None,None, 3:]
            S = scale.view(1)
            points = self.points.reshape(1, -1, 3)
            origins = self.origins.reshape(1, -1, 3)
            points = torch.bmm(points * S, R.permute(0, 2, 1)) + T
            origins = torch.bmm(origins * S, R.permute(0, 2, 1)) + T
            self.points = points.view(num_strands, self.strand_length - 1, 3)
            self.origins = origins.view(num_strands, 1, 3)
            self.points_origins = torch.cat([self.origins, self.points], dim=1)
            self.dir = (self.points_origins[:, 1:] - self.points_origins[:, :-1]).view(-1, 3)

        # Add dynamics to the hair strands
        # Points shift
        if given_optical_flow is not None:
            
            accumulate_optical_flow = accumulate_optical_flow.view(num_strands, self.strand_length - 1, 3)
            optical_flow = given_optical_flow.view(num_strands, self.strand_length - 1, 3)
            points = self.points + optical_flow
            points = points + accumulate_optical_flow
            points = points.reshape(num_strands, self.strand_length - 1, 3)
            self.points = points
            self.points_origins = torch.cat([self.origins, points], dim=1)
            self.dir = (self.points_origins[:, 1:] - self.points_origins[:, :-1]).view(-1, 3)

        # elif poses_history is not None:
        #     # point : (frame_num-1, 3)

        #     pose_deform = self.get_pose_deform(poses_history)

        #     points = self.points + pose_deform.view(num_strands, self.strand_length - 1, 3)

        #     self.optical_flow_3D = torch.zeros_like(points) if self.pre_pose_deform is None else pose_deform - self.pre_pose_deform
        #     self.pre_pose_deform = pose_deform.detach()

        #     self.points = points
        #     self.points_origins = torch.cat([self.origins, points], dim=1)
        #     self.dir = (self.points_origins[:, 1:] - self.points_origins[:, :-1]).view(-1, 3)
        
        # TODO: scale is negative sometimes!
        self.width = self.width_raw[strands_idx] if scale is None else self.width_raw[strands_idx] * max(scale, 0.4)
        self.opacity = self.opacity_raw.view(self.num_strands, self.strand_length - 1, 1)[strands_idx].view(-1, 1)
        if not skip_color:
            self.features_dc = self.features_dc_raw.view(self.num_strands, self.strand_length - 1, 1, 3)[strands_idx].view(-1, 1, 3)
            self.features_rest = self.features_rest_raw.view(self.num_strands, self.strand_length - 1, (self.max_sh_degree + 1) ** 2 - 1, 3).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3) 

        # TODO: split the hair strands to different groups based on length condition
        # for each group, decrease the strand point number.
        # can do it iteratively util the length condition is satisfied
        # finlly merge the results.
        # ideally should get something like [ [group1_strand_indices], [group2_strand_indices], ...]

        self.xyz = (self.points_origins[:, 1:] + self.points_origins[:, :-1]).view(-1, 3) * 0.5

        self.scales = torch.ones_like(self.xyz)
        # chance that two points are too close
        self.scales[:, 0] = self.dir.norm(dim=-1) * 0.66 + 1e-6
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


    # given pose, scale. return the posed strand points
    def get_posed_points(self,num_strands = -1, backprop_into_prior = False, poses_history = None, pose = None, scale = None, given_optical_flow = None, accumulate_optical_flow = None):

        # determine the number of strands to sample
        if num_strands < self.num_strands and num_strands != -1:
            strands_idx = torch.randperm(self.num_strands)[:num_strands]
        else:
            strands_idx = torch.arange(self.num_strands)

        # only optimize prior
        if backprop_into_prior:
            points, dir, origins, strands_idx = self.sample_strands_from_prior(num_strands)
            points_origins = torch.cat([origins, points], dim=1)
            if num_strands == -1:
                num_strands = self.num_strands
        # directly optimize (structured) hair strand points
        else:      
            origins = self.origins_raw[strands_idx]
            if self.train_directions:
                dir = self.dir_raw.view(self.num_strands, self.strand_length - 1, 3)[strands_idx].view(-1, 3)
                points_origins = torch.cumsum(torch.cat([
                    origins, 
                    dir.view(num_strands, self.strand_length -1, 3)
                ], dim=1), dim=1)
                points = points_origins[:, 1:]
            else:
                points = self.points_raw[strands_idx]
                points_origins = torch.cat([origins, points], dim=1)
                dir = (points_origins[:, 1:] - points_origins[:, :-1]).view(-1, 3)
        

        # from canonical space to world space, used in orginal codes.
        if pose is not None and scale is not None:
            # add batch dimension
            R = so3_exponential_map(pose[None, :3])
            T = pose[None,None, 3:]
            S = scale.view(1)
            
            points = points.reshape(1, -1, 3)
            origins = origins.reshape(1, -1, 3)
            points = torch.bmm(points * S, R.permute(0, 2, 1)) + T
            origins = torch.bmm(origins * S, R.permute(0, 2, 1)) + T
            
            points = points.view(num_strands, self.strand_length - 1, 3)
            origins = origins.view(num_strands, 1, 3)
            points_origins = torch.cat([self.origins, self.points], dim=1)
            dir = (self.points_origins[:, 1:] - self.points_origins[:, :-1]).view(-1, 3)

        # Add dynamics to the hair strands
        # Points shift
        if given_optical_flow is not None:
            
            accumulate_optical_flow = accumulate_optical_flow.view(num_strands, self.strand_length - 1, 3)
            optical_flow = given_optical_flow.view(num_strands, self.strand_length - 1, 3)
            
            points = points + optical_flow + accumulate_optical_flow
            
            points = points.reshape(num_strands, self.strand_length - 1, 3)
            points_origins = torch.cat([origins, points], dim=1)
            dir = (points_origins[:, 1:] - points_origins[:, :-1]).view(-1, 3)

        elif poses_history is not None:
            # point : (frame_num-1, 3)

            pose_deform = self.get_pose_deform(poses_history)
            points = points + pose_deform.view(num_strands, self.strand_length - 1, 3)

            points_origins = torch.cat([origins, points], dim=1)
            dir = (points_origins[:, 1:] - points_origins[:, :-1]).view(-1, 3)

        return {'points': points, 'dir': dir, 'origins': origins, 'points_origins': points_origins, 'strands_idx': strands_idx}
        

    # given strand points, return xyz, scales, opacity, rotation, features_dc, features_rest
    def generate_hair_gaussians_from_points(self, strands_idx, points_origins = None, dir = None, skip_color = False, skip_smpl = False, scale = None):
        dir = dir if dir is not None else self.dir
        points_origins = points_origins if points_origins is not None else self.points_origins

        # TODO: scale is negative sometimes!
        self.width = self.width_raw[strands_idx] if scale is None else self.width_raw[strands_idx] * max(scale, 0.4)

        opacity = self.opacity_raw.view(self.num_strands, self.strand_length - 1, 1)[strands_idx].view(-1, 1)

        features_dc = self.features_dc_raw.view(self.num_strands, self.strand_length - 1, 1, 3)[strands_idx].view(-1, 1, 3)
        features_rest = self.features_rest_raw.view(self.num_strands, self.strand_length - 1, (self.max_sh_degree + 1) ** 2 - 1, 3).view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3) 

        # TODO: split the hair strands to different groups based on length condition
        # for each group, decrease the strand point number.
        # can do it iteratively util the length condition is satisfied
        # finlly merge the results.
        # ideally should get something like [ [group1_strand_indices], [group2_strand_indices], ...]

        xyz = (points_origins[:, 1:] + points_origins[:, :-1]).view(-1, 3) * 0.5

        scales = torch.ones_like(xyz)
        # chance that two points are too close
        scales[:, 0] = dir.norm(dim=-1) * 0.66 + 1e-6
        scales[:, 1:] = self.width.repeat(1, self.strand_length - 1).view(-1, 1)
        
        if not skip_smpl and self.simplify_strands:
            # Run line simplification
            MAX_ITERATIONS = 4
            xyz = xyz.view(-1, self.strand_length - 1, 3)
            dir = dir.view(-1, self.strand_length - 1, 3)
            num_gaussians = xyz.shape[1]
            len = (dir**2).sum(-1)
            if not skip_color:
                features_dc = features_dc.view(-1, self.strand_length - 1, 3)
                features_rest = features_rest.view(-1, self.strand_length - 1, -1)
            scaling = scales.view(-1, self.strand_length - 1, 3)
            opacity = opacity.view(-1, self.strand_length - 1, 1)
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
            xyz = xyz.view(-1, 3)
            dir = dir.view(-1, 3)
            if not skip_color:
                features_dc = features_dc.view(-1, 1, 3)
                features_rest = features_rest.view(-1, (self.max_sh_degree + 1) ** 2 - 1, 3)
            scales = scaling.view(-1, 3)
            opacity= opacity.view(-1, 1)
        
            if num_gaussians + 1 != self.prev_strand_length:
                print(f'Simplified strands from {self.prev_strand_length} to {num_gaussians + 1} points')
                self.prev_strand_length = num_gaussians + 1

        scales = self.scales_inverse_activation(scales)

        # Assign geometric features        
        rotation = parallel_transport(
            a=torch.cat(
                [
                    torch.ones_like(xyz[:, :1]),
                    torch.zeros_like(xyz[:, :2])
                ],
                dim=-1
            ),
            b= dir
        ).view(-1, 4) # rotation parameters that align x-axis with the segment direction

        return {'xyz': xyz, 'scales': scales, 'opacity': opacity, 'rotation': rotation, 'features_dc': features_dc, 'features_rest': features_rest}


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
        if self.train_opacity:
            self.opacity_raw = nn.Parameter(inverse_sigmoid(0.5 * torch.ones(self.num_strands * (self.strand_length - 1), 1, dtype=torch.float, device="cuda")).contiguous().requires_grad_(True))
        else:
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
    # TODO: maybe first do the pose deformation then do the hair genearation
    def generate(self,data):
        
        if len(data['pose'].shape) == 1:
            batched = False
            B = 1
            pose = data['pose'].unsqueeze(0)
            images = data['images'].unsqueeze(0)
            camera_center = data['camera_center'].unsqueeze(0)
            poses_history = data['poses_history'].unsqueeze(0)
            fovx = data['fovx'].unsqueeze(0)
            fovy = data['fovy'].unsqueeze(0)
            world_view_transform = data['world_view_transform'].unsqueeze(0)
        else:
            batched = True
            B = data['pose'].shape[0]
            pose = data['pose']
            images = data['images']
            camera_center = data['camera_center']
            poses_history = data['poses_history']
            fovx = data['fovx']
            fovy = data['fovy']
            world_view_transform = data['world_view_transform']
            
        
        hair_data = {}

        xyz = self.get_xyz.unsqueeze(0).repeat(B, 1, 1)
        scales = self.get_scales.unsqueeze(0).repeat(B, 1, 1)   
        rotation = self.get_rotation.unsqueeze(0).repeat(B, 1, 1)
        dir = self.dir.unsqueeze(0).repeat(B, 1, 1)
        
        # need 32 channels for the color
        # color = torch.zeros([B, self.xyz.shape[0], 16], device=xyz.device)
        color = torch.zeros([B, self.xyz.shape[0], 32], device=xyz.device)
        # TODO: If we decide that hair only has diffuse color, then the following direction staff is not needed
        for b in range(B):
            # view dependent/independent color
            shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            dir_pp = (self.get_xyz - camera_center[b].repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            color[b,:,:3] = torch.clamp_min(sh2rgb + 0.5, 0.0) 

        
        color[...,3:6] = self.get_seg_label.unsqueeze(0).repeat(B, 1, 1)
        opacity = self.get_opacity.unsqueeze(0).repeat(B, 1, 1)

        # velocity = self.optical_flow_3D.view(1, -1, 3).repeat(B, 1, 1)


        pre_poses = poses_history[0, :-1, :]
        # # from canonical space to world space, used in orginal codes.
        # if 'pose' in data:
        #     R = so3_exponential_map(data['pose'][:, :3])
        #     T = data['pose'][:, None, 3:]
        #     S = data['scale'].view(1)
        #     # R = so3_exponential_map(data['flame_pose'][:, :3])
        #     # T = data['flame_pose'][:, None, 3:]
        #     # S = data['flame_scale'].view(1)
        #     xyz = torch.bmm(xyz * S, R.permute(0, 2, 1)) + T
        #     # 
        #     # pre_xyz = torch.bmm(canonical_xyz * S, R.permute(0, 2, 1)) + T

        #     dir = torch.bmm(dir, R.permute(0, 2, 1))
        #     velocity = torch.bmm(velocity, R.permute(0, 2, 1))


        #     rotation_matrix = quaternion_to_matrix(rotation)
        #     rotation_matrix = rearrange(rotation_matrix, 'b n x y -> (b n) x y')
        #     R = rearrange(R.unsqueeze(1).repeat(1, rotation.shape[1], 1, 1), 'b n x y -> (b n) x y')
        #     rotation_matrix = rearrange(torch.bmm(R, rotation_matrix), '(b n) x y -> b n x y', b=B)
        #     rotation = matrix_to_quaternion(rotation_matrix)

        #     scales = scales * S
        
        # TODO: get direction 2d from xyz and direction
        dir2D = []
        velocity2D = []
        for b in range(B):
            image_height =  images.shape[2]
            image_width = images.shape[3]
            dir2D.append(self.get_direction_2d(fovx[b], fovy[b], 
                                               image_height, image_width,
                                               world_view_transform[b], 
                                               xyz[b], dir[b]))
            # # TODO, velocity should not be normalized
            # velocity2D.append(self.get_direction_2d(fovx[b], fovy[b], 
            #                                         image_height, image_width,
            #                                         world_view_transform[b], 
            #                                         xyz[b], velocity[b]))
        dir2D = torch.stack(dir2D, dim=0)
        color[..., 6:9] = dir2D

        # velocity2D = torch.stack(velocity2D, dim=0)
        # color[..., 9:12] = velocity2D

        hair_data['xyz'] = xyz
        hair_data['color'] = color
        hair_data['scales'] = scales
        hair_data['rotation'] = rotation
        hair_data['opacity'] = opacity

        return hair_data