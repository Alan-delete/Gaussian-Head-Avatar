# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead
import open3d as o3d

from lib.module.gaussian_model import GaussianModel
from lib.utils.graphics_utils import compute_face_orientation
from lib.utils.general_utils import inverse_sigmoid, eval_sh
from lib.network.PositionalEmbedding import get_embedder
from lib.network.MLP import MLP

# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()

        self.pos_embedding, _ = get_embedder(multires = 4)
        self.pose_deform_mlp = MLP([81, 256, 256, 3], last_op=nn.Tanh()).cuda()
        # initialize the pose_deform_mlp as zero
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.zeros_(m.weight)
                m.bias.data.fill_(0.001)
        self.pose_deform_mlp.apply(init_weights)
        

    def disable_static_parameters(self):
        if self.flame_param is not None:
            self.flame_param['static_offset'].requires_grad = False
            self.flame_param['shape'].requires_grad = False

        self._xyz.requires_grad = False
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._opacity.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
    

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if 'static_offset' in meshes[0]:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']).squeeze(0),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        # verts, verts_cano = self.flame_model(
        #     flame_param['shape'][None, ...],
        #     flame_param['expr'][[timestep]],
        #     flame_param['rotation'][[timestep]],
        #     flame_param['neck_pose'][[timestep]],
        #     flame_param['jaw_pose'][[timestep]],
        #     flame_param['eyes_pose'][[timestep]],
        #     flame_param['translation'][[timestep]],
        #     zero_centered_at_root_node=False,
        #     return_landmarks=False,
        #     return_verts_cano=True,
        #     static_offset=flame_param['static_offset'],
        #     dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        # )
        verts, verts_cano, landmarks = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=True,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.landmarks = landmarks
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # static_offset
        if not self.disable_flame_static_offset:
            self.flame_param['static_offset'].requires_grad = True
            param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
            self.optimizer.add_param_group(param_static_offset)

        # add pose-dependent dynamic offset
        pose_deformer_params = {'params': self.pose_deform_mlp.parameters(), 'lr': 1e-5, "name": "pose_deform_mlp"}
        self.optimizer.add_param_group(pose_deformer_params)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def save_ply(self, path):
        super().save_ply(path)

        # npz_path = Path(path).parent / "flame_param.npz"
        npz_path = path.replace('.ply', '_flame_param.npz') 
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

        # mesh_path = Path(path).parent / "head_mesh_latest.ply"
        # generic_mesh = o3d.geometry.TriangleMesh()
        # generic_mesh.vertices = o3d.utility.Vector3dVector(self.verts.squeeze(0).detach().cpu().numpy())
        # generic_mesh.triangles = o3d.utility.Vector3iVector(self.faces.squeeze(0).detach().cpu().numpy())
        # generic_mesh.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(mesh_path, generic_mesh)

        # posed_gaussian_point_cloud_path = Path(path).parent / "posed_gaussian_point_cloud.ply"
        # xyz = self.get_xyz.squeeze(0).detach().cpu().numpy()
        # generic_mesh = o3d.geometry.PointCloud()
        # generic_mesh.points = o3d.utility.Vector3dVector(xyz)
        # o3d.io.write_point_cloud(posed_gaussian_point_cloud_path, generic_mesh)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            # npz_path = Path(path).parent / "flame_param.npz"
            npz_path = path.replace('.ply', '_flame_param.npz')
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            checkpoint_num_timesteps = flame_param['expr'].shape[0]

            # if the number of timesteps in the checkpoint is different from the current one, use the current one
            # TODO: remove this part. Currently it exists because we want to use the checkpoint of static scene for dynamic scene training
            if checkpoint_num_timesteps != self.num_timesteps:
                flame_param['translation'] = self.flame_param['translation'].clone()
                flame_param['rotation'] = self.flame_param['rotation'].clone()
                flame_param['neck_pose'] = self.flame_param['neck_pose'].clone()
                flame_param['jaw_pose'] = self.flame_param['jaw_pose'].clone()
                flame_param['eyes_pose'] = self.flame_param['eyes_pose'].clone()
                flame_param['expr'] = self.flame_param['expr'].clone()
                flame_param['dynamic_offset'] = self.flame_param['dynamic_offset'].clone()


            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                # keep the static parameters
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                # update the dynamic parameters
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers

        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def generate(self, data):
        
        self.select_mesh_by_timestep(data['timestep'])

        B = data['exp_coeff'].shape[0]

        xyz = self.get_xyz.unsqueeze(0).repeat(B, 1, 1)

        color = torch.zeros([B, xyz.shape[1], 32], device=xyz.device)
        # TODO: If we decide that hair only has diffuse color, then the following direction staff is not needed
        for b in range(B):
            # view dependent/independent color
            shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            dir_pp = (self.get_xyz - data['camera_center'][b].repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            color[b,:,:3] = torch.clamp_min(sh2rgb + 0.5, 0.0) 

        scales = self.get_scaling.unsqueeze(0).repeat(B, 1, 1) 

        rotation = self.get_rotation.unsqueeze(0).repeat(B, 1, 1)

        opacity = self.get_opacity.unsqueeze(0).repeat(B, 1, 1) 


        delta_xyz = torch.zeros_like(xyz, device=xyz.device)
        delta_attributes = torch.zeros([B, xyz.shape[1], scales.shape[2] + rotation.shape[2] + opacity.shape[2]], device=xyz.device)

        # # TODO: save the weigts
        # if data['poses_history'] is not None:
        #     for b in range(B):
        #         # # color, [features + pose_embedding] -> [color]
        #         # feature_pose_controlled = feature[b]
        #         # pose_color_input = torch.cat([feature_pose_controlled.t(), 
        #         #                                 self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, feature_pose_controlled.shape[0])], 0)[None]
        #         # pose_color = self.pose_color_mlp(pose_color_input)[0].t()
        #         # color[b] += pose_color

        #         # # attributes: scales, rotation, opacity, [features + pose_embedding] -> [attributes]
        #         # pose_attributes_input = pose_color_input
        #         # pose_attributes = self.pose_attributes_mlp(pose_attributes_input)[0].t()
        #         # delta_attributes[b] += pose_attributes 

        #         # xyz deform, [xyz_embedding + pose_embedding] -> [xyz]
        #         xyz_pose_controlled = xyz[b]
        #         pose_deform_input = torch.cat([self.pos_embedding(xyz_pose_controlled).t(), 
        #                                     self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, xyz_pose_controlled.shape[0])], 0)[None]
        #         pose_deform = self.pose_deform_mlp(pose_deform_input)[0].t()
        #         delta_xyz[b] += pose_deform 

        xyz = xyz + delta_xyz

        # delta_scales = delta_attributes[:, :, 0:3]
        # scales = self.scales.unsqueeze(0).repeat(B, 1, 1) + delta_scales 
        # scales = torch.exp(scales)

        # delta_rotation = delta_attributes[:, :, 3:7]
        # rotation = self.rotation.unsqueeze(0).repeat(B, 1, 1) + delta_rotation 
        # rotation = torch.nn.functional.normalize(rotation, dim=2)

        # delta_opacity = delta_attributes[:, :, 7:8]
        # opacity = self.opacity.unsqueeze(0).repeat(B, 1, 1) + delta_opacity 
        # opacity = torch.sigmoid(opacity)



        # data['exp_deform'] = exp_deform
        color[...,3:6] = self.get_seg_label.unsqueeze(0).repeat(B, 1, 1)


        data['xyz'] = xyz
        data['color'] = color
        data['scales'] = scales
        data['rotation'] = rotation
        data['opacity'] = opacity
        return data


    @property
    def get_seg_label(self):
        # seg_label_unstruct = torch.cat([self.seg_label_activation(self.seg_label[..., :2]), torch.zeros_like(self.get_opacity)], dim = -1)
        seg_label_unstruct = torch.zeros_like(self._xyz) 
        seg_label_unstruct[:, 1] = 1
        return seg_label_unstruct
