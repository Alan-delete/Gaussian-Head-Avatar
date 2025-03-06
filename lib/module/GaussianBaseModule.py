import os
import numpy as np
import torch
from torch import nn
import tqdm
from plyfile import PlyData, PlyElement

from lib.utils.general_utils import inverse_sigmoid

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 1, 0] = 2 * (x*y - r*z)
    R[:, 2, 0] = 2 * (x*z + r*y)
    R[:, 0, 1] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 2, 1] = 2 * (y*z - r*x)
    R[:, 0, 2] = 2 * (x*z - r*y)
    R[:, 1, 2] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

class GaussianBaseModule(nn.Module):
    def __init__(self, optimizer = None):
        super(GaussianBaseModule, self).__init__()
        self.optimizer = optimizer
        self.GS_parameter_names = ['xyz', 'feature', 'features_dc', 'features_rest', 'scales', 'rotation', 'opacity']
        self.xyz = torch.empty(0)

        # mlp
        self.feature = torch.empty(0)
        # sh
        self.features_dc = torch.empty(0)
        self.features_rest = torch.empty(0)
        
        self.scales = torch.empty(0)
        self.rotation = torch.empty(0)
        self.opacity = torch.empty(0)

        self.label_hair = torch.empty(0)
        self.label_body = torch.empty(0)
        self.seg_label = torch.empty(0)
        
        self.scales_activation = torch.exp
        self.scales_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.label_activation = torch.sigmoid
        self.inverse_label_activation = inverse_sigmoid

        self.seg_label_activation = torch.nn.Softmax(dim=-1)

        self.rotation_activation = torch.nn.functional.normalize

        self.orient_conf_activation = torch.exp
        self.orient_conf_inverse_activation = torch.log

        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.percent_dense = 0.01    

    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_scales(self):
        return self.scales_activation(self.scales)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)
    
    @property
    def get_hair_label(self):
        return self.label_activation(self.label_hair)
    
    @property
    def get_body_label(self):
        return self.label_activation(self.label_body)
    
    @property
    def get_seg_label(self):
        return self.seg_label_activation(self.seg_label)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

    @property
    def get_features(self):
        features_dc = self.features_dc
        features_rest = self.features_rest
        return torch.cat((features_dc, features_rest), dim=1)    


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

        # Get unstructured Gaussian directions
        scaling_unstruct = self.get_scales
        i = torch.arange(self.scales.shape[0], device='cuda')[:, None].repeat(1, 3).view(-1)
        j = scaling_unstruct.argsort(dim=-1, descending=True).view(-1)
        sorted_R_unstruct = build_rotation(self.rotation)[i, j].view(-1, 3, 3)
        sorted_S_unstruct = scaling_unstruct[i, j].view(-1, 3)
        self._dir = sorted_R_unstruct[:, 0] * sorted_S_unstruct[:, 0, None]

        #dir3D = F.normalize(self._dir, dim=-1)
        dir2D = (self._dir[:, None, :] @ T)[:, 0]
        dir2D = torch.nn.functional.normalize(dir2D, dim=-1)
        # make only strctured gassiuan has direction
        # dir2D[:self._dir_unstruct.shape[0]] = torch.zeros_like(dir2D[:self._dir_unstruct.shape[0]])
        return dir2D
    

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def save_ply_gaussian_group(self, path, tar_seg = 2):
        dir = os.path.dirname(path)
        name = os.path.basename(path)
        os.mkdir(dir, exist_ok=True)


        # Re-initialize the buffers
        # self.get_features
        self.get_opacity
        self.get_scales
        self.get_rotation

        # get the mask of corresponding label
        selected_mask = self.get_seg_label[:, tar_seg] > 0.5
        selected_mask = selected_mask.detach().cpu().numpy()

        xyz = self.xyz.detach().cpu().numpy()[selected_mask]
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[selected_mask]
        f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[selected_mask]
        opacities = self.opacity.detach().cpu().numpy()[selected_mask]
        scale = self.scales.detach().cpu().numpy()[selected_mask]
        rotation = self.rotation.detach().cpu().numpy()[selected_mask]


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)      

    def save_ply(self, path):
        dir = os.path.dirname(path)
        name = os.path.basename(path)
        os.mkdir(dir, exist_ok=True)

        # Re-initialize the buffers
        # self.get_feature
        self.get_opacity
        self.get_scales
        self.get_rotation

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scales.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(f'{dir}/{name}')

    @staticmethod
    def replace_tensor_to_optimizer(optimizer, tensor, name):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if "name" not in group:
                continue
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state != None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state != None:
                    optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_opacity(self):
        opacities_new = torch.min(self.opacity, self.inverse_opacity_activation(torch.ones_like(self.opacity))*0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(self.optimizer, opacities_new, "opacity")
        self.opacity = optimizable_tensors["opacity"]
  

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            
            if "name" not in group or group["name"] not in self.GS_parameter_names:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        for GS_parameter in self.GS_parameter_names:
            new_tensor = optimizable_tensors[GS_parameter]
            setattr(self, GS_parameter, new_tensor)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:

            if "name" not in group or group["name"] not in self.GS_parameter_names:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, densification_dict):

        optimizable_tensors = self.cat_tensors_to_optimizer(densification_dict)

        for GS_parameter in self.GS_parameter_names:
            new_tensor = optimizable_tensors[GS_parameter]
            setattr(self, GS_parameter, new_tensor)

        self.xyz_gradient_accum = torch.zeros_like(self.opacity, device=self.xyz.device)
        self.denom = torch.zeros_like(self.opacity, device=self.xyz.device)

        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device=self.xyz.device)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device= self.xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scales, dim=1).values > self.percent_dense*scene_extent)
        d = {}
        stds = self.get_scales[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device= self.xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.rotation[selected_pts_mask]).repeat(N,1,1)

        d['xyz'] = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected_pts_mask].repeat(N, 1)
        d['scales'] = self.scales_inverse_activation(self.get_scales[selected_pts_mask].repeat(N,1) / (0.8*N))

        for GS_parameter in self.GS_parameter_names:
            if GS_parameter not in ['xyz', 'scales']:
                new_tensor = getattr(self, GS_parameter)[selected_pts_mask].repeat_interleave(N,dim=0)
                d[GS_parameter] = new_tensor

        self.densification_postfix(densification_dict = d)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.xyz.device , dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scales, dim=1).values <= self.percent_dense*scene_extent)
        
        d = {}
        for GS_parameter in self.GS_parameter_names:
            new_tensor = getattr(self, GS_parameter)[selected_pts_mask]
            d[GS_parameter] = new_tensor

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scales, new_rotation)
        self.densification_postfix(densification_dict = d)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = self.get_opacity.squeeze() < min_opacity
        # prune_mask = torch.logical_or(prune_mask, self.get_hair_label.squeeze() > 0.5)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scales.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
 

