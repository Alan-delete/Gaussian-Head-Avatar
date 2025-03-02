import torch
from torch import nn
from einops import rearrange
import tqdm
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from simple_knn._C import distCUDA2

from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder
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
    # def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scales, new_rotation):
    #     d = {"xyz": new_xyz,
    #          "f_dc": new_features_dc,
    #          "f_rest": new_features_rest,
    #          "opacity": new_opacities,
    #         #  "label_hair": new_labels_hair,
    #         #  "label_body": new_labels_body,
    #         #  "seg_label": new_seg_label,
    #          "scales" : new_scales,
    #          "rotation" : new_rotation}

    #     optimizable_tensors = self.cat_tensors_to_optimizer(d)

    #     self.xyz = optimizable_tensors["xyz"]
    #     self.features_dc = optimizable_tensors["f_dc"]
    #     self.features_rest = optimizable_tensors["f_rest"]
    #     self.opacity = optimizable_tensors["opacity"]
    #     # self._label_hair_unstruct = optimizable_tensors["label_hair"]
    #     # self._label_body_unstruct = optimizable_tensors["label_body"]
    #     # self._seg_label_unstruct = optimizable_tensors["seg_label"]
    #     self.scales = optimizable_tensors["scales"]
    #     self.rotation = optimizable_tensors["rotation"]

    #     self.xyz_gradient_accum = torch.zeros_like(self.opacity, device=self.xyz.device)
    #     self.denom = torch.zeros_like(self.opacity, device=self.xyz.device)

    #     self.max_radii2D = torch.zeros((self.xyz.shape[0]), device=self.xyz.device)

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

        # d['rotation'] = self.rotation[selected_pts_mask].repeat(N,1)
        # d['feature'] = self.feature[selected_pts_mask].repeat(N,1,1)
        # d['features_dc'] = self.features_dc[selected_pts_mask].repeat(N,1,1)
        # d['features_rest'] = self.features_rest[selected_pts_mask].repeat(N,1,1)
        # d['opacity'] = self.opacity[selected_pts_mask].repeat(N,1)
        # new_label_hair = self._label_hair_unstruct[selected_pts_mask].repeat(N,1)
        # new_label_body = self._label_body_unstruct[selected_pts_mask].repeat(N,1)
        # new_seg_label = self._seg_label_unstruct[selected_pts_mask].repeat(N,1)

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity,  new_scales, new_rotation)
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
 


class GaussianHeadModule(GaussianBaseModule):
    def __init__(self, cfg, xyz, feature, landmarks_3d_neutral, add_mouth_points=False, optimizer=None, GS_parameter_names = ["xyz", "feature", "scales", "rotation", "opacity"]):
        super(GaussianHeadModule, self).__init__()

        if add_mouth_points and cfg.num_add_mouth_points > 0:
            mouth_keypoints = landmarks_3d_neutral[48:66]
            mouth_center = torch.mean(mouth_keypoints, dim=0, keepdim=True)
            mouth_center[:, 2] = mouth_keypoints[:, 2].min()
            max_dist = (mouth_keypoints - mouth_center).abs().max(0)[0]
            points_add = (torch.rand([cfg.num_add_mouth_points, 3]) - 0.5) * 1.6 * max_dist + mouth_center
        
            xyz = torch.cat([xyz, points_add])
            feature = torch.cat([feature, torch.zeros([cfg.num_add_mouth_points, feature.shape[1]])])

        # record the GS related parameters, involved in the densification and pruning
        self.GS_parameter_names = GS_parameter_names

        self.xyz = nn.Parameter(xyz)
        self.feature = nn.Parameter(feature)
        self.register_buffer('landmarks_3d_neutral', landmarks_3d_neutral)

        dist2 = torch.clamp_min(distCUDA2(self.xyz.cuda()), 0.0000001).cpu()
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self.scales = nn.Parameter(scales)

        rots = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        rots[:, 0] = 1
        self.rotation = nn.Parameter(rots)

        self.opacity = nn.Parameter(inverse_sigmoid(0.3 * torch.ones((xyz.shape[0], 1))))

        self.xyz_gradient_accum = torch.zeros_like(self.opacity, device="cuda")
        self.denom = torch.zeros_like(self.opacity, device="cuda")
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")


        self.exp_color_mlp = MLP(cfg.exp_color_mlp, last_op=None)
        self.pose_color_mlp = MLP(cfg.pose_color_mlp, last_op=None)
        self.exp_attributes_mlp = MLP(cfg.exp_attributes_mlp, last_op=None)
        self.pose_attributes_mlp = MLP(cfg.pose_attributes_mlp, last_op=None)
        self.exp_deform_mlp = MLP(cfg.exp_deform_mlp, last_op=nn.Tanh())
        self.pose_deform_mlp = MLP(cfg.pose_deform_mlp, last_op=nn.Tanh())

        self.pos_embedding, _ = get_embedder(cfg.pos_freq)
        
        self.exp_coeffs_dim = cfg.exp_coeffs_dim
        self.dist_threshold_near = cfg.dist_threshold_near
        self.dist_threshold_far = cfg.dist_threshold_far
        self.deform_scale = cfg.deform_scale
        self.attributes_scale = cfg.attributes_scale
    
    def generate(self, data):
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


    # def _prune_optimizer(self, mask):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         stored_state = self.optimizer.state.get(group['params'][0], None)

    #         if "name" not in group or group["name"] not in self.GS_parameter_names:
    #             continue

    #         if stored_state is not None:
    #             stored_state["exp_avg"] = stored_state["exp_avg"][mask]
    #             stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors

    # def prune_points(self, mask):
    #     valid_points_mask = ~mask
    #     optimizable_tensors = self._prune_optimizer(valid_points_mask)

    #     self.xyz = optimizable_tensors["xyz"]
    #     self.feature = optimizable_tensors["feature"]
    #     self.opacity = optimizable_tensors["opacity"]
    #     self.scales = optimizable_tensors["scales"]
    #     self.rotation = optimizable_tensors["rotation"]

    #     self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
    #     self.denom = self.denom[valid_points_mask]
    #     self.max_radii2D = self.max_radii2D[valid_points_mask]

    # def cat_tensors_to_optimizer(self, tensors_dict):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
            
    #         assert len(group["params"]) == 1

    #         # the parameter group is not in the dict, meaning it is not the parameters of the Gaussian Splatting model
    #         if ("name" not in group) or (group["name"] not in self.GS_parameter_names):
    #             continue

    #         extension_tensor = tensors_dict[group["name"]]
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:

    #             stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
    #             stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]

    #     return optimizable_tensors

    # def densification_postfix(self, new_xyz, new_features, new_opacities, new_scales, new_rotation):
    #     d = {"xyz": new_xyz,
    #          "feature": new_features,    
    #          "opacity": new_opacities,
    #          "scales" : new_scales,
    #          "rotation" : new_rotation}

    #     optimizable_tensors = self.cat_tensors_to_optimizer(d)
    #     self.xyz = optimizable_tensors["xyz"]
    #     self.feature = optimizable_tensors["feature"]
    #     self.opacity = optimizable_tensors["opacity"]
    #     self.scales = optimizable_tensors["scales"]
    #     self.rotation = optimizable_tensors["rotation"]

    #     self.xyz_gradient_accum = torch.zeros_like(self.opacity, device=self.xyz.device)
    #     self.denom = torch.zeros_like(self.opacity, device=self.xyz.device)

    #     self.max_radii2D = torch.zeros((self.xyz.shape[0]), device=self.xyz.device)

    # def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    #     n_init_points = self.xyz.shape[0]
    #     # Extract points that satisfy the gradient condition
    #     padded_grad = torch.zeros((n_init_points), device= self.xyz.device)
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
    #     selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(self.get_scales, dim=1).values > self.percent_dense*scene_extent)

    #     stds = self.get_scales[selected_pts_mask].repeat(N,1)
    #     means =torch.zeros((stds.size(0), 3),device= self.xyz.device)
    #     samples = torch.normal(mean=means, std=stds)
    #     rots = build_rotation(self.rotation[selected_pts_mask]).repeat(N,1,1)
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[selected_pts_mask].repeat(N, 1)
    #     new_scales = self.scales_inverse_activation(self.get_scales[selected_pts_mask].repeat(N,1) / (0.8*N))
    #     new_rotation = self.rotation[selected_pts_mask].repeat(N,1)
    #     new_features = self.feature[selected_pts_mask].repeat(N,1,1)
    #     new_opacity = self.opacity[selected_pts_mask].repeat(N,1)

    #     self.densification_postfix(new_xyz, new_features, new_opacity,  new_scales, new_rotation)

    #     prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.xyz.device , dtype=bool)))
    #     self.prune_points(prune_filter)

    # def densify_and_clone(self, grads, grad_threshold, scene_extent):
    #     # Extract points that satisfy the gradient condition
    #     selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(self.get_scales, dim=1).values <= self.percent_dense*scene_extent)
        
    #     new_xyz = self.xyz[selected_pts_mask]
    #     new_features = self.feature[selected_pts_mask]
    #     new_opacities = self.opacity[selected_pts_mask]
    #     new_scales = self.scales[selected_pts_mask]
    #     new_rotation = self.rotation[selected_pts_mask]

    #     self.densification_postfix(new_xyz, new_features, new_opacities, new_scales, new_rotation)

    # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0

    #     self.densify_and_clone(grads, max_grad, extent)
    #     self.densify_and_split(grads, max_grad, extent)

    #     prune_mask = self.get_opacity.squeeze() < min_opacity
    #     prune_mask = torch.logical_or(prune_mask, self.get_hair_label.squeeze() > 0.5)
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         big_points_ws = self.get_scales.max(dim=1).values > 0.1 * extent
    #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)

    #     torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # batched
        if len(viewspace_point_tensor.shape) > 2: 
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[0, update_filter,:2], dim=-1, keepdim=True)
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

import os
import sys
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
        return 
        # Get unstructured Gaussian directions
        scaling_unstruct = self.get_scaling[:self._scaling_unstruct.shape[0]]
        i = torch.arange(self._scaling_unstruct.shape[0], device='cuda')[:, None].repeat(1, 3).view(-1)
        j = scaling_unstruct.argsort(dim=-1, descending=True).view(-1)
        sorted_R_unstruct = build_rotation(self._rotation_unstruct)[i, j].view(-1, 3, 3)
        sorted_S_unstruct = scaling_unstruct[i, j].view(-1, 3)
        self._dir_unstruct = sorted_R_unstruct[:, 0] * sorted_S_unstruct[:, 0, None]

        self._dir = torch.cat([self._dir_unstruct, self._dir_struct])

        #dir3D = F.normalize(self._dir, dim=-1)
        dir2D = (self._dir[:, None, :] @ T)[:, 0]
        dir2D = torch.nn.functional.normalize(dir2D, dim=-1)
        # make only strctured gassiuan has direction
        # dir2D[:self._dir_unstruct.shape[0]] = torch.zeros_like(dir2D[:self._dir_unstruct.shape[0]])
        return dir2D

# the Gaussian model that combines Gaussian Head Module(unstruct) and Gaussian Hair Module(struct)
class GaussianModule(GaussianBaseModule):
    def __init__(self, cfg, gaussianhead, optimizer_unstruct, gaussianhair = None, optimizer_struct = None):
        super(GaussianModule, self).__init__()
        self.cfg = cfg

        self.gaussian_head = gaussianhead
        self.optimizer_unstruct = optimizer_unstruct

        self.gaussian_hair = gaussianhair
        self.optimizer_struct = optimizer_struct
        
        pass

