import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import argparse

from config.config import config_train

from lib.dataset.Dataset import GaussianDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.MeshHeadModule import MeshHeadModule
from lib.module.GaussianHeadModule import GaussianHeadModule
from lib.module.GaussianHairModule import GaussianHairModule
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.module.flame_gaussian_model import FlameGaussianModel
from lib.recorder.Recorder import GaussianHeadTrainRecorder
from lib.trainer.GaussianHeadTrainer import GaussianHeadTrainer
from lib.trainer.GaussianHeadHairTrainer import GaussianHeadHairTrainer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_s2_N031.yaml')
    parser.add_argument('--dataroot', type=str, default='')
    arg = parser.parse_args()

    cfg = config_train()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    # cfg_name = os.path.basename(arg.config).split('.')[0]
    # cfg.recorder.name = cfg_name
    if arg.dataroot != '':
        arg_cfg = ['dataroot', arg.dataroot]
        cfg.dataset.merge_from_list(arg_cfg)
    
    # debug select frames is to only load a few frames for debugging
    dataset = GaussianDataset(cfg.dataset)
    dataloader = DataLoaderX(dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True) 

    # test_dataset = GaussianDataset(cfg.dataset, train=False) 
    # test_dataloader = DataLoaderX(test_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True)

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)
    
    if os.path.exists(cfg.load_gaussianhead_checkpoint):
        gaussianhead_state_dict = torch.load(cfg.load_gaussianhead_checkpoint, map_location=lambda storage, loc: storage)
        gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=gaussianhead_state_dict['xyz'], 
                                          feature=gaussianhead_state_dict['feature'],
                                          landmarks_3d_neutral=gaussianhead_state_dict['landmarks_3d_neutral']).to(device)
        gaussianhead.load_state_dict(gaussianhead_state_dict)
    else:
        meshhead_state_dict = torch.load(cfg.load_meshhead_checkpoint, map_location=lambda storage, loc: storage)
        meshhead = MeshHeadModule(cfg.meshheadmodule, meshhead_state_dict['landmarks_3d_neutral']).to(device)
        meshhead.load_state_dict(meshhead_state_dict)
        meshhead.subdivide()
        with torch.no_grad():
            data = meshhead.reconstruct_neutral()

        # contraint the number of vertices to 100000, otherwise out of memory for 24GB
        select_indices = torch.randperm(data['verts'].shape[0])[:50000]
        gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=data['verts'][select_indices].cpu(),
                                          feature=torch.atanh(data['verts_feature'][select_indices].cpu()), 
                                        #   landmarks_3d_neutral=meshhead.landmarks_3d_neutral.detach().cpu(),
                                          landmarks_3d_neutral=dataset.init_landmarks_3d_neutral,
                                          add_mouth_points=True).to(device)
        # gaussianhead.exp_color_mlp.load_state_dict(meshhead.exp_color_mlp.state_dict())
        # gaussianhead.pose_color_mlp.load_state_dict(meshhead.pose_color_mlp.state_dict())
        # gaussianhead.exp_deform_mlp.load_state_dict(meshhead.exp_deform_mlp.state_dict())
        # gaussianhead.pose_deform_mlp.load_state_dict(meshhead.pose_deform_mlp.state_dict())
        
        # release memory
        meshhead = meshhead.cpu()
        del meshhead
        torch.cuda.empty_cache()
    
    gaussians = FlameGaussianModel(0, disable_flame_static_offset = cfg.gaussianhairmodule.enable , n_shape= dataset.shape_dims, n_expr=dataset.exp_dims)
    # process meshes
    T = len(dataset.samples)

    if gaussians.binding != None:
        gaussians.load_meshes(train_meshes=dataset.train_meshes,
                              test_meshes={},
                              tgt_train_meshes = {},
                              tgt_test_meshes = {})
        cameras_extent = 4.907987451553345
        gaussians.create_from_pcd(None, cameras_extent)

        gaussians.training_setup(cfg.flame_gaussian_module)
        

    # create hair gaussian, 
    gaussianhair = GaussianHairModule(cfg.gaussianhairmodule).to(device)
    gaussianhair.update_mesh_alignment_transform(dataset.R, dataset.T, dataset.S, flame_mesh_path = dataset.flame_mesh_path)

    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    if os.path.exists(cfg.load_supres_checkpoint):
        supres.load_state_dict(torch.load(cfg.load_supres_checkpoint, map_location=lambda storage, loc: storage))

    camera = CameraModule()
    recorder = GaussianHeadTrainRecorder(cfg)

    torch.autograd.set_detect_anomaly(True)
    start_epoch = cfg.start_epoch

    # load_state_dict() does not interfere with autograd, it just overwrites the .data of parameters.
    if cfg.resume_training:
        start_epoch = 351
        # gaussianhead_checkpoint =  f'%s/%s/gaussianhead_latest' % (recorder.checkpoint_path, recorder.name)
        # gaussianhair_checkpoint =  f'%s/%s/gaussianhair_latest' % (recorder.checkpoint_path, recorder.name)
        gaussianhead_checkpoint =  f'%s/%s/gaussianhead_epoch_%d' % (recorder.checkpoint_path, recorder.name, start_epoch)
        gaussianhair_checkpoint =  f'%s/%s/gaussianhair_epoch_%d' % (recorder.checkpoint_path, recorder.name, start_epoch)
        gaussians_ply_checkpoint =  f'%s/%s/head_latest.ply' % (recorder.checkpoint_path, recorder.name)
        gaussians_ply_checkpoint = '/local/home/haonchen/Gaussian-Head-Avatar/checkpoints/flame_gaussian_renderme/025000_head.ply'
        
        if os.path.exists(gaussians_ply_checkpoint):
            gaussians.load_ply(gaussians_ply_checkpoint, has_target= False)
            gaussians.training_setup(cfg.flame_gaussian_module)
        if os.path.exists(gaussianhair_checkpoint):
            gaussianhair.load_state_dict(torch.load(gaussianhair_checkpoint, map_location=lambda storage, loc: storage))
        # if os.path.exists(gaussianhead_checkpoint):
        #     gaussianhead.load_state_dict(torch.load(gaussianhead_checkpoint, map_location=lambda storage, loc: storage))
        
        # start_epoch = int(gaussianhead_checkpoint.split('/')[-1].split('_')[0])
        start_epoch += 1

    else:
        # only reset points_raw if not resume training, otherwise gaussianhair.transform will be backpropagated wrongly
        gaussianhair.reset_strands()

    # gaussians.select_mesh_by_timestep(0)
    # breakpoint()

    
    # TODO: move the gaussianhead optimizer into the gaussianhead module
    optimized_parameters = [{'params' : supres.parameters(), 'lr' : cfg.lr_net, 'name' : 'supres'},]

    gaussianhead_optimized_parameters = [{'params' : gaussianhead.xyz, 'lr' : cfg.lr_net * 0.1, 'name' : 'xyz'},
                            {'params' : gaussianhead.feature, 'lr' : cfg.lr_net * 0.1, 'name' : 'feature'},
                            {'params' : gaussianhead.exp_color_mlp.parameters(), 'lr' : cfg.lr_net, 'name' : 'exp_color_mlp'},
                            {'params' : gaussianhead.pose_color_mlp.parameters(), 'lr' : cfg.lr_net, 'name' : 'pose_color_mlp'},
                            {'params' : gaussianhead.exp_deform_mlp.parameters(), 'lr' : cfg.lr_net, 'name' : 'exp_deform_mlp'},
                            {'params' : gaussianhead.pose_deform_mlp.parameters(), 'lr' : cfg.lr_net, 'name' : 'pose_deform_mlp'},
                            {'params' : gaussianhead.exp_attributes_mlp.parameters(), 'lr' : cfg.lr_net, 'name' : 'exp_attributes_mlp'},
                            {'params' : gaussianhead.pose_attributes_mlp.parameters(), 'lr' : cfg.lr_net, 'name' : 'pose_attributes_mlp'},
                            {'params' : gaussianhead.scales, 'lr' : cfg.lr_net * 0.3, 'name' : 'scales'},
                            {'params' : gaussianhead.rotation, 'lr' : cfg.lr_net * 0.1, 'name' : 'rotation'},
                            {'params' : gaussianhead.opacity, 'lr' : cfg.lr_net, 'name' : 'opacity'},
                            # {'params' : gaussianhead.scales, 'lr' : cfg.lr_net * 3, 'name' : 'scales'},
                            # {'params' : gaussianhead.rotation, 'lr' : cfg.lr_net * 0.5, 'name' : 'rotation'},
                            # {'params' : gaussianhead.opacity, 'lr' : cfg.lr_net * 10, 'name' : 'opacity'},
                            {'params' : gaussianhead.seg_label, 'lr' : cfg.lr_net , 'name' : 'seg_label'},
                            {'params' : gaussianhead.features_dc, 'lr' : cfg.lr_net * 50, 'name' : 'features_dc'},
                            {'params' : gaussianhead.features_rest, 'lr' : cfg.lr_net * 50, 'name' : 'features_rest'},
                            ]


    gaussianhead.optimizer = torch.optim.Adam(gaussianhead_optimized_parameters)
    # gaussianhead.scheduler_args = get_expon_lr_func(lr_init=cfg.position_lr_init*0.1*self.spatial_lr_scale,
    #                                                 lr_final=cfg.position_lr_final*self.spatial_lr_scale,
    #                                                 lr_delay_mult=cfg.position_lr_delay_mult,
    #                                                 max_steps=cfg.position_lr_max_steps)

    if os.path.exists(cfg.load_delta_poses_checkpoint):
        delta_poses = torch.load(cfg.load_delta_poses_checkpoint)
    else:
        delta_poses = torch.zeros([dataset.num_exp_id, 6]).to(device)

    if cfg.optimize_pose:
        delta_poses = delta_poses.requires_grad_(True)
        optimized_parameters.append({'params' : delta_poses, 'lr' : cfg.lr_pose})
    else:
        delta_poses = delta_poses.requires_grad_(False)

    optimizer = torch.optim.Adam(optimized_parameters)


    if cfg.flame_gaussian_module.enable:
        gaussianhead = gaussians
    else:
        gaussianhead = gaussianhead 
    

    if cfg.gaussianhairmodule.enable:
        gaussianhair = gaussianhair
    else:
        gaussianhair = None
        arg_cfg = ['train_segment', False]
        cfg.merge_from_list(arg_cfg)


    trainer = GaussianHeadHairTrainer(dataloader, delta_poses, gaussianhead, gaussianhair,supres, camera, optimizer, recorder, cfg.gpu_id, cfg)
    trainer.train(start_epoch, start_epoch + cfg.num_epochs)
