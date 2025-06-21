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
from lib.module.flame_gaussian_model import FlameGaussianModel
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import GaussianHeadTrainRecorder
from config.config import config_reenactment

from lib.dataset.Dataset import ReenactmentDataset
from lib.recorder.Recorder import ReenactmentRecorder
from lib.apps.Reenactment import Reenactment
from lib.apps.Reenactment_hair import Reenactment_hair


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_s2_N031.yaml')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--test_camera_id', type=int, default=23)
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
    # dataset = GaussianDataset(cfg.dataset, split_strategy='all')
    dataset = GaussianDataset(cfg.dataset, split_strategy='test')
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
        select_indices = range(data['verts'].shape[0])
        gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=data['verts'][select_indices].cpu(),
                                          feature=torch.atanh(data['verts_feature'][select_indices].cpu()), 
                                        #   landmarks_3d_neutral=meshhead.landmarks_3d_neutral.detach().cpu(),
                                          landmarks_3d_neutral=dataset.init_landmarks_3d_neutral,
                                          add_mouth_points=True).to(device)
        # release memory
        meshhead = meshhead.cpu()
        del meshhead
        torch.cuda.empty_cache()
    
    gaussians = FlameGaussianModel(0, disable_flame_static_offset = True, n_shape= dataset.shape_dims, n_expr=dataset.exp_dims)
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


    start_epoch = 390
    
    random_seed = '174731702465'

    if cfg.checkpoint_seed != -1:
        random_seed = str(cfg.checkpoint_seed)

    gaussianhead_checkpoint =  f'%s/%s/gaussianhead_latest_%s' % (recorder.checkpoint_path, recorder.name, random_seed)

    gaussianhair_checkpoint =  f'%s/%s/gaussianhair_latest_%s' % (recorder.checkpoint_path, recorder.name, random_seed)

    gaussians_ply_checkpoint =  f'%s/%s/head_latest_%s.ply' % (recorder.checkpoint_path, recorder.name, random_seed)
    

    # first check if the direct load checkpoint path exists
    if cfg.gaussianheadmodule.load_gaussianhead_checkpoint != '':
        gaussianhead_checkpoint = cfg.gaussianheadmodule.load_gaussianhead_checkpoint
    if cfg.gaussianhairmodule.load_gaussianhair_checkpoint != '':
        gaussianhair_checkpoint = cfg.gaussianhairmodule.load_gaussianhair_checkpoint
    if cfg.flame_gaussian_module.load_flame_gaussian_checkpoint != '':
        gaussians_ply_checkpoint = cfg.flame_gaussian_module.load_flame_gaussian_checkpoint

    
    if not cfg.flame_gaussian_module.enable and os.path.exists(gaussianhead_checkpoint):
        gaussianhead.load_state_dict(torch.load(gaussianhead_checkpoint, map_location=lambda storage, loc: storage))
        print('load gaussianhead checkpoint from %s' % gaussianhead_checkpoint)
    if os.path.exists(gaussianhair_checkpoint):
        gaussianhair.load_state_dict(torch.load(gaussianhair_checkpoint, map_location=lambda storage, loc: storage), strict=False)
        print('load gaussianhair checkpoint from %s' % gaussianhair_checkpoint)

    if cfg.flame_gaussian_module.enable and os.path.exists(gaussians_ply_checkpoint):
        gaussians.load_ply(gaussians_ply_checkpoint, has_target= False)
        print('load gaussians checkpoint from %s' % gaussians_ply_checkpoint)
    
    # start_epoch = int(gaussianhead_checkpoint.split('/')[-1].split('_')[0])
    start_epoch += 1

    gaussianhair.update_mesh_alignment_transform(dataset.R, dataset.T, dataset.S, flame_mesh_path = dataset.flame_mesh_path)

    if os.path.exists(cfg.load_delta_poses_checkpoint):
        delta_poses = torch.load(cfg.load_delta_poses_checkpoint)
    else:
        delta_poses = torch.zeros([dataset.num_exp_id, 6]).to(device)

    delta_poses = delta_poses.requires_grad_(False)

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

    app = Reenactment_hair(dataloader, gaussianhead, gaussianhair,supres, camera, recorder, cfg.gpu_id, freeview=False, camera_id=arg.test_camera_id)
    app.run()
