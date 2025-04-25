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
                                          landmarks_3d_neutral=meshhead.landmarks_3d_neutral.detach().cpu(),
                                          add_mouth_points=True).to(device)
        # release memory
        meshhead = meshhead.cpu()
        del meshhead
        torch.cuda.empty_cache()
    
    # create hair gaussian, 
    gaussianhair = GaussianHairModule(cfg.gaussianhairmodule).to(device)
    gaussianhair.update_mesh_alignment_transform(dataset.R, dataset.T, dataset.S, flame_mesh_path = dataset.flame_mesh_path)

    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    if os.path.exists(cfg.load_supres_checkpoint):
        supres.load_state_dict(torch.load(cfg.load_supres_checkpoint, map_location=lambda storage, loc: storage))

    camera = CameraModule()
    recorder = None

    torch.autograd.set_detect_anomaly(True)
    start_epoch = cfg.start_epoch


    start_epoch = 312
    # gaussianhead_checkpoint =  f'%s/%s/gaussianhead_latest' % (recorder.checkpoint_path, recorder.name)
    # gaussianhair_checkpoint =  f'%s/%s/gaussianhair_latest' % (recorder.checkpoint_path, recorder.name)
    gaussianhead_checkpoint =  f'%s/%s/gaussianhead_epoch_%d' % (recorder.checkpoint_path, recorder.name, start_epoch)
    gaussianhair_checkpoint =  f'%s/%s/gaussianhair_epoch_%d' % (recorder.checkpoint_path, recorder.name, start_epoch)
    gaussianhead.load_state_dict(torch.load(gaussianhead_checkpoint, map_location=lambda storage, loc: storage))
    gaussianhair.load_state_dict(torch.load(gaussianhair_checkpoint, map_location=lambda storage, loc: storage))
    # start_epoch = int(gaussianhead_checkpoint.split('/')[-1].split('_')[0])
    start_epoch += 1


    if os.path.exists(cfg.load_delta_poses_checkpoint):
        delta_poses = torch.load(cfg.load_delta_poses_checkpoint)
    else:
        delta_poses = torch.zeros([dataset.num_exp_id, 6]).to(device)

    delta_poses = delta_poses.requires_grad_(False)

    app = Reenactment_hair(dataloader, gaussianhead, gaussianhair,supres, camera, recorder, cfg.gpu_id, False)
    app.run()
