import os
import torch
from torch.utils.data import ConcatDataset
import argparse

from config.config import config_train

from lib.dataset.Dataset import MeshDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.MeshHeadModule import MeshHeadModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import MeshHeadTrainRecorder
from lib.trainer.MeshHeadTrainer import MeshHeadTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_s1_N031.yaml')
    parser.add_argument('--dataroot', type=str, nargs='+', default=[])
    arg = parser.parse_args()

    cfg = config_train()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()


    if len(cfg.dataset.dataroot) > 0:
        datasets = []
        for dataroot in arg.dataroot:
            arg_cfg = ['dataroot', dataroot]
            cfg.dataset.merge_from_list(arg_cfg)
            single_dataset = MeshDataset(cfg.dataset)
            datasets.append(single_dataset)
        dataset = ConcatDataset(datasets)
    else:
        dataset = MeshDataset(cfg.dataset)
        
    dataloader = DataLoaderX(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True) 

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    meshhead = MeshHeadModule(cfg.meshheadmodule, dataset.init_landmarks_3d_neutral).to(device)
    if os.path.exists(cfg.load_meshhead_checkpoint):
        meshhead.load_state_dict(torch.load(cfg.load_meshhead_checkpoint, map_location=lambda storage, loc: storage))
    else:
        meshhead.pre_train_sphere(300, device)
    
    camera = CameraModule()
    recorder = MeshHeadTrainRecorder(cfg.recorder)

    optimizer = torch.optim.Adam([{'params' : meshhead.landmarks_3d_neutral, 'lr' : cfg.lr_lmk},
                                  {'params' : meshhead.geo_mlp.parameters(), 'lr' : cfg.lr_net},
                                  {'params' : meshhead.exp_color_mlp.parameters(), 'lr' : cfg.lr_net},
                                  {'params' : meshhead.pose_color_mlp.parameters(), 'lr' : cfg.lr_net},
                                  {'params' : meshhead.exp_deform_mlp.parameters(), 'lr' : cfg.lr_net},
                                  {'params' : meshhead.pose_deform_mlp.parameters(), 'lr' : cfg.lr_net}])
    trainer = MeshHeadTrainer(dataloader, meshhead, camera, optimizer, recorder, cfg.gpu_id)
    trainer.train(0, 50)

