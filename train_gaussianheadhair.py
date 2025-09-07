import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import argparse

from config.config import config_train

from lib.dataset.Dataset import GaussianDataset
from lib.dataset.Dataset import MultiDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.MeshHeadModule import MeshHeadModule
from lib.module.GaussianBaseModule import GaussianBaseModule
from lib.module.GaussianHeadModule import GaussianHeadModule
from lib.module.GaussianHairModule import GaussianHairModule
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.module.flame_gaussian_model import FlameGaussianModel
from lib.recorder.Recorder import GaussianHeadTrainRecorder
from lib.trainer.GaussianHeadHairTrainer import GaussianHeadHairTrainer
from lib.apps.Reenactment_hair import Reenactment_hair


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def build_dataloader(cfg, dataroots, split_strategy="train"):
    """Create dataloader from single or multiple datasets."""
    if len(dataroots) > 0:
        datasets = []
        for root in dataroots:
            arg_cfg = ["dataroot", root]
            cfg.dataset.merge_from_list(arg_cfg)
            datasets.append(GaussianDataset(cfg.dataset, split_strategy=split_strategy))
        dataset = MultiDataset(datasets)
    else:
        dataset = GaussianDataset(cfg.dataset, split_strategy=split_strategy)

    shuffle = split_strategy == "train"
    return DataLoaderX(dataset, batch_size=cfg.batch_size, shuffle=shuffle, pin_memory=True), dataset


def load_checkpoint_if_exists(module, checkpoint_path, strict=True):
    """Load checkpoint into module if path exists."""
    if os.path.exists(checkpoint_path):
        module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=strict)
        print(f"Loaded checkpoint from {checkpoint_path}")
        return True
    return False


if __name__ == "__main__":
    # ---------------- Argument Parsing ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_s2_N031.yaml")
    parser.add_argument("--dataroot", type=str, nargs="+", default=[])
    arg = parser.parse_args()

    # ---------------- Config ----------------
    cfg = config_train()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    # ---------------- Data ----------------
    dataloader, dataset = build_dataloader(cfg, arg.dataroot, split_strategy="train")

    # ---------------- Device ----------------
    device = torch.device(f"cuda:{cfg.gpu_id}")
    torch.cuda.set_device(cfg.gpu_id)

    # ---------------- Super-Resolution Module ----------------
    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    load_checkpoint_if_exists(supres, cfg.load_supres_checkpoint)
    optimized_parameters = [{"params": supres.parameters(), "lr": cfg.lr_net, "name": "supres"}]

    # ---------------- Gaussian Head / Mesh Head ----------------
    if cfg.gaussianheadmodule.enable and not cfg.flame_gaussian_module.enable:
        if load_checkpoint_if_exists(None, cfg.load_gaussianhead_checkpoint):
            gaussianhead_state = torch.load(cfg.load_gaussianhead_checkpoint, map_location="cpu")
            gaussianhead = GaussianHeadModule(
                cfg.gaussianheadmodule,
                xyz=gaussianhead_state["xyz"],
                feature=gaussianhead_state["feature"],
                landmarks_3d_neutral=gaussianhead_state["landmarks_3d_neutral"],
            ).to(device)
            gaussianhead.load_state_dict(gaussianhead_state)
        else:
            meshhead_state = torch.load(cfg.load_meshhead_checkpoint, map_location="cpu")
            meshhead = MeshHeadModule(cfg.meshheadmodule, meshhead_state["landmarks_3d_neutral"]).to(device)
            meshhead.load_state_dict(meshhead_state)
            meshhead.subdivide()
            with torch.no_grad():
                data = meshhead.reconstruct_neutral()

            select_indices = range(data["verts"].shape[0])
            gaussianhead = GaussianHeadModule(
                cfg.gaussianheadmodule,
                xyz=data["verts"][select_indices].cpu(),
                feature=torch.atanh(data["verts_feature"][select_indices].cpu()),
                landmarks_3d_neutral=meshhead.landmarks_3d_neutral.detach().cpu(),
                add_mouth_points=True,
            ).to(device)

            # Transfer weights
            for attr in ["exp_color_mlp", "pose_color_mlp", "exp_deform_mlp", "pose_deform_mlp"]:
                getattr(gaussianhead, attr).load_state_dict(getattr(meshhead, attr).state_dict())

            # Cleanup
            meshhead = meshhead.cpu()
            del meshhead
            torch.cuda.empty_cache()

            # Optimizer for gaussian head
            gaussianhead_optimized_parameters = [
                {"params": gaussianhead.xyz, "lr": cfg.lr_net * 0.1, "name": "xyz"},
                {"params": gaussianhead.feature, "lr": cfg.lr_net * 0.1, "name": "feature"},
                {"params": gaussianhead.exp_color_mlp.parameters(), "lr": cfg.lr_net, "name": "exp_color_mlp"},
                {"params": gaussianhead.pose_color_mlp.parameters(), "lr": cfg.lr_net, "name": "pose_color_mlp"},
                {"params": gaussianhead.exp_deform_mlp.parameters(), "lr": cfg.lr_net, "name": "exp_deform_mlp"},
                {"params": gaussianhead.pose_deform_mlp.parameters(), "lr": cfg.lr_net, "name": "pose_deform_mlp"},
                {"params": gaussianhead.exp_attributes_mlp.parameters(), "lr": cfg.lr_net, "name": "exp_attributes_mlp"},
                {"params": gaussianhead.pose_attributes_mlp.parameters(), "lr": cfg.lr_net, "name": "pose_attributes_mlp"},
                {"params": gaussianhead.scales, "lr": cfg.lr_net * 0.3, "name": "scales"},
                {"params": gaussianhead.rotation, "lr": cfg.lr_net * 0.1, "name": "rotation"},
                {"params": gaussianhead.opacity, "lr": cfg.lr_net, "name": "opacity"},
                {"params": gaussianhead.seg_label, "lr": cfg.lr_net, "name": "seg_label"},
                {"params": gaussianhead.features_dc, "lr": cfg.lr_net * 50, "name": "features_dc"},
                {"params": gaussianhead.features_rest, "lr": cfg.lr_net * 50, "name": "features_rest"},
            ]
            gaussianhead.optimizer = torch.optim.Adam(gaussianhead_optimized_parameters)
    else:
        gaussianhead = GaussianBaseModule().to(device)

    # ---------------- Flame Gaussians ----------------
    gaussians = FlameGaussianModel(
        0,
        disable_flame_static_offset=cfg.gaussianhairmodule.enable,
        not_finetune_flame_params=True,
        n_shape=dataset.shape_dims,
        n_expr=dataset.exp_dims,
    )
    if cfg.flame_gaussian_module.enable and gaussians.binding is not None:
        gaussians.load_meshes(
            train_meshes=dataset.train_meshes, test_meshes={}, tgt_train_meshes={}, tgt_test_meshes={}
        )

    # ---------------- Hair Gaussian ----------------
    gaussianhair = GaussianHairModule(cfg.gaussianhairmodule).to(device)
    gaussianhair.update_mesh_alignment_transform(dataset.R, dataset.T, dataset.S, flame_mesh_path=dataset.flame_mesh_path)

    # ---------------- Camera + Recorder ----------------
    camera = CameraModule()
    recorder = GaussianHeadTrainRecorder(cfg)

    # ---------------- Resume Training ----------------
    start_epoch, checkpoint_seed = cfg.start_epoch, cfg.checkpoint_seed
    if cfg.resume_training:
        start_epoch = 13
        gh_ckpt = cfg.gaussianheadmodule.load_gaussianhead_checkpoint or f"{recorder.checkpoint_path}/{recorder.name}/gaussianhead_latest_{checkpoint_seed}"
        ghair_ckpt = cfg.gaussianhairmodule.load_gaussianhair_checkpoint or f"{recorder.checkpoint_path}/{recorder.name}/gaussianhair_latest_{checkpoint_seed}"
        gply_ckpt = cfg.flame_gaussian_module.load_flame_gaussian_checkpoint or f"{recorder.checkpoint_path}/{recorder.name}/head_latest_{checkpoint_seed}.ply"

        if not cfg.flame_gaussian_module.enable:
            load_checkpoint_if_exists(gaussianhead, gh_ckpt)
        if cfg.flame_gaussian_module.enable:
            if os.path.exists(gply_ckpt):
                gaussians.load_ply(gply_ckpt, has_target=False)
                gaussians.training_setup(cfg.flame_gaussian_module)
                print(f"Loaded gaussians from {gply_ckpt}")
        load_checkpoint_if_exists(gaussianhair, ghair_ckpt, strict=False)

        start_epoch += 1
    else:
        gaussianhair.reset_strands()
        if cfg.flame_gaussian_module.enable:
            gaussians.create_from_pcd(None, cameras_extent=4.907987451553345)
            gaussians.training_setup(cfg.flame_gaussian_module)

    # ---------------- Pose Optimization ----------------
    if os.path.exists(cfg.load_delta_poses_checkpoint):
        delta_poses = torch.load(cfg.load_delta_poses_checkpoint)
    else:
        delta_poses = torch.zeros([dataset.num_exp_id, 6]).to(device)

    if cfg.optimize_pose:
        delta_poses.requires_grad_(True)
        optimized_parameters.append({"params": delta_poses, "lr": cfg.lr_pose})
    else:
        delta_poses.requires_grad_(False)

    optimizer = torch.optim.Adam(optimized_parameters)

    # ---------------- Module Selection ----------------
    if cfg.flame_gaussian_module.enable:
        gaussianhead = gaussians
    elif not cfg.gaussianheadmodule.enable:
        gaussianhead = None
        cfg.merge_from_list(["train_segment", False])

    if not cfg.gaussianhairmodule.enable:
        gaussianhair = None
        cfg.merge_from_list(["train_segment", False])

    # ---------------- Trainer ----------------
    trainer = GaussianHeadHairTrainer(
        dataloader, delta_poses, gaussianhead, gaussianhair, supres, camera, optimizer, recorder, cfg.gpu_id, cfg
    )
    trainer.train(start_epoch, start_epoch + cfg.num_epochs)

    # ---------------- Test / Inference ----------------
    dataloader, dataset = build_dataloader(cfg, arg.dataroot, split_strategy="test")
    app = Reenactment_hair(
        dataloader, gaussianhead, gaussianhair, supres, camera, recorder, cfg.gpu_id, freeview=False, camera_id=25
    )
    app.run()
