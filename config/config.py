import os
from yacs.config import CfgNode as CN
 

class config_base():

    def __init__(self):
        self.cfg = CN()
    
    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self,config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()


class config_train(config_base):

    def __init__(self):
        super(config_train, self).__init__()

        self.cfg.gpu_id = 0                                     # which gpu is used
        self.cfg.load_meshhead_checkpoint = ''                  # checkpoint path of mesh head
        self.cfg.load_gaussianhead_checkpoint = ''              # checkpoint path of gaussian head
        self.cfg.load_supres_checkpoint = ''                    # checkpoint path of super resolution network
        self.cfg.load_delta_poses_checkpoint = ''               # checkpoint path of per-frame offset of head pose
        self.cfg.lr_net = 0.0                                   # learning rate for models and networks
        self.cfg.lr_lmk = 0.0                                   # learning rate for 3D landmarks
        self.cfg.lr_pose = 0.0                                  # learning rate for delta_poses
        self.cfg.batch_size = 1                                 # recommend batch_size = 1
        self.cfg.optimize_pose = False                          # optimize delta_poses or not
        self.cfg.use_supres = True                              # use super resolution network or not, if not, directly use first 3 channels of the 32 channels 
        self.cfg.train_segment = False                          # train the segmentation or not
        self.cfg.train_optical_flow = False                       # train the optical flow or not
        self.cfg.resume_training = False                        # resume training or not
        self.cfg.start_epoch = 0
        self.cfg.num_epochs = 300                            # number of epochs for training
        self.cfg.num_iterations = -1                      # number of iterations for training, will overwrite num_epochs
        self.cfg.static_scene_init = False                     # initialize the scene with static images or not
        self.cfg.static_training_util_iter = 25000              # static training util iteration
        self.cfg.checkpoint_seed = -1

        self.cfg.loss_weights = CN()
        self.cfg.loss_weights.rgb_hr = 1.0                      # loss for high resolution image
        self.cfg.loss_weights.rgb_lr = 1.0                      # loss for low resolution image
        self.cfg.loss_weights.dssim = 2e-1                      # loss for SSIM
        self.cfg.loss_weights.vgg = 1e-1                        # loss for perceptual loss
        self.cfg.loss_weights.segment = 1.25e1                  # loss for segmentation
        self.cfg.loss_weights.transform_reg = 1e-4              # loss for transformation regularization
        self.cfg.loss_weights.dir = 1e1                         # loss for prior
        self.cfg.loss_weights.mesh_dist = 1e-4                  # loss for mesh distance
        self.cfg.loss_weights.knn_feature = 1e-4                # loss for knn feature loss
        self.cfg.loss_weights.orient = 1e-1                     # loss for orientation
        self.cfg.loss_weights.strand_feature = 1e-2              # loss for strand feature
        self.cfg.loss_weights.sign_distance = 1e-2              # loss for sign distance
        self.cfg.loss_weights.deform_reg = 2e-2                  # loss for deformation regularization

        self.cfg.dataset = CN()
        self.cfg.dataset.dataroot = ''                          # root of the dataset
        self.cfg.dataset.camera_ids = []                        # which cameras are used
        self.cfg.dataset.test_camera_ids = [0]                   # which cameras are used for testing
        self.cfg.dataset.original_resolution = 2048             # original image resolution, should match the intrinsic
        self.cfg.dataset.resolution = 512                       # image resolution for rendering
        self.cfg.dataset.coarse_scale_factor = 0.25                # scale factor for coarse image
        self.cfg.dataset.num_sample_view = 8                    # number of sampled images from different views during mesh head training
        self.cfg.dataset.selected_frames = []                           # selected frames for training, if not set, all frames will be used
        
        self.cfg.meshheadmodule = CN()
        self.cfg.meshheadmodule.geo_mlp = []                    # dimensions of geometry MLP
        self.cfg.meshheadmodule.exp_color_mlp = []              # dimensions of expression color MLP
        self.cfg.meshheadmodule.pose_color_mlp = []             # dimensions of pose color MLP
        self.cfg.meshheadmodule.exp_deform_mlp = []             # dimensions of expression deformation MLP
        self.cfg.meshheadmodule.pose_deform_mlp = []            # dimensions of pose deformation MLP
        self.cfg.meshheadmodule.pos_freq = 4                    # frequency of positional encoding
        self.cfg.meshheadmodule.model_bbox = []                 # bounding box of the head model
        self.cfg.meshheadmodule.dist_threshold_near = 0.1       # threshold t1
        self.cfg.meshheadmodule.dist_threshold_far = 0.2        # thresgold t2
        self.cfg.meshheadmodule.deform_scale = 0.3              # scale factor for deformation
        self.cfg.meshheadmodule.subdivide = False               # subdivide the tetmesh (resolution: 128 --> 256) or not

        self.cfg.supresmodule = CN()
        self.cfg.supresmodule.input_dim = 32                    # input dim, equal to the channel number of the multi-channel color
        self.cfg.supresmodule.output_dim = 3                    # output dim, euqal to the channel number of the final image
        self.cfg.supresmodule.network_capacity = 64             # dimension of the network's last conv layer

        self.cfg.gaussianheadmodule = CN()
        self.cfg.gaussianheadmodule.load_gaussianhead_checkpoint = ''              # checkpoint path of gaussian head
        self.cfg.gaussianheadmodule.enable = True               # whether to use the Gaussian head module
        self.cfg.gaussianheadmodule.num_add_mouth_points = 0    # number of the points added around mouth landmarks while initialization
        self.cfg.gaussianheadmodule.exp_color_mlp = []          # dimensions of expression color MLP
        self.cfg.gaussianheadmodule.pose_color_mlp = []         # dimensions of pose color MLP
        self.cfg.gaussianheadmodule.exp_attributes_mlp = []     # dimensions of expression attribute MLP
        self.cfg.gaussianheadmodule.pose_attributes_mlp = []    # dimensions of pose attribute MLP
        self.cfg.gaussianheadmodule.exp_deform_mlp = []         # dimensions of expression deformation MLP
        self.cfg.gaussianheadmodule.pose_deform_mlp = []        # dimensions of pose deformation MLP
        self.cfg.gaussianheadmodule.exp_coeffs_dim = 64         # dimension of the expression coefficients
        self.cfg.gaussianheadmodule.pos_freq = 4                # frequency of positional encoding
        self.cfg.gaussianheadmodule.dist_threshold_near = 0.1   # threshold t1
        self.cfg.gaussianheadmodule.dist_threshold_far = 0.2    # thresgold t2
        self.cfg.gaussianheadmodule.deform_scale = 0.3          # scale factor for deformation
        self.cfg.gaussianheadmodule.attributes_scale = 0.05     # scale factor for attribute offset
        self.cfg.gaussianheadmodule.densify = False             # densify the GS or not    
        self.cfg.gaussianheadmodule.densify_from_iter = 500     # densify the GS from this iteration
        self.cfg.gaussianheadmodule.densify_until_iter = 15_000
        self.cfg.gaussianheadmodule.densification_interval = 100
        self.cfg.gaussianheadmodule.densify_grad_threshold = 0.0002
        self.cfg.gaussianheadmodule.opacity_reset_interval = 3_000
        self.cfg.gaussianheadmodule.opacity_reg_from_iter = 30_000
        self.cfg.gaussianheadmodule.gaussian_pruning_threshold = 0.5


        self.cfg.flame_gaussian_module = CN()
        self.cfg.flame_gaussian_module.load_flame_gaussian_checkpoint = ''              # checkpoint path of gaussian head
        self.cfg.flame_gaussian_module.enable = True
        self.cfg.flame_gaussian_module.iterations = 600_000  # 30_000 (original)
        self.cfg.flame_gaussian_module.position_lr_init = 0.005  # (scaled up according to mean triangle scale)  #0.00016 (original)
        self.cfg.flame_gaussian_module.position_lr_final = 0.00005  # (scaled up according to mean triangle scale) # 0.0000016 (original)
        self.cfg.flame_gaussian_module.position_lr_delay_mult = 0.01
        self.cfg.flame_gaussian_module.position_lr_max_steps = 600_000  # 30_000 (original)
        self.cfg.flame_gaussian_module.feature_lr = 0.0025
        self.cfg.flame_gaussian_module.opacity_lr = 0.05
        self.cfg.flame_gaussian_module.scaling_lr = 0.017  # (scaled up according to mean triangle scale)  # 0.005 (original)
        self.cfg.flame_gaussian_module.rotation_lr = 0.001
        self.cfg.flame_gaussian_module.densify = True
        self.cfg.flame_gaussian_module.densification_interval = 1_000  # 100 (original)
        self.cfg.flame_gaussian_module.opacity_reset_interval = 60_000 # 3000 (original)
        self.cfg.flame_gaussian_module.densify_from_iter = 10_000  # 500 (original)
        self.cfg.flame_gaussian_module.densify_until_iter = 600_000  # 15_000 (original)
        self.cfg.flame_gaussian_module.densify_grad_threshold = 0.0002
        
        # GaussianAvatars
        self.cfg.flame_gaussian_module.flame_expr_lr = 1e-3
        self.cfg.flame_gaussian_module.flame_trans_lr = 1e-6
        self.cfg.flame_gaussian_module.flame_pose_lr = 1e-5
        self.cfg.flame_gaussian_module.percent_dense = 0.01
        self.cfg.flame_gaussian_module.lambda_dssim = 0.2
        self.cfg.flame_gaussian_module.lambda_xyz = 1e-2
        self.cfg.flame_gaussian_module.threshold_xyz = 1.
        self.cfg.flame_gaussian_module.metric_xyz = False
        self.cfg.flame_gaussian_module.lambda_scale = 1.
        self.cfg.flame_gaussian_module.threshold_scale = 0.6
        self.cfg.flame_gaussian_module.metric_scale = False
        self.cfg.flame_gaussian_module.lambda_dynamic_offset = 0.
        self.cfg.flame_gaussian_module.lambda_laplacian = 0.
        self.cfg.flame_gaussian_module.lambda_dynamic_offset_std = 0  #1.


        self.cfg.gaussianhairmodule = CN()
        self.cfg.gaussianhairmodule.load_gaussianhair_checkpoint = ''              # checkpoint path of gaussian hair
        self.cfg.gaussianhairmodule.enable = True               # whether to use the Gaussian hair module
        # contain all pose related 
        self.cfg.gaussianhairmodule.pose_deform_method = 'attention' # deformation method, can be 'Attention' or 'MLP'
        self.cfg.gaussianhairmodule.pose_color_mlp = []         # dimensions of pose color MLP
        self.cfg.gaussianhairmodule.pose_attributes_mlp = []    # dimensions of pose attribute MLP  
        self.cfg.gaussianhairmodule.pose_deform_mlp = []        # dimensions of pose deformation MLP
        self.cfg.gaussianhairmodule.pose_point_mlp = []         # dimensions of pose point MLP
        self.cfg.gaussianhairmodule.pose_prior_mlp = []        # dimension of the pose coefficients
        self.cfg.gaussianhairmodule.pos_freq = 4                # frequency of positional encoding
        self.cfg.gaussianhairmodule.dist_threshold_near = 0.1   # threshold t1
        self.cfg.gaussianhairmodule.dist_threshold_far = 0.2    # thresgold t2
        self.cfg.gaussianhairmodule.deform_scale = 0.3          # scale factor for deformation
        self.cfg.gaussianhairmodule.attributes_scale = 0.05     # scale factor for attribute offset
        self.cfg.gaussianhairmodule.num_strands = 10140         # number of strands
        self.cfg.gaussianhairmodule.strand_length = 100         # length of the strand
        self.cfg.gaussianhairmodule.strand_scale = 0.001       # scale factor for the strand  
        
        self.cfg.gaussianhairmodule.simplify_strands = False    # simplify the strands or not
        self.cfg.gaussianhairmodule.aspect_ratio = 1            # aspect ratio of the hair
        self.cfg.gaussianhairmodule.quantile = 1                # quantile of the hair width
        self.cfg.gaussianhairmodule.train_directions = False    # train the hair directions or not
        self.cfg.gaussianhairmodule.train_features_rest = False # train the rest features or not
        self.cfg.gaussianhairmodule.train_width = False         # train the hair width or not
        self.cfg.gaussianhairmodule.train_opacity = False       # train the hair opacity or not
        self.cfg.gaussianhairmodule.sh_degree = 3               # degree of spherical harmonics
        
        self.cfg.gaussianhairmodule.strands_reset_from_iter = 4_000
        self.cfg.gaussianhairmodule.strands_reset_until_iter = 15_000
        self.cfg.gaussianhairmodule.strands_reset_interval = 1_000

        # for perm 
        self.cfg.gaussianhairmodule.lrs = CN()  
        self.cfg.gaussianhairmodule.milestones = CN()    
        self.cfg.gaussianhairmodule.lrs.theta = 0.005
        self.cfg.gaussianhairmodule.lrs.beta = 0.01
        self.cfg.gaussianhairmodule.lrs.G_raw = 0.001
        self.cfg.gaussianhairmodule.lrs.G_superres = 0.001
        self.cfg.gaussianhairmodule.lrs.G_res = 0.001
        self.cfg.gaussianhairmodule.milestones.theta = [0, 15_000]
        self.cfg.gaussianhairmodule.milestones.G_raw = [0, 15_000]
        self.cfg.gaussianhairmodule.milestones.G_superres = [0, 15_000]
        self.cfg.gaussianhairmodule.milestones.beta = [1_000, 15_000]
        self.cfg.gaussianhairmodule.milestones.G_res = [1_000, 15_000]

        # for raw strucutured gaussian
        self.cfg.gaussianhairmodule.position_lr_init = 0.00016
        self.cfg.gaussianhairmodule.position_lr_final = 0.0000016
        self.cfg.gaussianhairmodule.position_lr_delay_mult = 0.01
        self.cfg.gaussianhairmodule.position_lr_max_steps = 30_000
        self.cfg.gaussianhairmodule.feature_lr = 0.0025
        self.cfg.gaussianhairmodule.opacity_lr = 0.05
        self.cfg.gaussianhairmodule.label_lr = 0.05
        self.cfg.gaussianhairmodule.orient_conf_lr = 0.05
        self.cfg.gaussianhairmodule.scaling_lr = 0.005
        self.cfg.gaussianhairmodule.rotation_lr = 0.001



        self.cfg.recorder = CN()
        self.cfg.recorder.debug_tool = 'tensorboard'            # debug tool, tensorboard or wandb
        self.cfg.recorder.name = ''                             # name of the avatar
        self.cfg.recorder.logdir = ''                           # directory of the tensorboard log
        self.cfg.recorder.checkpoint_path = ''                  # path to the saved checkpoints
        self.cfg.recorder.result_path = ''                      # path to the visualization results
        self.cfg.recorder.save_freq = 1                         # how often the checkpoints are saved
        self.cfg.recorder.show_freq = 1                         # how often the visualization results are saved



class config_reenactment(config_base):

    def __init__(self):
        super(config_reenactment, self).__init__()

        self.cfg.gpu_id = 0                                     # which gpu is used
        self.cfg.load_gaussianhead_checkpoint = ''              # checkpoint path of gaussian head
        self.cfg.load_supres_checkpoint = ''                    # checkpoint path of super resolution network

        self.cfg.dataset = CN()
        self.cfg.dataset.dataroot = ''                          # root of the dataset
        self.cfg.dataset.image_files = ''                       # file names of input images
        self.cfg.dataset.param_files = ''                       # file names of BFM parameters (head pose and expression coefficients)
        self.cfg.dataset.camera_path = ''                       # path of a specific camera
        self.cfg.dataset.pose_code_path = ''                    # path of a specific pose code (as network input)
        self.cfg.dataset.freeview = False                       # freeview rendering or using the specific camera
        self.cfg.dataset.original_resolution = 2048             # original image resolution, should match the intrinsic
        self.cfg.dataset.resolution = 512                       # image resolution for rendering
        
        self.cfg.supresmodule = CN()
        self.cfg.supresmodule.input_dim = 32                    # input dim, equal to the channel number of the multi-channel color
        self.cfg.supresmodule.output_dim = 3                    # output dim, euqal to the channel number of the final image
        self.cfg.supresmodule.network_capacity = 64             # dimension of the network's last conv layer

        self.cfg.gaussianheadmodule = CN()
        self.cfg.gaussianheadmodule.num_add_mouth_points = 0    # number of the points added around mouth landmarks while initialization
        self.cfg.gaussianheadmodule.exp_color_mlp = []          # dimensions of expression color MLP
        self.cfg.gaussianheadmodule.pose_color_mlp = []         # dimensions of pose color MLP
        self.cfg.gaussianheadmodule.exp_attributes_mlp = []     # dimensions of expression attribute MLP
        self.cfg.gaussianheadmodule.pose_attributes_mlp = []    # dimensions of pose attribute MLP
        self.cfg.gaussianheadmodule.exp_deform_mlp = []         # dimensions of expression deformation MLP
        self.cfg.gaussianheadmodule.pose_deform_mlp = []        # dimensions of pose deformation MLP
        self.cfg.gaussianheadmodule.exp_coeffs_dim = 64         # dimension of the expression coefficients
        self.cfg.gaussianheadmodule.pos_freq = 4                # frequency of positional encoding
        self.cfg.gaussianheadmodule.dist_threshold_near = 0.1   # threshold t1
        self.cfg.gaussianheadmodule.dist_threshold_far = 0.2    # thresgold t2
        self.cfg.gaussianheadmodule.deform_scale = 0.3          # scale factor for deformation
        self.cfg.gaussianheadmodule.attributes_scale = 0.05     # scale factor for attribute offset

        self.cfg.recorder = CN()
        self.cfg.recorder.name = ''                             # name of the avatar
        self.cfg.recorder.result_path = ''                      # path to the visualization results