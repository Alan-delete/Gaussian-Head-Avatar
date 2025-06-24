export GPU="7"
export CAMERA="PINHOLE"
export EXP_NAME_1="stage1"
export EXP_NAME_2="stage2"
export EXP_NAME_3="stage3"
export EXP_PATH_1=$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1


# Manual steps for now:

# # Renderme dataset
# SUBJECT="0018_e0_raw"
# SUBJECT="0094_e0_raw"
# SUBJECT="0094_h1_7bk_raw"
# SUBJECT="0094_h0_raw"
# SUBJECT="0094_h1_6bn_raw-007"
# SUBJECT="0138_h0_raw"
# SUBJECT="0138_h1_7bk_raw"
# SUBJECT="0322_h1_4bn_raw"
# SUBJECT="0094_h1_3bn_raw"
# DATA_PATH="/local/home/haonchen/Gaussian-Head-Avatar/datasets/RenderMe/$SUBJECT"


# NeRSemble dataset
PROJECT_DIR="/local/home/haonchen/Gaussian-Head-Avatar"
SUBJECT="258"
# SUBJECT="100"
SEQUENCE="HAIR"
SEQUENCE="EXP-1-head"
DATA_PATH="/local/home/haonchen/Gaussian-Head-Avatar/datasets/mini_demo_dataset/$SUBJECT"


#
# Ensure that the following environment variables are accessible to the script:
# PROJECT_DIR and DATA_PATH 
#

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

#################
# PREPROCESSING #
#################



# Input data path
DATA_ROOT="$PROJECT_DIR/datasets/NeRSemble"

# Output data path
DATA_PATH="$DATA_ROOT/$SUBJECT/sequences/${SEQUENCE}"
# DATA_PATH="/local/home/haonchen/Gaussian-Head-Avatar/datasets/mini_demo_dataset/$SUBJECT"

# temporary output folders
TRACK_OUTPUT_FOLDER="$PROJECT_DIR/datasets/output/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_wBg_staticOffset"
EXPORT_OUTPUT_FOLDER="$PROJECT_DIR/datasets/export/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_whiteBg_staticOffset_maskBelowLine"

# # # Donwload NeRsemble dataset
conda activate gha2   
# # # nersemble-data download datasets/NeRSemble/ --participant 258 --sequence 'HAIR','EXP-1-head'
cd  $PROJECT_DIR/preprocess
# python preprocess_nersemble.py --data_source $DATA_ROOT --data_output $DATA_PATH --id_list $SUBJECT --sequence $SEQUENCE


# Run VHAP tracking 
conda activate VHAP
cd $PROJECT_DIR/ext/VHAP

# CUDA_VISIBLE_DEVICES="$GPU" python vhap/preprocess_video.py \
# --input ${DATA_ROOT}/${SUBJECT}/sequences/${SEQUENCE}* \
# --downsample_scales 2 4 \
# --matting_method background_matting_v2

# # Align and track faces
# CUDA_VISIBLE_DEVICES="$GPU" python vhap/track_nersemble_v2.py --data.root_folder ${DATA_ROOT} \
# --exp.output_folder $TRACK_OUTPUT_FOLDER \
# --data.subject $SUBJECT --data.sequence $SEQUENCE \
# --model.no_use_static_offset --data.n_downsample_rgb 4  

# # Export tracking results into a NeRF-style dataset
# CUDA_VISIBLE_DEVICES="$GPU" python vhap/export_as_nerf_dataset.py \
# --src_folder ${TRACK_OUTPUT_FOLDER} \
# --tgt_folder ${EXPORT_OUTPUT_FOLDER} --background-color white

# # Convert structure to images | frame_id | image_camera_id.jpg 
# cd  $PROJECT_DIR/preprocess
# python convert_nersemble.py --data_source ${DATA_ROOT} --intermediate_data ${EXPORT_OUTPUT_FOLDER} --data_output ${DATA_PATH}




# conda deactivate && conda activate gha2
# cd $PROJECT_DIR/preprocess
# python preprocess_renderme.py --smc_path ../datasets/RenderMe/raw/$SUBJECT.smc --output_dir ../datasets/RenderMe/

# # Run Matte-Anything
# # Error: bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
# # Error: TypeError: __init__() got an unexpected keyword argument 'color_lookup'
# # Sol: change sv.BoxAnnotator to sv.BoundingBoxAnnotator
# conda deactivate && conda activate matte_anything
# cd $PROJECT_DIR/preprocess
# # img_size 256 and kernel_size 5 tested good
# CUDA_VISIBLE_DEVICES="$GPU" python calc_masks.py \
#     --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 512 \
#     --kernel_size 5

# # Run background matting
# conda deactivate && conda activate gha2
# cp $PROJECT_DIR/preprocess/remove_background_nersemble.py $PROJECT_DIR/ext/BackgroundMattingV2/
# cd $PROJECT_DIR/ext/BackgroundMattingV2
# CUDA_VISIBLE_DEVICES="$GPU" python remove_background_nersemble.py --data_path $DATA_PATH --dataset renderme

# # face parsing
# cd $PROJECT_DIR/preprocess
# conda deactivate && conda activate gha2
# CUDA_VISIBLE_DEVICES="$GPU" python face_parse.py --model resnet34 --weight $PROJECT_DIR/ext/face-parsing/weights/resnet34.pt --input $DATA_PATH/images  --output $DATA_PATH/face-parsing


# # face mask from neural haircut
# cd $PROJECT_DIR/preprocess
# conda deactivate && conda activate matte_anything
# # conda deactivate && conda activate gha2
# CUDA_VISIBLE_DEVICES="$GPU" python calc_masks.py \
#     --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 2048\
#     --kernel_size 5 \
#     --MODNET_ckpt $PROJECT_DIR/assets/MODNet/modnet_photographic_portrait_matting.ckpt \
#     --CDGNET_ckpt $PROJECT_DIR/assets/CDGNet/LIP_epoch_149.pth \
#     --ext_dir $PROJECT_DIR/ext/
#     # --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 512 \



# # conda activate sapiens_lite 
# # cd $PROJECT_DIR/src/preprocessing && ./depth.sh


# # # landmark detection and FLAME fitting
# cd $PROJECT_DIR/preprocess
# conda deactivate && conda activate mv-3dmm-fitting
# # CUDA_VISIBLE_DEVICES="$GPU" python detect_landmarks.py \
# #     --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks --image_size 2048
# CUDA_VISIBLE_DEVICES="$GPU" python fitting.py \
#     --config $PROJECT_DIR/config/FLAME_fitting_NeRSemble_031.yaml \
#     --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks \
#     --param_folder $DATA_PATH/FLAME_params --camera_folder $DATA_PATH/cameras --image_size 2048
    
# CUDA_VISIBLE_DEVICES="$GPU" python fitting.py \
#     --config $PROJECT_DIR/config/BFM_fitting_NeRSemble_031.yaml \
#     --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks \
#     --param_folder $DATA_PATH/params --camera_folder $DATA_PATH/cameras --image_size 2048



# conda deactivate 
# cd $PROJECT_DIR/ext/DenseMatching
# conda activate dense_matching_env
# python -c "from admin.environment import create_default_local_file; create_default_local_file()"
# cp $PROJECT_DIR/preprocess/calc_optical_flow.py  $PROJECT_DIR/ext/DenseMatching/calc_optical_flow.py
# # CUDA_VISIBLE_DEVICES="$GPU" python calc_optical_flow.py --model GLUNet_GOCor --pre_trained_model dynamic --img_dir $DATA_PATH/images --optical_flow_dir $DATA_PATH/optical_flow
# CUDA_VISIBLE_DEVICES="$GPU" python calc_optical_flow.py --model PDCNet_plus --pre_trained_model megadepth --img_dir $DATA_PATH/images --optical_flow_dir $DATA_PATH/optical_flow


# # SMPLX tracking
# conda activate multihmr
# source .multihmr/bin/activate
# cp $PROJECT_DIR/preprocess/smplx_tracking.py $PROJECT_DIR/ext/multi-hmr/
# cd $PROJECT_DIR/ext/multi-hmr
# # python3.9 smplx_tracking.py \
# #     --img_folder $DATA_PATH1/images \
# #     --out_folder $DATA_PATH1/smplx \
# #     --extra_views 0 \
# #     --model_name multiHMR_896_L
# python3.9 demo.py \
#     --img_folder example_data \
#     --out_folder demo_out \
#     --extra_views 0 \
#     --model_name multiHMR_896_L

# conda activate depth-pro 
# cd $PROJECT_DIR/preprocess
# CUDA_VISIBLE_DEVICES="$GPU" python calc_depth.py \
#     --data_path $DATA_PATH --module_path $PROJECT_DIR/ext/ml-depth-pro 

# # Arrange raw images into a 3D Gaussian Splatting format
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python preprocess_raw_images.py \
#     --data_path $DATA_PATH

# # Run COLMAP reconstruction and undistort the images and cameras
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python convert.py -s $DATA_PATH \
#     --camera $CAMERA --max_size 1024


# # Filter images using their IQA scores
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python filter_extra_images.py \
#     --data_path $DATA_PATH --max_imgs 128

# # Resize images
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python resize_images.py --data_path $DATA_PATH


# # Calculate orientation maps
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
#     --img_path $DATA_PATH/images_2 \
#     --mask_path $DATA_PATH/masks_2/hair \
#     --orient_dir $DATA_PATH/orientations_2/angles \
#     --conf_dir $DATA_PATH/orientations_2/vars \
#     --filtered_img_dir $DATA_PATH/orientations_2/filtered_imgs \
#     --vis_img_dir $DATA_PATH/orientation_2/vis_imgs

# # cd $PROJECT_DIR/preprocess 
# conda deactivate && conda activate gha2
# CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
#     --img_dir $DATA_PATH/images --mask_dir $DATA_PATH/NeuralHaircut_masks/hair --orient_dir $DATA_PATH/orientation_maps \
#     --conf_dir $DATA_PATH/orientation_confidence_maps --filtered_img_dir $DATA_PATH/orientation_filtered_imgs --vis_img_dir $DATA_PATH/orientation_vis_imgs

# # Run OpenPose
# conda deactivate && cd $PROJECT_DIR/ext/openpose
# mkdir $DATA_PATH/openpose
# CUDA_VISIBLE_DEVICES="$GPU" ./build/examples/openpose/openpose.bin \
#     --image_dir $DATA_PATH/images_4 \
#     --scale_number 4 --scale_gap 0.25 --face --hand --display 0 \
#     --write_json $DATA_PATH/openpose/json \
#     --write_images $DATA_PATH/openpose/images --write_images_format jpg

# # Run Face-Alignment
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python calc_face_alignment.py \
#     --data_path $DATA_PATH --image_dir "images_4"

# # Run PIXIE
# conda deactivate && conda activate pixie-env
# cd $PROJECT_DIR/ext/PIXIE
# CUDA_VISIBLE_DEVICES="$GPU" python demos/demo_fit_face.py \
#     -i $DATA_PATH/images_4 -s $DATA_PATH/pixie \
#     --saveParam True --lightTex False --useTex False \
#     --rasterizer_type pytorch3d \
#     --saveObj True

# conda deactivate && conda activate pixie-env
# cd $PROJECT_DIR/ext/PIXIE
# CUDA_VISIBLE_DEVICES="$GPU" python demos/demo_fit_body.py \
#     -i $DATA_PATH/images_4 -s $DATA_PATH/pixie_body \
#     --saveParam True --lightTex False --useTex False \
#     --rasterizer_type pytorch3d \
#     --saveObj True


# # run multi-view optimization in NeuralHaircut, need new conf to fit the whole body
# conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/ext/NeuralHaircut/src/multiview_optimization
# CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf $PROJECT_DIR/src/preprocessing/conf/train_person_whole_body.conf \
#     --batch_size 128 --train_rotation True --train_pose True --train_shape True\
#     --save_path $DATA_PATH/body_fitting/$EXP_NAME_1/stage_1 \
#     --data_path $DATA_PATH \
#     --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl \
#     --checkpoint_path $DATA_PATH/pixie_body/000006/000006_param.pkl 

# # Merge all PIXIE predictions in a single file
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python merge_smplx_predictions.py \
#     --data_path $DATA_PATH

# # Convert COLMAP cameras to txt
# conda deactivate && conda activate gaussian_splatting_hair
# mkdir $DATA_PATH/sparse_txt
# CUDA_VISIBLE_DEVICES="$GPU" colmap model_converter \
#     --input_path $DATA_PATH/sparse/0  \
#     --output_path $DATA_PATH/sparse_txt --output_type TXT

# # Convert COLMAP cameras to H3DS format
# conda deactivate && conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python colmap_parsing.py \
#     --path_to_scene $DATA_PATH

# # Remove raw files to preserve disk space
# rm -rf $DATA_PATH/input $DATA_PATH/images $DATA_PATH/masks $DATA_PATH/iqa*

# ##################
# # RECONSTRUCTION #
# ##################

cd $PROJECT_DIR
conda activate gha2 
# CUDA_VISIBLE_DEVICES="$GPU" python train_meshhead.py --config config/train_meshhead_N$SUBJECT.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python train_meshhead.py --config config/train_meshhead_renderme.yaml --dataroot $DATA_PATH
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianhead.py --config config/train_gaussianhead_N031.yaml
CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianhead.py --config config/train_gaussianhead_N$SUBJECT.yaml --dataroot $DATA_PATH

# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_N$SUBJECT.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme_single.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianhead.py --config config/train_gaussianhead_renderme.yaml --dataroot $DATA_PATH

# CUDA_VISIBLE_DEVICES="$GPU" python train_opticalflow.py --config config/train_gaussianhead_hair_renderme_optical_flow.yaml --dataroot $DATA_PATH

# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme_mlp.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_hair_renderme.yaml --dataroot $DATA_PATH --test_camera_id 25 
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_hair_renderme_single.yaml --dataroot $DATA_PATH --test_camera_id 25
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_flame_gaussian_renderme.yaml --dataroot $DATA_PATH

# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme_single_onlyhair.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_hair_renderme_single_onlyhair.yaml --dataroot $DATA_PATH


# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme_onlyhair.yaml --dataroot $DATA_PATH

# Run 3D Gaussian Splatting reconstruction
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussians.py \
#     -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
#     --trainable_cameras --trainable_intrinsics --use_barf \
#     --lambda_dorient 0.1 --train_orient_conf --render_direction

# conda activate GaussianHaircutv2 && cd $PROJECT_DIR/src
# # conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python train_latent_strands_perm_holistic_joint.py \
#     -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
#     -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
#     --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
#     --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
#     --lambda_dmask 0.1 --lambda_dorient 0.05 --lambda_dsds 0.01 \
#     --position_lr_init 0.0000016 --position_lr_max_steps 10000 \
#     --use_barf \
#     --port "800$GPU" --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured_perm_joint.yaml"

    # --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \

# conda activate gaussian_splatting_hair_abs && cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussians.py \
#     -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
#     --trainable_cameras --trainable_intrinsics --use_barf \
#     --lambda_dorient 0.1 --train_orient_conf --render_direction

# # Run FLAME mesh fitting
# conda activate gaussian_splatting_hair
# cd $PROJECT_DIR/ext/NeuralHaircut/src/multiview_optimization

# CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1.conf \
#     --batch_size 1 --train_rotation True --fixed_images True \
#     --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_1 \
#     --data_path $DATA_PATH \
#     --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

# CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1.conf \
#     --batch_size 4 --train_rotation True --fixed_images True \
#     --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_2 \
#     --checkpoint_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_1/opt_params_final \
#     --data_path $DATA_PATH \
#     --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

# CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1_.conf \
#     --batch_size 32 --train_rotation True --train_shape True \
#     --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_3 \
#     --checkpoint_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_2/opt_params_final \
#     --data_path $DATA_PATH \
#     --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

# # Crop the reconstructed scene
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python scale_scene_into_sphere.py \
#     --path_to_data $DATA_PATH \
#     -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000

# # Remove hair Gaussians that intersect with the FLAME head mesh
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python filter_flame_intersections.py \
#     --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
#     -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000 \
#     --project_dir $PROJECT_DIR/ext/NeuralHaircut

# # Run rendering for training views
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python render_gaussians.py \
#     -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" \
#     --skip_test --scene_suffix "_cropped" --iteration 30000 \
#     --trainable_cameras --trainable_intrinsics --use_barf \
#     --render_direction

# # Get FLAME mesh scalp maps
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python extract_non_visible_head_scalp.py \
#     --project_dir $PROJECT_DIR/ext/NeuralHaircut --data_dir $DATA_PATH \
#     --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
#     --cams_path $DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1/cameras/30000_matrices.pkl \
#     -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1"

# # Run latent hair strands reconstruction
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python train_latent_strands.py \
#     -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
#     --model_path_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2" \
#     --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
#     --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
#     --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
#     --lambda_dmask 0.1 --lambda_dorient 0.1 --render_direction --lambda_dsds 0.01 \
#     --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
#     --trainable_cameras --trainable_intrinsics --use_barf \
#     --iterations 20000 --port "800$GPU"

# # Run hair strands reconstruction
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python train_strands.py \
#     -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
#     --model_path_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
#     --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
#     --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
#     --start_checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
#     --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
#     --lambda_dmask 0.1 --lambda_dorient 0.1 --render_direction --lambda_dsds 0.01 \
#     --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
#     --position_lr_init 0.0000016 --position_lr_max_steps 10000 \
#     --trainable_cameras --trainable_intrinsics --use_barf \
#     --iterations 10000 --port "800$GPU"

# # arrive here already
# rm -rf "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1/train_cropped"

# ##################
# # VISUALIZATIONS #
# ##################

# # Export the resulting strands as pkl and ply
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python export_curves.py \
#     --data_dir $DATA_PATH --model_name $EXP_NAME_3 --iter 10000 \
#     --flame_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/stage_3/mesh_final.obj" \
#     --scalp_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/scalp_data/scalp.obj" \
#     --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml"

# # Render the visualizations
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python render_video.py \
#     --blender_path "/home/ezakharov/Libraries/blender-3.6.11-linux-x64/blender" \
#     --input_path "$DATA_PATH" --exp_name_1 "$EXP_NAME_1" --exp_name_3 "$EXP_NAME_3"

# # Render the strands
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
# CUDA_VISIBLE_DEVICES="$GPU" python render_strands.py \
#     -s $DATA_PATH --data_dir "$DATA_PATH" --data_device 'cpu' --skip_test \
#     -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iteration 30000 \
#     --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
#     --model_hair_path "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
#     --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
#     --checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
#     --checkpoint_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3/checkpoints/10000.pth" \
#     --pointcloud_path_head "$EXP_PATH_1/point_cloud/iteration_30000/raw_point_cloud.ply" \
#     --interpolate_cameras --speed_up 4 --max_frames 200

# # Make the video
# conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
# CUDA_VISIBLE_DEVICES="$GPU" python concat_video.py \
#     --input_path "$DATA_PATH" --exp_name_3 "$EXP_NAME_3"