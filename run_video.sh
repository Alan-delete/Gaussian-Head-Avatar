export GPU="5"

# NeRSemble dataset
PROJECT_DIR="/local/home/haonchen/Gaussian-Head-Avatar"
SUBJECT="226"
SUBJECT="black"
SEQUENCE="HAIR"
SEQUENCE="EXP-1-head"
# DATA_PATH="/local/home/haonchen/Gaussian-Head-Avatar/datasets/mini_demo_dataset/$SUBJECT"



# # Renderme dataset
# SUBJECT="0094_h1_3bn_raw"
# DATA_PATH="/local/home/haonchen/Gaussian-Head-Avatar/datasets/RenderMe/$SUBJECT"
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
DATA_PATH1="$DATA_ROOT/$SUBJECT/sequences/${SEQUENCE1}"

# temporary output folders
TRACK_OUTPUT_FOLDER="$PROJECT_DIR/datasets/output/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_wBg_staticOffset"
EXPORT_OUTPUT_FOLDER="$PROJECT_DIR/datasets/export/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_whiteBg_staticOffset_maskBelowLine"

# # # Donwload NeRsemble dataset
conda activate gha2   
# nersemble-data download datasets/NeRSemble/ --participant $SUBJECT --sequence 'HAIR','EXP-1-head'
cd  $PROJECT_DIR/preprocess
# python preprocess_nersemble.py --data_source $DATA_ROOT --data_output $DATA_PATH --id_list $SUBJECT --sequence $SEQUENCE

# python preprocess_havatar.py --data_source $DATA_ROOT --data_output $DATA_PATH --id_list $SUBJECT --sequence $SEQUENCE


# # Run VHAP tracking 
# conda activate VHAP
# cd $PROJECT_DIR/ext/VHAP

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

# # Preprocess RenderMe dataset
# conda deactivate && conda activate gha2
# cd $PROJECT_DIR/preprocess
# python preprocess_renderme.py --smc_path ../datasets/RenderMe/raw/$SUBJECT.smc --output_dir ../datasets/RenderMe/


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



# # # # landmark detection and FLAME fitting
# cd $PROJECT_DIR/preprocess
# conda deactivate && conda activate mv-3dmm-fitting
# CUDA_VISIBLE_DEVICES="$GPU" python detect_landmarks.py \
#     --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks --image_size 2048
# CUDA_VISIBLE_DEVICES="$GPU" python fitting.py \
#     --config $PROJECT_DIR/config/FLAME_fitting_NeRSemble_031.yaml \
#     --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks \
#     --param_folder $DATA_PATH/FLAME_params --camera_folder $DATA_PATH/cameras --image_size 2048
    
# CUDA_VISIBLE_DEVICES="$GPU" python fitting.py \
#     --config $PROJECT_DIR/config/BFM_fitting_NeRSemble_031.yaml \
#     --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks \
#     --param_folder $DATA_PATH/params --camera_folder $DATA_PATH/cameras --image_size 2048



# conda activate depth-pro 
# cd $PROJECT_DIR/preprocess
# CUDA_VISIBLE_DEVICES="$GPU" python calc_depth.py \
#     --data_path $DATA_PATH --module_path $PROJECT_DIR/ext/ml-depth-pro 


# # Calculate orientation maps
# cd $PROJECT_DIR/preprocess 
# conda deactivate && conda activate gha2
# CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
#     --img_dir $DATA_PATH/images --mask_dir $DATA_PATH/NeuralHaircut_masks/hair --orient_dir $DATA_PATH/orientation_maps \
#     --conf_dir $DATA_PATH/orientation_confidence_maps --filtered_img_dir $DATA_PATH/orientation_filtered_imgs --vis_img_dir $DATA_PATH/orientation_vis_imgs


# ##################
# # RECONSTRUCTION #
# ##################

cd $PROJECT_DIR
conda activate gha2 
# CUDA_VISIBLE_DEVICES="$GPU" python train_meshhead.py --config config/train_meshhead_N$SUBJECT.yaml --dataroot $DATA_PATH
CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_N$SUBJECT.yaml --dataroot $DATA_PATH # $DATA_PATH1 # $DATA_PATH2
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config  config/train_gaussianhead_hair_N$SUBJECT.yaml --dataroot $DATA_PATH --test_camera_id 25 

# CUDA_VISIBLE_DEVICES="$GPU" python train_meshhead.py --config config/train_meshhead_renderme.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_hair_renderme.yaml --dataroot $DATA_PATH --test_camera_id 25 

# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme_single.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_hair_renderme_single.yaml --dataroot $DATA_PATH --test_camera_id 25


# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianhead.py --config config/train_gaussianhead_renderme.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_renderme.yaml --dataroot $DATA_PATH --test_camera_id 25 

# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme_single_onlyhair.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_hair_renderme_single_onlyhair.yaml --dataroot $DATA_PATH

# CUDA_VISIBLE_DEVICES="$GPU" python train_opticalflow.py --config config/train_gaussianhead_hair_renderme_optical_flow.yaml --dataroot $DATA_PATH


