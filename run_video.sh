export GPU="5"
#
# Ensure that the following environment variables are accessible to the script:
# PROJECT_DIR and DATA_PATH 
#

PROJECT_DIR="/local/home/haonchen/Gaussian-Head-Avatar"

# For NeRSemble dataset
SUBJECT="226"
# SUBJECT="black"
SEQUENCE="HAIR"
SEQUENCE1="EXP-1-head"

# Input data path
DATA_ROOT="$PROJECT_DIR/datasets/NeRSemble"

# Output data path
DATA_PATH="$DATA_ROOT/$SUBJECT/sequences/${SEQUENCE}"
DATA_PATH1="$DATA_ROOT/$SUBJECT/sequences/${SEQUENCE1}"


# # For Renderme dataset
# SUBJECT="0094_h1_3bn_raw"
# DATA_PATH="/local/home/haonchen/Gaussian-Head-Avatar/datasets/RenderMe/$SUBJECT"

# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

#################
# PREPROCESSING #
#################

# # check and run "preprocess.sh"
# bash preprocess.sh

# ##################
# # RECONSTRUCTION #
# ##################

cd $PROJECT_DIR
conda activate gha2 
# CUDA_VISIBLE_DEVICES="$GPU" python train_meshhead.py --config config/train_meshhead_N$SUBJECT.yaml --dataroot $DATA_PATH
CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_N$SUBJECT.yaml --dataroot $DATA_PATH # $DATA_PATH1
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config  config/train_gaussianhead_hair_N$SUBJECT.yaml --dataroot $DATA_PATH --test_camera_id 25 

# CUDA_VISIBLE_DEVICES="$GPU" python train_meshhead.py --config config/train_meshhead_renderme.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianheadhair.py --config config/train_gaussianhead_hair_renderme.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_hair_renderme.yaml --dataroot $DATA_PATH --test_camera_id 25 

# CUDA_VISIBLE_DEVICES="$GPU" python train_gaussianhead.py --config config/train_gaussianhead_renderme.yaml --dataroot $DATA_PATH
# CUDA_VISIBLE_DEVICES="$GPU" python reenactment_hair.py --config config/train_gaussianhead_renderme.yaml --dataroot $DATA_PATH --test_camera_id 25 

# CUDA_VISIBLE_DEVICES="$GPU" python train_opticalflow.py --config config/train_gaussianhead_hair_renderme_optical_flow.yaml --dataroot $DATA_PATH


