export GPU="4"
# NeRSemble dataset, use your own path
PROJECT_DIR="/local/home/haonchen/Gaussian-Head-Avatar"

# Input data path
DATA_ROOT="$PROJECT_DIR/datasets/NeRSemble"

# Define arrays of subjects and sequences
SUBJECTS=("226")
SEQUENCES=("HAIR" )
SUBJECTS_STR=$(IFS=, ; echo "${SUBJECTS[*]}")
SEQUENCES_STR=$(IFS=, ; echo "${SEQUENCES[*]}")


#################
# PREPROCESSING #
#################


# Activate Conda for environment switching
eval "$(conda shell.bash hook)"

conda activate gha


# # # Donwload NeRsemble dataset
echo "Command: nersemble-data download datasets/NeRSemble/ --participant $SUBJECTS_STR --sequence $SEQUENCES_STR"
yes | nersemble-data download datasets/NeRSemble/ --participant $SUBJECTS_STR --sequence $SEQUENCES_STR 

# Iterate over each subject
for SUBJECT in "${SUBJECTS[@]}"; do
  # Iterate over each sequence
  for SEQUENCE in "${SEQUENCES[@]}"; do
    echo "Processing Subject: $SUBJECT | Sequence: $SEQUENCE"
    
    DATA_PATH="$DATA_ROOT/$SUBJECT/sequences/${SEQUENCE}"
    TRACK_OUTPUT_FOLDER="$PROJECT_DIR/datasets/output/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_wBg_staticOffset"
    EXPORT_OUTPUT_FOLDER="$PROJECT_DIR/datasets/export/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_whiteBg_staticOffset_maskBelowLine"
    
    
    ##############
    # Preprocess #
    ##############
    conda activate gha
    cd $PROJECT_DIR/preprocess
    python preprocess_nersemble.py --data_source $DATA_ROOT --data_output $DATA_PATH --id_list $SUBJECT --sequence $SEQUENCE
    # python preprocess_havatar.py --data_source $DATA_ROOT --data_output $DATA_PATH --id_list $SUBJECT --sequence $SEQUENCE
    # python preprocess_renderme.py --smc_path ../datasets/RenderMe/raw/$SUBJECT.smc --output_dir ../datasets/RenderMe/



    ##############
    # VHAP way(not used for now) #
    ##############

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


    # # Run background matting
    # conda deactivate && conda activate gha2
    # cp $PROJECT_DIR/preprocess/remove_background_nersemble.py $PROJECT_DIR/ext/BackgroundMattingV2/
    # cd $PROJECT_DIR/ext/BackgroundMattingV2
    # CUDA_VISIBLE_DEVICES="$GPU" python remove_background_nersemble.py --data_path $DATA_PATH --dataset renderme


    ##############
    # Mask Calc  #
    ##############

    # # face parsing
    # cd $PROJECT_DIR/preprocess
    # conda deactivate && conda activate gha2
    # CUDA_VISIBLE_DEVICES="$GPU" python face_parse.py --model resnet34 --weight $PROJECT_DIR/ext/face-parsing/weights/resnet34.pt --input $DATA_PATH/images  --output $DATA_PATH/face-parsing


    CUDA_VISIBLE_DEVICES="$GPU" python calc_masks.py \
      --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 2048 \
      --kernel_size 5 \
      --MODNET_ckpt $PROJECT_DIR/assets/MODNet/modnet_photographic_portrait_matting.ckpt \
      --CDGNET_ckpt $PROJECT_DIR/assets/CDGNet/LIP_epoch_149.pth \
      --ext_dir $PROJECT_DIR/ext/
    # --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 512 \

    ##########################
    # Landmark + FLAME fit  #
    ##########################
    conda activate mv-3dmm-fitting
    CUDA_VISIBLE_DEVICES="$GPU" python detect_landmarks.py \
      --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks --image_size 2048

    CUDA_VISIBLE_DEVICES="$GPU" python fitting.py \
      --config $PROJECT_DIR/config/FLAME_fitting_NeRSemble_031.yaml \
      --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks \
      --param_folder $DATA_PATH/FLAME_params --camera_folder $DATA_PATH/cameras --image_size 2048

    CUDA_VISIBLE_DEVICES="$GPU" python fitting.py \
      --config $PROJECT_DIR/config/BFM_fitting_NeRSemble_031.yaml \
      --image_folder $DATA_PATH/images --landmark_folder $DATA_PATH/landmarks \
      --param_folder $DATA_PATH/params --camera_folder $DATA_PATH/cameras --image_size 2048

    ###########################
    # Orientation Maps       #
    ###########################
    conda activate gha
    CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
      --img_dir $DATA_PATH/images --mask_dir $DATA_PATH/NeuralHaircut_masks/hair --orient_dir $DATA_PATH/orientation_maps \
      --conf_dir $DATA_PATH/orientation_confidence_maps --filtered_img_dir $DATA_PATH/orientation_filtered_imgs \
      --vis_img_dir $DATA_PATH/orientation_vis_imgs

    # conda deactivate 
    # cd $PROJECT_DIR/ext/DenseMatching
    # conda activate dense_matching_env
    # python -c "from admin.environment import create_default_local_file; create_default_local_file()"
    # cp $PROJECT_DIR/preprocess/calc_optical_flow.py  $PROJECT_DIR/ext/DenseMatching/calc_optical_flow.py
    # # CUDA_VISIBLE_DEVICES="$GPU" python calc_optical_flow.py --model GLUNet_GOCor --pre_trained_model dynamic --img_dir $DATA_PATH/images --optical_flow_dir $DATA_PATH/optical_flow
    # CUDA_VISIBLE_DEVICES="$GPU" python calc_optical_flow.py --model PDCNet_plus --pre_trained_model megadepth --img_dir $DATA_PATH/images --optical_flow_dir $DATA_PATH/optical_flow

    # conda activate depth-pro 
    # cd $PROJECT_DIR/preprocess
    # CUDA_VISIBLE_DEVICES="$GPU" python calc_depth.py \
    #     --data_path $DATA_PATH --module_path $PROJECT_DIR/ext/ml-depth-pro 

    echo "Finished $SUBJECT - $SEQUENCE"
    echo "------------------------------------"
  done
done
