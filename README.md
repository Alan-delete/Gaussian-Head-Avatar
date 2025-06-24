# Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians
## [Paper](https://arxiv.org/abs/2312.03029) | [Project Page](https://yuelangx.github.io/gaussianheadavatar/)
<img src="imgs/teaser.jpg" width="840" height="396"/> 

## Requirements
* Create a conda environment.
```
conda env create -f environment.yaml
```
* Install [Pytorch3d](https://github.com/facebookresearch/pytorch3d).
```
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html
```
* Install [kaolin](https://github.com/NVIDIAGameWorks/kaolin).
```
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.0_cu113.html
```
* Install diff-gaussian-rasterization and simple_knn from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). Note, for rendering 32-channel images, please modify "NUM_CHANNELS 3" to "NUM_CHANNELS 32" in "diff-gaussian-rasterization/cuda_rasterizer/config.h".
```
cd path/to/gaussian-splatting
# Modify "submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h"
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
* Download ["tets_data.npz"](https://drive.google.com/file/d/1SMkp8v8bDyYxEdyq25jWnAX1zeQuAkNq/view?usp=drive_link) and put it into "assets/".


## Datasets
We provide instructions for preprocessing [NeRSemble dataset](https://tobias-kirschstein.github.io/nersemble/):
* Apply to download [NeRSemble dataset](https://tobias-kirschstein.github.io/nersemble/) and unzip it into "path/to/raw_NeRSemble/".
* Extract the images, cameras and background for specific identities into a structured dataset "NeRSemble/{id}".
```
cd preprocess
python preprocess_nersemble.py
```
* Remove background using [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2). Please git clone the code. Download [pytorch_resnet101.pth](https://drive.google.com/file/d/1zysR-jW6jydA2zkWfevxD1JpQHglKG1_/view?usp=drive_link) and put it into "path/to/BackgroundMattingV2/assets/". Then run the script we provide "preprocess/remove_background_nersemble.py".
```
cp preprocess/remove_background_nersemble.py path/to/BackgroundMattingV2/
cd path/to/BackgroundMattingV2
python remove_background_nersemble.py
```
* Fit BFM model for head pose and expression coefficients using [Multiview-3DMM-Fitting](https://github.com/YuelangX/Multiview-3DMM-Fitting). Please follow the instructions.

We provide a [mini demo dataset](https://drive.google.com/file/d/1OddIml-gJgRQU4YEP-T6USzIQyKSaF7I/view?usp=drive_link) for checking whether the code is runnable. Note, before downloading it, you must first sign the [NeRSemble Terms of Use](https://forms.gle/H4JLdUuehqkBNrBo7).


## Extra Preprocessing
We have tested two datasets, one renderme and NeRSemble.

### NeRSemble
Since we adopt GHA as the base, we use their file structure. Firstly, set the environment varibale to you corresponding directory.

``` bash
# Your work folder
PROJECT_DIR="/local/home/haonchen/Gaussian-Head-Avatar"

# Input data path, e.g where you gonna put the downloaded NeRSemble data
DATA_ROOT="$PROJECT_DIR/datasets/NeRSemble"

# Target subject and sequence
SUBJECT="258"
SEQUENCE="EXP-1-head"
DATA_PATH="$DATA_ROOT/$SUBJECT/sequences/${SEQUENCE}"

# temporary output folders, usually no need to change their paths
TRACK_OUTPUT_FOLDER="$PROJECT_DIR/datasets/output/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_wBg_staticOffset"
EXPORT_OUTPUT_FOLDER="$PROJECT_DIR/datasets/export/nersemble_v2/${SUBJECT}_${SEQUENCE}_v16_DS4_whiteBg_staticOffset_maskBelowLine"

```

Where you specify the sequence and subject of NeRSemble

#### Donwload NeRsemble dataset to your $DATA_ROOT directory 
Please follow the instruction of NeRSemble dataset
``` bash
conda activate gha2   
nersemble-data download $DATA_ROOT --participant $SUBJECT --sequence $SEQUENCE 

```

#### Preprocess

1. 
``` bash
cd  $PROJECT_DIR/preprocess
python preprocess_nersemble.py --data_source $DATA_ROOT --data_output $DATA_PATH --id_list $SUBJECT --sequence $SEQUENCE
```

2.  Now we have done the GHA preprocessing(some image editing).
After that, run VHAP for FLAME fitting and head mask


``` bash
# Run VHAP tracking 
conda activate VHAP
cd $PROJECT_DIR/ext/VHAP

CUDA_VISIBLE_DEVICES="$GPU" python vhap/preprocess_video.py \
--input ${DATA_ROOT}/${SUBJECT}/sequences/${SEQUENCE}* \
--downsample_scales 2 4 \
--matting_method background_matting_v2

# Align and track faces
CUDA_VISIBLE_DEVICES="$GPU" python vhap/track_nersemble_v2.py --data.root_folder ${DATA_ROOT} \
--exp.output_folder $TRACK_OUTPUT_FOLDER \
--data.subject $SUBJECT --data.sequence $SEQUENCE \
--model.no_use_static_offset --data.n_downsample_rgb 4  

# Export tracking results into a NeRF-style dataset
CUDA_VISIBLE_DEVICES="$GPU" python vhap/export_as_nerf_dataset.py \
--src_folder ${TRACK_OUTPUT_FOLDER} \
--tgt_folder ${EXPORT_OUTPUT_FOLDER} --background-color white

# Convert structure to images | frame_id | image_camera_id.jpg 
cd  $PROJECT_DIR/preprocess
python convert_nersemble.py --data_source ${DATA_ROOT} --intermediate_data ${EXPORT_OUTPUT_FOLDER} --data_output ${DATA_PATH}
```
3. Run hair mask detection

``` bash
# face mask from neural haircut
cd $PROJECT_DIR/preprocess
conda deactivate && conda activate matte_anything
# conda deactivate && conda activate gha2
CUDA_VISIBLE_DEVICES="$GPU" python calc_masks.py \
    --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 2048\
    --kernel_size 5 \
    --MODNET_ckpt $PROJECT_DIR/assets/MODNet/modnet_photographic_portrait_matting.ckpt \
    --CDGNET_ckpt $PROJECT_DIR/assets/CDGNet/LIP_epoch_149.pth \
    --ext_dir $PROJECT_DIR/ext/
    # --data_path $DATA_PATH --model_dir $PROJECT_DIR/ext/Matte-Anything --img_size 512 \
```

4. Run the orientation 

```bash
cd $PROJECT_DIR/preprocess 
conda deactivate && conda activate gha2
CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
    --img_dir $DATA_PATH/images --mask_dir $DATA_PATH/NeuralHaircut_masks/hair --orient_dir $DATA_PATH/orientation_maps \
    --conf_dir $DATA_PATH/orientation_confidence_maps --filtered_img_dir $DATA_PATH/orientation_filtered_imgs --vis_img_dir $DATA_PATH/orientation_vis_imgs

```

5. Run GHA FLAME fitting(optional, if you want to test GHA for head reconstruction)
``` bash
# # landmark detection and FLAME fitting
cd $PROJECT_DIR/preprocess
conda deactivate && conda activate mv-3dmm-fitting
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
```
ps: Apparently some data can be reused from previous step, but for simplicity(and less bugy), just use the original commands from GHA.



## Training
First, edit the config file, for example "config/train_meshhead_N031", and train the geometry guidance model.
```
python train_meshhead.py --config config/train_meshhead_N031.yaml
```
Second, edit the config file "config/train_gaussianhead_N031", and train the gaussian head avatar.
```
python train_gaussianhead.py --config config/train_gaussianhead_N031.yaml
```

## Reenactment
Once the two-stage training is completed, the trained avatar can be reenacted by a sequence of expression coefficients. Please specify the avatar checkpoints and the source data in the config file "config/reenactment_N031.py" and run the reenactment application.
```
python reenactment.py --config config/reenactment_N031.yaml
```


## Acknowledgement
Part of the code is borrowed from [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

## Citation
```
@inproceedings{xu2023gaussianheadavatar,
  title={Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians},
  author={Xu, Yuelang and Chen, Benwang and Li, Zhe and Zhang, Hongwen and Wang, Lizhen and Zheng, Zerong and Liu, Yebin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
