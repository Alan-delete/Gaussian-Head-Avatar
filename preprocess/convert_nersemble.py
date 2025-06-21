import os
import numpy as np
import cv2
import glob
import json
from tqdm import tqdm
import argparse


def CropImage(left_up, crop_size, image=None, K=None):
    crop_size = np.array(crop_size).astype(np.int32)
    left_up = np.array(left_up).astype(np.int32)

    if not K is None:
        K[0:2,2] = K[0:2,2] - np.array(left_up)

    if not image is None:
        if left_up[0] < 0:
            image_left = np.zeros([image.shape[0], -left_up[0], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image_left, image])
            left_up[0] = 0
        if left_up[1] < 0:
            image_up = np.zeros([-left_up[1], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image_up, image])
            left_up[1] = 0
        if crop_size[0] + left_up[0] > image.shape[1]:
            image_right = np.zeros([image.shape[0], crop_size[0] + left_up[0] - image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image, image_right])
        if crop_size[1] + left_up[1] > image.shape[0]:
            image_down = np.zeros([crop_size[1] + left_up[1] - image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image, image_down])

        image = image[left_up[1]:left_up[1]+crop_size[1], left_up[0]:left_up[0]+crop_size[0], :]

    return image, K


def ResizeImage(target_size, source_size, image=None, K=None):
    if not K is None:
        K[0,:] = (target_size[0] / source_size[0]) * K[0,:]
        K[1,:] = (target_size[1] / source_size[1]) * K[1,:]

    if not image is None:
        image = cv2.resize(image, dsize=target_size)
    return image, K




def extract_frames(id_list):

    frame_step = 5

    for id in id_list:
        # camera_path = os.path.join(DATA_SOURCE, 'camera_params', id, 'camera_params.json')
        camera_path = os.path.join(DATA_SOURCE, id, 'calibration', 'camera_params.json')
        with open(camera_path, 'r') as f:
            camera = json.load(f)

        fids = {}
        for camera_id in camera['world_2_cam'].keys():
            fids[camera_id] = 0
            # background_path = os.path.join(DATA_SOURCE, 'sequence_BACKGROUND_part-1', id, 'BACKGROUND', 'image_%s.jpg' % camera_id)
            background_path = os.path.join(DATA_SOURCE, id, 'sequences','BACKGROUND', 'image_%s.jpg' % camera_id)
            background = cv2.imread(background_path)
            background, _ = CropImage(LEFT_UP, CROP_SIZE, background, None)
            background, _ = ResizeImage(SIZE, CROP_SIZE, background, None)
            os.makedirs(os.path.join(DATA_OUTPUT, id, 'background'), exist_ok=True)
            cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'background', 'image_' + camera_id + '.jpg'), background)
        
        # video_folders = glob.glob(os.path.join(DATA_SOURCE, '*', id, '*'))
        video_folders = glob.glob(os.path.join(DATA_SOURCE, id, 'sequences', '*', "*"))
        for video_folder in video_folders:
           
            if ('tongue' in video_folder) or ('GLASSES' in video_folder) or \
                ('FREE' in video_folder) or ('BACKGROUND' in video_folder) or ('SEN' in video_folder) \
                or ('EMO' in video_folder):
                continue
            
            video_paths = glob.glob(os.path.join(video_folder, 'cam_*'))
            
            # only care about the /EXP-1-head and /HAIR
            # should've only downloaded the head and hair folders
            # video_paths = [video_path for video_path in video_paths if 'head' in video_path or 'HAIR' in video_path]
            # video_paths = [video_path for video_path in video_paths if 'head' in video_path ]
            
            for idx, video_path in tqdm(enumerate(video_paths)):
                camera_id = video_path[-13:-4]
                extrinsic = np.array(camera['world_2_cam'][camera_id][:3])
                intrinsic = np.array(camera['intrinsics'])
                _, intrinsic = CropImage(LEFT_UP, CROP_SIZE, None, intrinsic)
                _, intrinsic = ResizeImage(SIZE, CROP_SIZE, None, intrinsic)
                
                cap = cv2.VideoCapture(video_path)
                count = -1
                while(1): 
                    _, image = cap.read()
                    if image is None:
                        break
                    count += 1
                    
                    if count % frame_step != 0:
                        continue

                    visible = (np.ones_like(image) * 255).astype(np.uint8)
                    image, _ = CropImage(LEFT_UP, CROP_SIZE, image, None)
                    image, _ = ResizeImage(SIZE, CROP_SIZE, image, None)
                    visible, _ = CropImage(LEFT_UP, CROP_SIZE, visible, None)
                    visible, _ = ResizeImage(SIZE, CROP_SIZE, visible, None)
                    image_lowres = cv2.resize(image, SIZE_LOWRES)
                    visible_lowres = cv2.resize(visible, SIZE_LOWRES)
                    os.makedirs(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id]), exist_ok=True)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_' + camera_id + '.jpg'), image)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_lowres_' + camera_id + '.jpg'), image_lowres)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_' + camera_id + '.jpg'), visible)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_lowres_' + camera_id + '.jpg'), visible_lowres)
                    os.makedirs(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id]), exist_ok=True)
                    np.savez(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
                    
                    fids[camera_id] += 1
                    

# Used to move the masks and flame parameters we get from VHAP to the right place
if __name__ == "__main__":
    # TODO: left up should be ajusted
    EXPECTED_INPUT_SIZE = [2200, 3208]
    LEFT_UP = [-200, 304]
    CROP_SIZE = [2600, 2600]
    SIZE = [2048, 2048]
    SIZE_LOWRES = [256, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='../datasets/NeRSemble')
    parser.add_argument('--data_output', type=str, default='../datasets/mini_demo_dataset/100')
    # parser.add_argument('--id_list', type=str, nargs='+', default=['258'])
    parser.add_argument('--subject', type=str, default='258')
    parser.add_argument('--sequence', type=str, default='EXP-1-head')
    arg = parser.parse_args()

    transform_json_path = os.path.join(arg.data_source, 'transforms.json')
    if not os.path.exists(transform_json_path):
        print(f"Transform JSON file not found at {transform_json_path}. Please check the data source path.")
        exit(1)
    with open(transform_json_path, 'r') as f:
        transforms = json.load(f)



    src_mask_folder = os.path.join(arg.data_source, 'fg_masks')
    src_mask_paths = sorted(glob.glob(os.path.join(src_mask_folder, '*.png')))

    tar_mask_folder = os.path.join(arg.data_output, 'masks')
    os.makedirs(tar_mask_folder, exist_ok=True)
    # datasets/export/nersemble_v2/258_HAIR_v16_DS4_whiteBg_staticOffset_maskBelowLine/fg_masks/00000_00.png
    # to data_folder/masks/0000/image_00.png
    for i, src_mask_path in tqdm(enumerate(src_mask_paths)):
        frame_info = transforms['frames'][i]
        camera_id = frame_info['camera_id']
        frame_id = frame_info['timestep_id']
        # mask_name = os.path.basename(src_mask_path).replace('.png', '')
        # frame_id = mask_name.split('_')[0]
        # camera_id = mask_name.split('_')[1]
        # frame_id should be at most 4 digits
        frame_id = '%04d' % int(frame_id)
        tar_mask_path = os.path.join(tar_mask_folder, frame_id, 'image_' + camera_id + '.jpg')
        os.makedirs(os.path.dirname(tar_mask_path), exist_ok=True)
        
        mask = cv2.imread(src_mask_path)
        mask = cv2.resize(mask, EXPECTED_INPUT_SIZE)
        
        mask, _ = CropImage(LEFT_UP, CROP_SIZE, mask, None)
        mask, _ = ResizeImage(SIZE, CROP_SIZE, mask, None)
        
        mask = mask[..., :1]  # keep only one channel
        cv2.imwrite(tar_mask_path, mask)


    scr_flame_folder = os.path.join(arg.data_source, 'flame_param')
    src_flame_paths = glob.glob(os.path.join(scr_flame_folder, '*.npz'))
    tar_flame_folder = os.path.join(arg.data_output, 'FLAME_params')
    os.makedirs(tar_flame_folder, exist_ok=True)
    for src_flame_path in tqdm(src_flame_paths):
        flame_name = os.path.basename(src_flame_path).replace('.npz', '')
        # frame_id should be at most 4 digits
        frame_id = '%04d' % int(flame_name)
        tar_flame_path = os.path.join(tar_flame_folder, frame_id, 'params.npz')
        os.makedirs(os.path.dirname(tar_flame_path), exist_ok=True)
        flame_param = np.load(src_flame_path)
        np.savez(tar_flame_path, **flame_param)

     

    DATA_SOURCE =  arg.data_source
    DATA_OUTPUT = arg.data_output

    # # DATA_OUTPUT = '../NeRSemble'
    # # extract_frames(['031', '036'])
    # extract_frames(arg.id_list)