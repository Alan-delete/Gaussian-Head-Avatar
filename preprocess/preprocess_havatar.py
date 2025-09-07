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




def extract_frames(id_list, sequence):

    frame_step = 2

    for id in id_list:
        # camera_path = os.path.join(DATA_SOURCE, 'camera_params', id, 'camera_params.json')
        camera_folder = os.path.join(DATA_SOURCE, id, 'cameras')
        # with open(camera_path, 'r') as f:
        #     camera = json.load(f)
        camera_file_names = os.listdir(camera_folder)
        fids = {}
        camera_ids = [camera_name.split('.')[0].split('_')[1] for camera_name in camera_file_names]

        for camera_id in camera_ids:
            fids[camera_id] = 0

        video_folders = glob.glob(os.path.join(DATA_SOURCE, id, 'sequences', sequence, "*"))
        for video_folder in video_folders:
           
            if ('tongue' in video_folder) or ('GLASSES' in video_folder) or \
                ('FREE' in video_folder) or ('BACKGROUND' in video_folder) or ('SEN' in video_folder) \
                or ('EMO' in video_folder):
                continue
            
            video_paths = glob.glob(os.path.join(video_folder, 'image_*.mp4'))
            

            for idx, video_path in tqdm(enumerate(video_paths)):
                camera_id = video_path[-12:-4]
                camera_path = os.path.join(camera_folder, 'camera_' + camera_id + '.npz')
                camera = np.load(camera_path, allow_pickle=True)
                extrinsic = camera['extrinsic']
                intrinsic = camera['intrinsic']

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

                    image_lowres = cv2.resize(image, SIZE_LOWRES)
                    visible_lowres = cv2.resize(visible, SIZE_LOWRES)
                    os.makedirs(os.path.join(DATA_OUTPUT, 'images', '%04d' % fids[camera_id]), exist_ok=True)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, 'images', '%04d' % fids[camera_id], 'image_' + camera_id + '.jpg'), image)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, 'images', '%04d' % fids[camera_id], 'image_lowres_' + camera_id + '.jpg'), image_lowres)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, 'images', '%04d' % fids[camera_id], 'visible_' + camera_id + '.jpg'), visible)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, 'images', '%04d' % fids[camera_id], 'visible_lowres_' + camera_id + '.jpg'), visible_lowres)
                    os.makedirs(os.path.join(DATA_OUTPUT, 'cameras', '%04d' % fids[camera_id]), exist_ok=True)
                    
                    np.savez(os.path.join(DATA_OUTPUT, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
                    
                    fids[camera_id] += 1
                    

if __name__ == "__main__":
    # TODO: left up should be ajusted
    LEFT_UP = [0, 0]
    CROP_SIZE = [2600, 2600]
    SIZE = [2048, 2048]
    SIZE_LOWRES = [256, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='../datasets/NeRSemble')
    parser.add_argument('--data_output', type=str, default='../datasets/mini_demo_dataset/258')
    parser.add_argument('--id_list', type=str, nargs='+', default=['258'])
    parser.add_argument('--sequence', type=str, default='EXP-1-head')
    arg = parser.parse_args()

    DATA_SOURCE =  arg.data_source
    DATA_OUTPUT = arg.data_output

    extract_frames(arg.id_list, arg.sequence)