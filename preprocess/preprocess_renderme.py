import os
import numpy as np
import cv2
import glob
import json
import time
import h5py
import torch
import tqdm
import sys
import argparse

from calendar import c
from functools import partial
import json
from unittest.mock import NonCallableMagicMock
from pydub import AudioSegment

class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None
        self.actor_id = self.smc.attrs['actor_id']
        self.performance_part = self.smc.attrs['performance_part']
        self.capture_date = self.smc.attrs['capture_date']
        self.actor_info = dict(
            age=self.smc.attrs['age'],
            color=self.smc.attrs['color'], 
            gender=self.smc.attrs['gender'],
            height=self.smc.attrs['height'], 
            weight=self.smc.attrs['weight'] 
            )
        self.Camera_info = dict(
            num_device=self.smc['Camera'].attrs['num_device'],
            num_frame=self.smc['Camera'].attrs['num_frame'],
            resolution=self.smc['Camera'].attrs['resolution'],
        )


    ###info 
    def get_actor_info(self):
        return self.actor_info
    
    def get_Camera_info(self):
        return self.Camera_info

    
    ### Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_id : Matrix_type : value
              )
            Notice:
                Camera_id(str) in {'00' ... '59'}
                Matrix_type in ['D', 'K', 'RT'] 
        """  
        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__
        self.__calibration_dict__ = dict()
        for ci in self.smc['Calibration'].keys():
            self.__calibration_dict__.setdefault(ci,dict())
            for mt in ['D', 'K', 'RT'] :
                self.__calibration_dict__[ci][mt] = \
                    self.smc['Calibration'][ci][mt][()]
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id 

        Args:
            Camera_id (int/str of a number):
                CameraID(str) in {'00' ... '60'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT'] 
        """            
        Camera_id = str(Camera_id).zfill(2)
        assert Camera_id in self.smc['Calibration'].keys(), f'Invalid Camera_id {Camera_id}'
        rs = dict()
        for k in ['D', 'K', 'RT'] :
            rs[k] = self.smc['Calibration'][Camera_id][k][()]
        return rs

    ### RGB image
    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    def get_img(self, Camera_id, Image_type, Frame_id=None, disable_tqdm=True):
        """Get image its Camera_id, Image_type and Frame_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'00'...'59'}
            Image_type(str) in 
                    {'Camera': ['color','mask']}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'-1 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
                'color': HWC(2048, 2448, 3) in bgr (uint8)
                'mask' : HW (2048, 2448) (uint8)
            multiple imgs :
                'color': NHWC(N, 2048, 2448, 3) in bgr (uint8)
                'mask' : NHW (N, 2048, 2448) (uint8)
        """ 
        Camera_id = str(Camera_id).zfill(2)
        assert Camera_id in self.smc["Camera"].keys(), f'Invalid Camera_id {Camera_id}'
        assert Image_type in self.smc["Camera"][Camera_id].keys(), f'Invalid Image_type {Image_type}'
        assert isinstance(Frame_id,(list,int, str, type(None))), f'Invalid Frame_id datatype {type(Frame_id)}'
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert Frame_id in self.smc["Camera"][Camera_id][Image_type].keys(), f'Invalid Frame_id {Frame_id}'
            if Image_type in ['color','mask']:
                img_byte = self.smc["Camera"][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
            if Image_type == 'mask':
                img_color = np.max(img_color,2).astype(np.uint8)
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc["Camera"][Camera_id][Image_type].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_img(Camera_id, Image_type,fi))
            return np.stack(rs,axis=0)
    
    def get_audio(self):
        """
        Get audio data.
        Returns:
            a dictionary of audio data consists of:
                audio_np_array: np.ndarray
                sample_rate: int
        """
        if "s" not in self.performance_part.split('_')[0]:
            print(f"no audio data in the performance part: {self.performance_part}")
            return None
        data = self.smc["Camera"]['00']['audio']
        return data
    
    def writemp3(self, f, sr, x, normalized=False):
        """numpy array to MP3"""
        channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(x * 2 ** 15)
        else:
            y = np.int16(x)
        song = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
        song.export(f, format="mp3", bitrate="320k")

    ###Keypoints2d
    def get_Keypoints2d(self, Camera_id,Frame_id=None):
        """Get keypoint2D by its Camera_group, Camera_id and Frame_id
        PS: Not all the Camera_id/Frame_id have detected keypoints2d.

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in {18...32}
                    Not all the view have detection result, so the key will miss too when there are no lmk2d result
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            single lmk2d : (106, 2)
            multiple lmk2d : (N, 106, 2)
            if no data,return None
        """ 
        Camera_id = str(Camera_id)
        assert Camera_id in [f'%02d'%i for i in range(18,33)], f'Invalid Camera_id {Camera_id}'
        assert isinstance(Frame_id,(list,int, str, type(None))), f'Invalid Frame_id datatype: {type(Frame_id)}'
        if Camera_id not in self.smc['Keypoints2d'].keys():
            print(f"not lmk2d result in camera id {Camera_id}")
            return None
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            assert Frame_id >= 0 and Frame_id<self.smc['Keypoints2d'].attrs['num_frame'], f'Invalid frame_index {Frame_id}'
            Frame_id = str(Frame_id)
            if Frame_id not in self.smc['Keypoints2d'][Camera_id].keys() or \
                self.smc['Keypoints2d'][Camera_id][Frame_id] is None or \
                len(self.smc['Keypoints2d'][Camera_id][Frame_id]) == 0:
                print(f"not lmk2d result in Camera_id/Frame_id {Camera_id}/{Frame_id}")
                return None
            return self.smc['Keypoints2d'][Camera_id][Frame_id]
        else:
            if Frame_id is None:
                return self.smc['Keypoints2d'][Camera_id]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                kpt2d = self.get_Keypoints2d(Camera_id,fi)
                if kpt2d is not None:
                    rs.append(kpt2d)
            return np.stack(rs,axis=0)

    ###Keypoints3d
    def get_Keypoints3d(self, Frame_id=None):
        """Get keypoint3D Frame_id
        PS: Not all the Frame_id have keypoints3d.

        Args:
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            Keypoints3d tensor: np.ndarray of shape ([N], ,3)
            if data do not exist: None
        """ 
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            assert Frame_id>=0 and Frame_id<self.smc['Keypoints3d'].attrs['num_frame'], \
                f'Invalid frame_index {Frame_id}'
            if str(Frame_id) not in self.smc['Keypoints3d'].keys() or \
                len(self.smc['Keypoints3d'][str(Frame_id)]) == 0:
                print(f"get_Keypoints3d: data of frame {Frame_id} do not exist.")
                return None
            return self.smc['Keypoints3d'][str(Frame_id)]
        else:
            if Frame_id is None:
                return self.smc['Keypoints3d']
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                kpt3d = self.get_Keypoints3d(fi)
                if kpt3d is not None:
                    rs.append(kpt3d)
            return np.stack(rs,axis=0)

    ###FLAME
    def get_FLAME(self, Frame_id=None):
        """Get FLAME (world coordinate) computed by flame-fitting processing pipeline.
        FLAME is only provided in expression part.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            dict:
                "global_pose"                   : double (3,)
                "neck_pose"                     : double (3,)
                "jaw_pose"                      : double (3,)
                "left_eye_pose"                 : double (3,)
                "right_eye_pose"                : double (3,)
                "trans"                         : double (3,)
                "shape"                         : double (100,)
                "exp"                           : double (50,)
                "verts"                         : double (5023,3)
                "albedos"                       : double (3,256,256)
        """
        if "e" not in self.performance_part.split('_')[0]:
            print(f"no flame data in the performance part: {self.performance_part}")
            return None
        if "FLAME" not in self.smc.keys():
            print("not flame parameters, please check the performance part.")
            return None
        flame = self.smc['FLAME']
        if Frame_id is None:
            return flame
        elif isinstance(Frame_id, list):
            frame_list = [str(fi) for fi in Frame_id]
            rs = []
            for fi in  tqdm.tqdm(frame_list):
                rs.append(self.get_FLAME(fi))
            return np.stack(rs,axis=0)
        elif isinstance(Frame_id, (int,str)):
            Frame_id = int(Frame_id)
            assert Frame_id>=0 and Frame_id<self.smc['FLAME'].attrs['num_frame'], f'Invalid frame_index {Frame_id}'
            return flame[str(Frame_id)]      
        else:
            raise TypeError('frame_id should be int, list or None.')
    
    ###uv texture map
    def get_uv(self, Frame_id=None, disable_tqdm=True):
        """Get uv map (image form) computed by flame-fitting processing pipeline.
        uv texture is only provided in expression part.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            a single img: HWC in bgr (uint8)
        """
        if "e" not in self.performance_part.split('_')[0]:
            print(f"no uv data in the performance part: {self.performance_part}")
            return None
        if "UV_texture" not in self.smc.keys():
            print("not uv texture, please check the performance part.")
            return None
        assert isinstance(Frame_id,(list,int, str, type(None))), f'Invalid Frame_id datatype {type(Frame_id)}'
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert Frame_id in self.smc['UV_texture'].keys(), f'Invalid Frame_id {Frame_id}'
            img_byte = self.smc['UV_texture'][Frame_id][()]
            img_color = self.__read_color_from_bytes__(img_byte)
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc['UV_texture'].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_uv(fi))
            return np.stack(rs,axis=0)
    
    ###scan mesh
    def get_scanmesh(self):
        """
        Get scan mesh data computed by Dense Mesh Reconstruction pipeline.
        Returns:
            dict:
                'vertex': np.ndarray of vertics point (n, 3)
                'vertex_indices': np.ndarray of vertex indices (m, 3)
        """
        if "e" not in self.performance_part.split('_')[0]:
            print(f"no scan mesh data in the performance part: {self.performance_part}")
            return None
        data = self.smc["Scan"]
        return data
    
    def write_ply(self, scan, outpath):
        from plyfile import PlyData, PlyElement
        vertex = np.empty(len(scan['vertex']), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(len(scan['vertex'])):
            vertex[i] = np.array([(scan['vertex'][i,0], scan['vertex'][i,1], scan['vertex'][i,2])], \
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        triangles = scan['vertex_indices']
        face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,)),
                           ('red', 'u1'), ('green', 'u1'),
                           ('blue', 'u1')])
        for i in range(len(triangles)):
            face[i] = np.array([
                ([triangles[i,0],triangles[i,1],triangles[i,2]], 255, 255, 255)
            ],
            dtype=[('vertex_indices', 'i4', (3,)),
                ('red', 'u1'), ('green', 'u1'),
                ('blue', 'u1')])
        PlyData([
                PlyElement.describe(vertex, 'vertex'),
                PlyElement.describe(face, 'face') 
                ], text=True).write(outpath)

    def get_scanmask(self, Camera_id=None):
        """Get image its Camera_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'00'...'59'}
        Returns:
            a single img : HW (2048, 2448) (uint8)
            multiple img: NHW (N, 2048, 2448)  (uint8)
        """ 
        if Camera_id is None:
            rs = []
            for i in range(60):
                rs.append(self.get_scanmask(f'{i:02d}'))
            return np.stack(rs, axis=0)
        assert isinstance(Camera_id, (str,int)), f'Invalid Camera_id type {Camera_id}'
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc["Camera"].keys(), f'Invalid Camera_id {Camera_id}'
        img_byte = self.smc["ScanMask"][Camera_id][()]
        img_color = self.__read_color_from_bytes__(img_byte)
        img_color = np.max(img_color,2).astype(np.uint8)
        return img_color           

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


from plyfile import PlyData, PlyElement
def main(smc_path, output_dir):
    smc_dir = os.path.dirname(os.path.abspath(smc_path))
    actor_part = smc_path.split('/')[-1].split('.')[0]
    st = time.time()
    print("reading smc: {}".format(actor_part))
    rd = SMCReader(smc_path)
    
    anno_path = os.path.join(smc_dir, actor_part.replace('raw', 'anno') + '.smc')
    if os.path.exists(anno_path):
        print("annotation exists, reading anno smc: {}".format(actor_part))
        anno_rd = SMCReader(anno_path)
    else:
        print("annotation path {} not exists, reading none anno smc".format(anno_path))
        anno_rd = None

    # rd = SMCReader(f'/mnt/lustre/share_data/pandongwei/RenFace_waic_20230718/{actor_part}.smc')
    print("SMCReader done, in %f sec\n" % ( time.time() - st ), flush=True)
    
    # basic info
    print(rd.get_actor_info())
    print(rd.get_Camera_info())
    
    video_info = rd.get_Camera_info()
    num_frame = video_info['num_frame']
    num_camera = video_info['num_device']
    # [2048, 2448]
    resolution = video_info['resolution']

    cameras = rd.get_Calibration_all()
    camera_pos = []
    for id, camera in cameras.items():
        print(f"Camera {id}: {camera.keys()}")
        position = camera['RT'][:, 3]
        camera_pos.append(position)
    # save position to ply
    vertex = np.empty(len(camera_pos), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i in range(len(camera_pos)):
        vertex[i] = np.array([(camera_pos[i][0], camera_pos[i][1], camera_pos[i][2])], \
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    PlyData([
            PlyElement.describe(vertex, 'vertex')
            ], text=True).write('camera_pos.ply')


    # img
    Camera_id = "25"
    Frame_id = 0
    if False and anno_rd is not None:
        image = rd.get_img(Camera_id, 'color', Frame_id)          # Load image for the specified camera and frame
        print(f"image.shape: {image.shape}") #(2048, 2448, 3)

        camera = rd.get_Calibration(Camera_id)
        extrinsic = camera['RT']
        R = extrinsic[:3, :3].T
        t = - R @ extrinsic[:3, 3]
        extrinsic = np.concatenate([R, t[:, None]], axis=1)
        intrinsic = np.array(camera['K'])

        # landmark
        lmk2d = anno_rd.get_Keypoints2d('25',Frame_id)
        print('kepoint2d',lmk2d.shape)
        lmk2ds = anno_rd.get_Keypoints2d('26', [1,2,3,4,5])
        print(f"lmk2ds.shape: {lmk2ds.shape}")
        lmk3d = anno_rd.get_Keypoints3d(4)
        print(f'kepoint3d shape: {lmk3d.shape}')
        lmk3ds = anno_rd.get_Keypoints3d([1,2,3,4,5])
        print(f'kepoint3d shape: {lmk3d.shape}')
        
        # draw landmark on imahe
        for i in range(lmk2d.shape[0]):
            image1 = cv2.circle(image, (int(lmk2d[i,0]), int(lmk2d[i,1])), 5, (0, 255, 0), -1)
        cv2.imwrite("rendermeTest.png", image1)

        # # save landmark3d
        # np.savez('lmk3d.npz', lmk3d=lmk3d)

        # project lmk3d to image
        N = lmk3d.shape[0]
        ones = np.ones((N, 1))
        points_h = np.concatenate([lmk3d, ones], axis=1)
        cam_coords = extrinsic @ points_h.T
        pixel_coords = intrinsic @ cam_coords
        U = pixel_coords[0, :] / pixel_coords[2, :]
        V = pixel_coords[1, :] / pixel_coords[2, :]
        for i in range(N):
            u_ = int(U[i].item())
            v_ = int(V[i].item())
            image2 = cv2.circle(image, (u_, v_), 5, (0, 0, 255), -1)
        cv2.imwrite("rendermeTest_project.png", image2)

        # save landmark to ply
        vertex = np.empty(len(lmk3d), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(len(lmk3d)):
            vertex[i] = np.array([(lmk3d[i,0], lmk3d[i,1], lmk3d[i,2])], \
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        PlyData([
                PlyElement.describe(vertex, 'vertex')
                ], text=True).write('lmk3d.ply')        


        # save the camera coordinates
        np.savez('cam_coords.npz', cam_coords=cam_coords.T)
        # save camera coordinates ply
        cam_coords = cam_coords.T
        cam_coords = cam_coords[:, :3]
        vertex = np.empty(len(cam_coords), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(len(cam_coords)):
            vertex[i] = np.array([(cam_coords[i,0], cam_coords[i,1], cam_coords[i,2])], \
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        PlyData([
                PlyElement.describe(vertex, 'vertex')
                ], text=True).write('cam_coords.ply')
        
        breakpoint()

        
        # flame
        if '_e' in actor_part:
            flame = anno_rd.get_FLAME(56)
            print(f"keys: {flame.keys()}")
            print(f"global_pose: {flame['global_pose'].shape}")
            print(f"neck_pose: {flame['neck_pose'].shape}")
            print(f"jaw_pose: {flame['jaw_pose'].shape}")
            print(f"left_eye_pose: {flame['left_eye_pose'].shape}")
            print(f"right_eye_pose: {flame['right_eye_pose'].shape}")
            print(f"trans: {flame['trans'].shape}")
            print(f"shape: {flame['shape'].shape}")
            print(f"exp: {flame['exp'].shape}")
            print(f"verts: {flame['verts'].shape}")
            print(f"albedos: {flame['albedos'].shape}")
            flame = anno_rd.get_FLAME()
            print(f"keys: {flame.keys()}")
            
        # uv texture
        if '_e' in actor_part:
            uv = anno_rd.get_uv(Frame_id)
            print(f"uv shape: {uv.shape}")
            uv = anno_rd.get_uv()
            print(f"uv shape: {uv.shape}")
            
        # scan mesh
        if '_e' in actor_part:
            scan = anno_rd.get_scanmesh()
            print(f"keys: {scan.keys()}")
            print(f"vertex: {scan['vertex'].shape}")
            print(f"vertex_indices: {scan['vertex_indices'].shape}")
            anno_rd.write_ply(scan, './test_scan.ply')
            
        # scan mask
        if '_e' in actor_part:
            scanmask = anno_rd.get_scanmask('03')
            print(f"scanmask.shape: {scanmask.shape}")
            scanmask = anno_rd.get_scanmask()
            print(f"scanmask.shape all: {scanmask.shape}")    

    frame_step = 3
    selected_frames = range(0, num_frame, frame_step)
    
    for i in selected_frames:
        os.makedirs(os.path.join(output_dir, actor_part, 'images', f'{i:04d}'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, actor_part, 'cameras', f'{i:04d}'), exist_ok=True)

    
    for camera_id in selected_frames:
        images = rd.get_img(camera_id, 'color', disable_tqdm=False)
        for image, frame_id in zip(images, range(num_frame)):
            visible = (np.ones_like(image) * 255).astype(np.uint8)
            image, _ = CropImage(LEFT_UP, CROP_SIZE, image, None)
            image, _ = ResizeImage(SIZE, CROP_SIZE, image, None)
            visible, _ = CropImage(LEFT_UP, CROP_SIZE, visible, None)
            visible, _ = ResizeImage(SIZE, CROP_SIZE, visible, None)
            image_lowres = cv2.resize(image, SIZE_LOWRES)
            visible_lowres = cv2.resize(visible, SIZE_LOWRES)
            cv2.imwrite(os.path.join(output_dir, actor_part, 'images', f'{frame_id:04d}', f'image_{camera_id:02d}.jpg'), image)
            cv2.imwrite(os.path.join(output_dir, actor_part, 'images', f'{frame_id:04d}', f'visible_{camera_id:02d}.jpg'), visible)
            cv2.imwrite(os.path.join(output_dir, actor_part, 'images', f'{frame_id:04d}', f'image_lowres_{camera_id:02d}.jpg'), image_lowres)
            cv2.imwrite(os.path.join(output_dir, actor_part, 'images', f'{frame_id:04d}', f'visible_lowres_{camera_id:02d}.jpg'), visible_lowres)
            if anno_rd is not None:
                camera = anno_rd.get_Calibration(camera_id)
            else:
                camera = rd.get_Calibration(camera_id)
            extrinsic = camera['RT']
            R = extrinsic[:3, :3].T
            t = - R @ extrinsic[:3, 3]
            extrinsic = np.concatenate([R, t[:, None]], axis=1)
            intrinsic = np.array(camera['K'])
            _, intrinsic = CropImage(LEFT_UP, CROP_SIZE, None, intrinsic)
            _, intrinsic = ResizeImage(SIZE, CROP_SIZE, None, intrinsic)
            np.savez(os.path.join(output_dir, actor_part, 'cameras', f'{frame_id:04d}', f'camera_{camera_id:02d}.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
    
    return 

    # img
    Camera_id = "25"
    Frame_id = 0

    breakpoint()
    image = rd.get_img(Camera_id, 'color', Frame_id)          # Load image for the specified camera and frame
    print(f"image.shape: {image.shape}") #(2048, 2448, 3)
    images = rd.get_img('04','color',disable_tqdm=False) # (N, 2048, 2448, 3)
    print(f'color {images.shape}, {images.dtype}')
    
    # mask
    mask = rd.get_img(Camera_id, 'mask', Frame_id)
    print(f"mask.shape: {mask.shape}") # (2048, 2448)
    l = [10, 13]
    mask = rd.get_img(13,'mask', l, disable_tqdm=False)
    mask = rd.get_img(13,'mask',disable_tqdm=False)
    print(f' mask {mask.dtype} {mask.shape}')

    # camera
    cameras = rd.get_Calibration_all()
    print(f"all_calib 30 RT: {cameras['30']['RT']}")
    camera = rd.get_Calibration(15)
    print(' split_calib ',camera)

    # from  ['D', 'K', 'RT'] to extrinsic matrix and intrinsic matrix
    # 5
    D = camera['D']
    # [3,3]
    K = camera['K']
    # [4,4]
    RT = camera['RT']
    print(f"D: {D.shape}, K: {K.shape}, RT: {RT.shape}")
    extrinsic = RT
    print(f"extrinsic: {extrinsic.shape}")
    intrinsic = K

    # [3,4]
    extrinsic = np.array(extrinsic[:3])
    # [3,3]
    intrinsic = np.array(intrinsic)
    # np.savez(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)

    # audio
    if '_s' in actor_part:
        audio = rd.get_audio()
        print('audio', audio['audio'].shape, 'sample_rate', np.array(audio['sample_rate']))
        sr = int(np.array(audio['sample_rate'])); arr = np.array(audio['audio'])
        rd.writemp3(f='./test.mp3', sr=sr, x=arr, normalized=True)
    


### test func
if __name__ == '__main__':
    # cut the left part and right part of the image to make it 2048x2048
    LEFT_UP = [(2448 - 2048)//2, 0]
    CROP_SIZE = [2048, 2048]
    SIZE = [2048, 2048]
    SIZE_LOWRES = [256, 256]

    # actor_part = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--smc_path', type=str, default='../datasets/RenderMe/raw/0094_h1_6bn_raw-007.smc')
    parser.add_argument('--output_dir', type=str, default='../datasets/RenderMe/')
    arg = parser.parse_args()

    os.makedirs(arg.output_dir, exist_ok=True)

    main(arg.smc_path, arg.output_dir)
