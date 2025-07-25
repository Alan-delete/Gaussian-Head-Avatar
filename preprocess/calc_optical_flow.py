import os
import cv2
import numpy as np
import glob
import torch
import argparse

import tqdm

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import imageio

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame=0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        opacity: float = 1.0,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            color_alpha=color_alpha,
        )
        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

            # Prepare the video file path
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list[2:-1]:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                y_min, y_max = (
                    tracks[query_frame, :, 1].min(),
                    tracks[query_frame, :, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if isinstance(query_frame, torch.Tensor):
                        query_frame_ = query_frame[n]
                    else:
                        query_frame_ = query_frame
                    color = self.color_map(norm(tracks[query_frame_, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (
                    tracks[0, segm_mask > 0, 1].min(),
                    tracks[0, segm_mask > 0, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(self.linewidth * 2),
                            color=vector_colors[t, i].astype(int),
                            visible=visibile,
                            color_alpha=color_alpha,
                        )
            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self.linewidth,
                    )
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb), alpha, np.array(original), 1 - alpha, 0
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211, 0, 0))
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
        rgb = np.array(rgb)
        return rgb



import os
import torch
import argparse
import imageio
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask
from validation.test_parser import define_model_parser


def pad_to_same_shape(im1, im2):
    # pad to same shape both images with zero
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)

    return im1, im2


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    define_model_parser(parser) 
    parser.add_argument('--img_dir',default='../datasets/mini_demo_dataset/031/images', type=str)
    parser.add_argument('--optical_flow_dir', default='../datasets/mini_demo_dataset/031/optical_flow', type=str)
    parser.add_argument('--model', default='GLUNet_GOCor', type=str)
    parser.add_argument('--pre_trained_model', default='dynamic', type=str)
    parser.add_argument('--batch_size', default=2, type=int)


    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    
    local_optim_iter = args.optim_iter if not args.local_optim_iter else int(args.local_optim_iter)


    os.makedirs(args.optical_flow_dir, exist_ok=True)
    frames_names = os.listdir(args.img_dir) 
    for frame in frames_names:
        mask_folder = os.path.join(args.optical_flow_dir , frame)
        os.makedirs(mask_folder, exist_ok=True)

    img_list = glob.glob(os.path.join(args.img_dir, '*', 'image_[0-9]*' ))
    # get the sequence of images of the camera
    camera_set = sorted(list(set([os.path.basename(img_path) for img_path in img_list])))
    camera_sequence = []
    for camera in camera_set:
        single_camera_sequence = [img_path for img_path in img_list if camera in img_path]
        single_camera_sequence.sort()
        camera_sequence.append(single_camera_sequence)

    # device = 'cuda'
    # grid_size = 50
    # # Run Offline CoTracker:
    # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    batch_size = args.batch_size
    with torch.no_grad():
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

    
    for camera_id, camera in enumerate(camera_sequence):

        warped_images = []
        init_frame = cv2.imread(camera[0])
        mask_init_frame = cv2.imread(camera[0].replace('image_', 'mask_')) / 255.
        mask_init_frame = mask_init_frame.round().astype(np.uint8)
        warped_image = init_frame * mask_init_frame
        warped_images.append(warped_image)

        # for idx in tqdm.tqdm(range(0, len(camera) - batch_size, batch_size)):
        #     batch_camera = camera[idx:idx + batch_size]
        #     batch_mask = [cv2.imread(img.replace('image_', 'mask_')) / 255. for img in batch_camera]
        #     batch_mask = [mask.round().astype(np.uint8) for mask in batch_mask]
        #     batch_frame_0 = [cv2.imread(img) for img in batch_camera]
        #     batch_frame_0 = [frame * mask for frame, mask in zip(batch_frame_0, batch_mask)]

        #     batch_camera = camera[idx + 1:idx + 1 + batch_size]
        #     batch_mask = [cv2.imread(img.replace('image_', 'mask_')) / 255. for img in batch_camera]
        #     batch_mask = [mask.round().astype(np.uint8) for mask in batch_mask]
        #     batch_frame_1 = [cv2.imread(img) for img in batch_camera]
        #     batch_frame_1 = [frame * mask for frame, mask in zip(batch_frame_1, batch_mask)]

        #     query_image = np.stack(batch_frame_0)
        #     reference_image = np.stack(batch_frame_1)

        #     with torch.no_grad():
        #         # save original ref image shape
        #         ref_image_shape = reference_image.shape[1:3]

        #         # pad both images to the same size, to be processed by network
        #         # query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
        #         query_image_ = query_image
        #         reference_image_ = reference_image

        #         # convert numpy to torch tensor and put it in right format
        #         query_image_ = torch.from_numpy(query_image_).permute(0, 3, 1, 2)
        #         reference_image_ = torch.from_numpy(reference_image_).permute(0, 3, 1, 2)

        #         # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
        #         # specific pre-processing (/255 and rescaling) are done within the function.

        #         # pass both images to the network, it will pre-process the images and ouput the estimated flow
        #         # in dimension 1x2xHxW
        #         if estimate_uncertainty:
        #             if args.flipping_condition:
        #                 raise NotImplementedError('No flipping condition with PDC-Net for now')

        #             estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
        #                                                                                             reference_image_,
        #                                                                                             mode='channel_first')
        #             confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
        #             confidence_map = confidence_map[:, :ref_image_shape[0], :ref_image_shape[1]]
        #         else:
        #             if args.flipping_condition and 'GLUNet' in args.model:
        #                 estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
        #                                                                             mode='channel_first')
        #             else:
        #                 estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')


        #         for i in range(batch_size):
        #             estimated_flow_numpy = estimated_flow[i].permute(1, 2, 0).cpu().numpy()
        #             estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
        #             # removes the padding

        #             warped_query_image = remap_using_flow_fields(query_image[i], estimated_flow_numpy[:, :, 0],
        #                                                         estimated_flow_numpy[:, :, 1]).astype(np.uint8)

        #             warped_image = remap_using_flow_fields(warped_image, estimated_flow_numpy[:, :, 0],
        #                                                     estimated_flow_numpy[:, :, 1]).astype(np.uint8)
        #             warped_images.append(warped_image)
                    

        #             cur_query_image =  cv2.cvtColor(query_image[i], cv2.COLOR_BGR2RGB)
        #             cur_reference_image = cv2.cvtColor(reference_image[i], cv2.COLOR_BGR2RGB)
        #             warped_query_image = cv2.cvtColor(warped_query_image, cv2.COLOR_BGR2RGB)

        #             # if estimate_uncertainty:
        #             #     color = [255, 102, 51]
        #             #     fig, axis = plt.subplots(1, 5, figsize=(30, 30))

        #             #     confident_mask = (confidence_map[i] > 0.50).astype(np.uint8)
        #             #     confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask*255, color=color)
        #             #     axis[2].imshow(confident_warped)
        #             #     axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
        #             #                     .format(args.model, args.pre_trained_model))
        #             #     axis[4].imshow(confidence_map[i], vmin=0.0, vmax=1.0)
        #             #     axis[4].set_title('Confident regions')
        #             # else:
        #             #     fig, axis = plt.subplots(1, 4, figsize=(30, 30))
        #             #     axis[2].imshow(warped_query_image)
        #             #     axis[2].set_title(
        #             #         'Warped query image according to estimated flow by {}_{}'.format(args.model, args.pre_trained_model))
        #             # axis[0].imshow(cur_query_image)
        #             # axis[0].set_title('Query image')
        #             # axis[1].imshow(cur_reference_image)
        #             # axis[1].set_title('Reference image')

        #             # axis[3].imshow(flow_to_image(estimated_flow_numpy))
        #             # axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))
        #             # fig.savefig(
        #             #     camera[idx+i].replace('images', 'optical_flow').replace('.jpg', '.png'),
        #             #     bbox_inches='tight')
        #             # plt.close(fig)


        #             # save the optical flow
        #             flow = estimated_flow[i].detach().cpu().numpy()
        #             np.save(camera[idx + i].replace('images', 'optical_flow').replace('.jpg', '.npy'), flow)

        #             # save the confidence map
        #             if estimate_uncertainty:
        #                 # confidence_map = (confidence_map * 255).astype(np.uint8)
        #                 np.save(camera[idx + i].replace('images', 'optical_flow').replace('.jpg', '_confidence_map.npy'), confidence_map[i])
            

        # for idx in tqdm.tqdm(range(len(warped_images)- 1, len(camera) - 1)):
        for idx in tqdm.tqdm(range(len(camera) - 1)):
            frame_0 = cv2.imread(camera[idx])
            mask_0 = cv2.imread(camera[idx].replace('image_', 'mask_')) / 255.
            mask_0 = mask_0.round().astype(np.uint8)
            frame_1 = cv2.imread(camera[idx + 1])
            mask_1 = cv2.imread(camera[idx + 1].replace('image_', 'mask_')) / 255.
            mask_1 = mask_1.round().astype(np.uint8)
            frame_0 = frame_0 * mask_0
            frame_1 = frame_1 * mask_1
            # video = torch.from_numpy(np.stack([frame_0, frame_1])).to(device).permute(0, 3, 1, 2)[None].float()
            # pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)

            query_image = frame_0
            reference_image = frame_1
            with torch.no_grad():
                # network, estimate_uncertainty = select_model(
                #     args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
                #     path_to_pre_trained_models=args.path_to_pre_trained_models)

                # save original ref image shape
                ref_image_shape = reference_image.shape[:2]

                # pad both images to the same size, to be processed by network
                query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
                # convert numpy to torch tensor and put it in right format
                query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
                reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

                # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
                # specific pre-processing (/255 and rescaling) are done within the function.

                # pass both images to the network, it will pre-process the images and ouput the estimated flow
                # in dimension 1x2xHxW
                if estimate_uncertainty:
                    if args.flipping_condition:
                        raise NotImplementedError('No flipping condition with PDC-Net for now')

                    estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
                                                                                                    reference_image_,
                                                                                                    mode='channel_first')
                    confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
                    confidence_map = confidence_map[:ref_image_shape[0], :ref_image_shape[1]]
                else:
                    if args.flipping_condition and 'GLUNet' in args.model:
                        estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
                                                                                    mode='channel_first')
                    else:
                        estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
                estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
                estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
                # removes the padding

                warped_query_image = remap_using_flow_fields(query_image, estimated_flow_numpy[:, :, 0],
                                                            estimated_flow_numpy[:, :, 1]).astype(np.uint8)
                
                warped_image = remap_using_flow_fields(warped_image, estimated_flow_numpy[:, :, 0],
                                                        estimated_flow_numpy[:, :, 1]).astype(np.uint8)
                warped_images.append(warped_image)

                flow = estimated_flow.detach().cpu().numpy()
                np.save(camera[idx].replace('images', 'optical_flow').replace('.jpg', '.npy'), flow)

                # save the confidence map
                if estimate_uncertainty:
                    # confidence_map = (confidence_map * 255).astype(np.uint8)
                    np.save(camera[idx].replace('images', 'optical_flow').replace('.jpg', '_confidence_map.npy'), confidence_map)


                query_image =  cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
                reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                warped_query_image = cv2.cvtColor(warped_query_image, cv2.COLOR_BGR2RGB)

                # if estimate_uncertainty:
                #     color = [255, 102, 51]
                #     fig, axis = plt.subplots(1, 5, figsize=(30, 30))

                #     confident_mask = (confidence_map > 0.50).astype(np.uint8)
                #     confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask*255, color=color)
                #     axis[2].imshow(confident_warped)
                #     axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
                #                     .format(args.model, args.pre_trained_model))
                #     axis[4].imshow(confidence_map, vmin=0.0, vmax=1.0)
                #     axis[4].set_title('Confident regions')
                # else:
                #     fig, axis = plt.subplots(1, 4, figsize=(30, 30))
                #     axis[2].imshow(warped_query_image)
                #     axis[2].set_title(
                #         'Warped query image according to estimated flow by {}_{}'.format(args.model, args.pre_trained_model))
                # axis[0].imshow(query_image)
                # axis[0].set_title('Query image')
                # axis[1].imshow(reference_image)
                # axis[1].set_title('Reference image')

                # axis[3].imshow(flow_to_image(estimated_flow_numpy))
                # axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))
                # fig.savefig(
                #     camera[idx].replace('images', 'optical_flow').replace('.jpg', '.png'),
                #     bbox_inches='tight')
                # plt.close(fig)
                # print('Saved image!')


        # save ground truth video
        output_path = os.path.join(args.optical_flow_dir, 'wrapped_video2_{}.mp4'.format(camera_id))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (warped_images[0].shape[1], warped_images[0].shape[0]))
        for i in range(len(camera)):
            # frame = cv2.cvtColor(warped_images[i], cv2.COLOR_BGR2RGB)
            frame = warped_images[i]
            out.write(frame)
        out.release()
        print('Saved video!')



            # # vis = Visualizer(save_dir = os.path.dirname(camera[idx].replace('images', 'optical_flow')), pad_value=120, linewidth=3)
            # # vis.visualize(video, pred_tracks, pred_visibility)

            # # save the optical flow
            # coord = pred_tracks[0][0].detach().cpu().round().long().numpy()
            # flow = pred_tracks[0][1].detach().cpu().numpy() - pred_tracks[0][0].detach().cpu().numpy()
            # # draw the optical flow on the image
            # for i in range(coord.shape[0]):
            #     frame_0 = cv2.arrowedLine(frame_0, tuple(coord[i]), tuple(coord[i] + flow[i].astype(int)), (0, 0, 255), 2)
            # cv2.imwrite(camera[idx].replace('images', 'optical_flow'), frame_0)
            # # save the optical flow
            # data = {'coord': coord, 'flow': flow, 'visibility': pred_visibility[0][0].detach().cpu().numpy()}   
            # np.save(camera[idx].replace('images', 'optical_flow').replace('.jpg', '.npy'), data)         


