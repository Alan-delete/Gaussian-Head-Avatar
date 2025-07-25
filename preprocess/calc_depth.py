import os
import argparse
import glob
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import depth_pro

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_depth(module_path, data_dir, folder_name):
    cwd = os.getcwd()
    os.chdir(module_path)

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval().to(device)
    # iterate over images
    # there are mask images also in image folder, so we need to filter out those
    # under the images, there are folders for each frame, we need to create the same folder structure for NeuralHaircut_masks
    frames_names = os.listdir(f'{data_dir}/images') 
    os.makedirs(f'{data_dir}/depths', exist_ok=True)
    for frame in frames_names:
        mask_folder = os.path.join(f'{data_dir}/depths', frame) 
        os.makedirs(mask_folder, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(data_dir, 'images', '*', f'image_[0-9]*.jpg')))
    # image_paths = sorted(glob.glob(os.path.join(data_dir, 'images', '*', f'image_lowres*.jpg')))

    for image_path in tqdm.tqdm(image_paths):
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image).to(device)

        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy()  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.

        # Save the depth npy.
        depth_output_file = image_path.replace("images", "depths").replace(".png", ".npy").replace(".jpg", ".npy")
        np.save(depth_output_file, depth)

        # follow the visualization in https://github.com/apple/ml-depth-pro/blob/main/src/depth_pro/cli/run.py
        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )

        # Save as color-mapped "turbo" jpg image.
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
            np.uint8
        )
        color_map_output_file = image_path.replace("images", "depths").replace(".png", ".jpg")
        Image.fromarray(color_depth).save(
            color_map_output_file, format="JPEG", quality=90
        )



if __name__ == "__main__":
    if device.type == "cuda":
        print("Using GPU for depth prediction.")
    else:
        print("Using CPU for depth prediction.")    
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--module_path', default='../../ext/ml-depth-pro/', type=str)
    parser.add_argument('--data_path', default='../../datasets/NeuralHaircut/jenya', type=str)
    parser.add_argument('--folder_name', default='images', type=str)

    args, _ = parser.parse_known_args()
    process_depth(args.module_path, args.data_path, args.folder_name)