import os
import sys
import argparse
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms


sys.path.append('../ext/face-parsing')
from models.bisenet import BiSeNet

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


import cv2
import numpy as np


ATTRIBUTES = [
    'background',
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat'
]

COLOR_LIST = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
]


def vis_parsing_maps(image, segmentation_mask, save_image=False, save_path="result.png"):
    # Create numpy arrays for image and segmentation mask
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)

    # hair mask
    hair_mask = np.zeros_like(segmentation_mask)
    hair_mask[segmentation_mask ==  ATTRIBUTES.index('hair')] = 1
    hair_mask_path = save_path.replace('face-parsing', 'face-parsing/hair')
    if not os.path.exists(os.path.dirname(hair_mask_path)):
        os.makedirs(os.path.dirname(hair_mask_path))
    cv2.imwrite(hair_mask_path, hair_mask * 255)

    # head mask
    head_mask = np.ones_like(segmentation_mask)
    head_mask[segmentation_mask ==  ATTRIBUTES.index('background')] = 0
    # maybe add cloth back
    head_mask[segmentation_mask ==  ATTRIBUTES.index('cloth')] = 0
    head_mask[segmentation_mask ==  ATTRIBUTES.index('hair')] = 0
    head_mask[segmentation_mask ==  ATTRIBUTES.index('hat')] = 0
    # head_mask_path = save_path.replace('.', '_head.')
    # head_mask_path = save_path.replace('face-parsing', 'masks/face')
    head_mask_path = save_path.replace('face-parsing', 'face-parsing/head')
    if not os.path.exists(os.path.dirname(head_mask_path)):
        os.makedirs(os.path.dirname(head_mask_path))
    cv2.imwrite(head_mask_path, head_mask * 255)


    # Create a color mask
    segmentation_mask_color = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3))

    num_classes = np.max(segmentation_mask)

    for class_index in range(1, num_classes + 1):
        class_pixels = np.where(segmentation_mask == class_index)
        segmentation_mask_color[class_pixels[0], class_pixels[1], :] = COLOR_LIST[class_index]

    segmentation_mask_color = segmentation_mask_color.astype(np.uint8)

    # Convert image to BGR format for blending
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blend the image with the segmentation mask
    blended_image = cv2.addWeighted(bgr_image, 0.6, segmentation_mask_color, 0.4, 0)

    # Save the result if required
    if save_image:
        cv2.imwrite(save_path, segmentation_mask)
        cv2.imwrite(save_path, blended_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return blended_image


def prepare_image(image: Image.Image, input_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Prepare an image for inference by resizing and normalizing it.

    Args:
        image: PIL Image to process
        input_size: Target size for resizing

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Resize the image
    resized_image = image.resize(input_size, resample=Image.BILINEAR)

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Apply transformations
    image_tensor = transform(resized_image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


def load_model(model_name: str, num_classes: int, weight_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load and initialize the BiSeNet model.

    Args:
        model_name: Name of the backbone model (e.g., "resnet18")
        num_classes: Number of segmentation classes
        weight_path: Path to the model weights file
        device: Device to load the model onto

    Returns:
        torch.nn.Module: Initialized and loaded model
    """
    model = BiSeNet(num_classes, backbone_name=model_name)
    model.to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        raise ValueError(f"Weights not found from given path ({weight_path})")

    model.eval()
    return model


def get_files_to_process(input_path: str) -> List[str]:
    """
    Get a list of image files to process based on the input path.

    Args:
        input_path: Path to a single image file or directory of images

    Returns:
        List[str]: List of file paths to process
    """
    if os.path.isfile(input_path):
        return [input_path]

    # Get all files from the directory
    # files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    files = glob.glob(os.path.join(input_path, 'image_*' ))

    # Filter for image files only
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [f for f in files if os.path.isfile(f) and f.lower().endswith(image_extensions)]


@torch.no_grad()
def inference(params: argparse.Namespace) -> None:
    """
    Run inference on images using the face parsing model.

    Args:
        params: Configuration namespace containing required parameters
    """
    frames = os.listdir(params.input)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    num_classes = 19  # Number of face parsing classes
    try:
        model = load_model(params.model, num_classes, params.weight, device)
        logger.info(f"Model loaded successfully: {params.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    for frame in tqdm(frames):

        output_path = os.path.join(params.output, frame)
        os.makedirs(output_path, exist_ok=True)


        image_folder = os.path.join(params.input, frame)

        # Get list of files to process
        files_to_process = get_files_to_process(image_folder)
        logger.info(f"Found {len(files_to_process)} files to process")

        # Process each file
        for file_path in tqdm(files_to_process, desc="Processing images"):
            filename = os.path.basename(file_path)
            save_path = os.path.join(output_path, filename)

            try:
                # Load and process the image
                image = Image.open(file_path).convert("RGB")
                mask_path = file_path.replace("image_", "mask_")
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                    # image = Image.composite(0, image, mask)
                    image = Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask)


                # Store original image resolution
                original_size = image.size  # (width, height)

                # Prepare image for inference
                image_batch = prepare_image(image).to(device)

                # Run inference
                output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
                predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

                # Convert mask to PIL Image for resizing
                mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))

                # Resize mask back to original image resolution
                restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)

                # Convert back to numpy array
                predicted_mask = np.array(restored_mask)

                # Visualize and save the results
                vis_parsing_maps(
                    image,
                    predicted_mask,
                    save_image=True,
                    save_path=save_path,
                )

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(f"Processing complete. Results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Validated command line arguments
    """
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34"],
        help="model name, i.e resnet18, resnet34"
    )
    parser.add_argument(
        "--weight",
        type=str,
        default="./weights/resnet18.pt",
        help="path to trained model, i.e resnet18/34"
    )
    parser.add_argument("--input", type=str, default="./assets/images/", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="./assets/results", help="path to save model outputs")

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input):
        raise ValueError(f"Input path does not exist: {args.input}")

    if not os.path.exists(os.path.dirname(args.weight)):
        logger.warning(f"Weight directory does not exist: {os.path.dirname(args.weight)}")

    return args


def main() -> None:
    """Main entry point of the script."""
    args = parse_args()
    inference(params=args)


if __name__ == "__main__":
    main()
