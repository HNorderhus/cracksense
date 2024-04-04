import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import argparse
import deeplab_model
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def extract_image_name(image_path):
    # Extracts the base name of an image file without its file extension.
    return os.path.splitext(os.path.basename(image_path))[0]


def run_inference(model, image_path, mode, save_figure):
    """
    Performs inference on a single image using a pre-trained DeepLab model and displays or saves the output.

    Parameters:
        model (torch.nn.Module): Pre-trained DeepLab model.
        image_path (str): Path to the input image.
        mode (str): Output visualization mode ('side_by_side', 'overlay', 'save_mask').
        save_figure (bool): If True, saves the output image; otherwise, displays it.

    Returns:
        None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    transforms_image = A.Compose([
            A.LongestMaxSize(max_size=768, interpolation=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    # Load and transform the image
    image = Image.open(image_path)
    image_np = np.asarray(image)
    transformed = transforms_image(image=image_np)
    image_transformed = transformed["image"]
    image_tensor = image_transformed.unsqueeze(0).to(device)

    # Perform inference
    outputs = model(image_tensor)["out"]
    _, preds = torch.max(outputs, 1)
    preds = preds.to("cpu")
    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

    # Convert segmentation predictions to colored image using the custom colormap
    custom_colormap = np.array([
        [0, 0, 0],        # Background (Black)
        [0, 0, 255],      # Control Point (Blue)
        [0, 255, 0],      # Vegetation (Green)
        [0, 255, 255],    # Efflorescence (Cyan)
        [255, 255, 0],    # Corrosion (Yellow)
        [255, 0, 0],      # Spalling (Red)
        [255,255,255]   # Crack (White)
    ])

    preds_color = custom_colormap[preds_np]
    preds_color = preds_color.astype(np.uint8)
    preds_pil = Image.fromarray(preds_color)

    # Visualization and saving options
    if mode == "side_by_side":
        # Plot the original image and the generated mask side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(preds_pil)
        axes[1].set_title('Generated Mask')
        axes[1].axis('off')

        plt.tight_layout()
        if save_figure:
            image_name = extract_image_name(image_path)
            plt.savefig(f"results/{image_name}_10percent.png")
        plt.close()
    elif mode == "overlay":
        # Overlay the mask on the original image with 30% opacity
        mask_with_opacity = cv2.addWeighted(image_np, 0.7, preds_color, 0.3, 0)
        plt.figure(figsize=(6,6))
        plt.imshow(mask_with_opacity)
        plt.title('Mask Overlay')
        plt.axis('off')
        if save_figure:
            image_name = extract_image_name(image_path)
            plt.savefig(f"results/{image_name}_overlay_ignore_F1.png")
        plt.close()
    elif mode == "save_mask":
        # Save only the segmentation mask
        image_name = extract_image_name(image_path)
        mask_filename = f"results/{image_name}_prun.png"
        preds_color_rgb = cv2.cvtColor(preds_color, cv2.COLOR_BGR2RGB)
        cv2.imwrite(mask_filename, preds_color_rgb)
        print(f"Saved generated mask.")


def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict", help='Path and name of the state dict')
    parser.add_argument("--image", help="Path and Name of the image")
    parser.add_argument("--mode", choices=["side_by_side", "overlay", "save_mask"], default="side_by_side", help="Visualization mode (default: side_by_side)")
    parser.add_argument("--save_figure", type=bool, default=False, help="Save the resulting figure")
    parser.add_argument("--pruned_model", help='Path to the pruned model file')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pruned model if specified, otherwise initialize and load state dict
    if args.pruned_model:
        model = torch.load(args.pruned_model, map_location=device)
    else:
        model = deeplab_model.initialize_model(num_classes=8, keep_feature_extract=True)
        model.load_state_dict(torch.load(args.state_dict, map_location=device))

    model = model.to(device)
    model.eval()

    # Run inference on the provided image
    run_inference(model, args.image, args.mode, args.save_figure)

if __name__ == "__main__":
    args_preprocess()
