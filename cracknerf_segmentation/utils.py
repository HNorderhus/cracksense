import os
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch import from_numpy
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryJaccardIndex
from skimage.morphology import disk, thin
import matplotlib.pyplot as plt
import random
import io
from PIL import Image
from torchvision import transforms


# -------------------------------------------
# functions to calculate IoU
# -------------------------------------------

def iou(pred, target, n_classes = 7):
    """
    Calculate Intersection over Union (IoU) for semantic segmentation predictions.

    This function computes the IoU for each class in semantic segmentation predictions.
    IoU measures the overlap between predicted and target masks for each class,
    excluding the background class (class "0").

    Args:
        pred (torch.Tensor): Predicted segmentation map.
        target (torch.Tensor): Target segmentation map (ground truth).
        n_classes (int, optional): Number of classes, excluding the background class. Defaults to 7.

    Returns:
        numpy.ndarray: An array of IoU scores, one for each class.

    Note:
        - The input 'pred' and 'target' should be PyTorch tensors with the same shape.
        - The background class (class "0") is excluded from the IoU calculation.
    """
    ious = []

    # convert 2D matrices to 1D arrays
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1,n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)


def transform2lines(output, target, tol=4):
    """ Transform predictions and labels to thinned representation for line-base, tolerant IoU. """
    # thin predictions and labels
    output_thin = np.uint8(thin(output.cpu().numpy()))
    target_thin = np.uint8(thin(target.cpu().numpy()))

    # apply tolerance
    output_thin_dil = cv2.dilate(output_thin, disk(tol), iterations=1)
    target_thin_dil = cv2.dilate(target_thin, disk(tol), iterations=1)

    # derive true/false positives and negatives
    tp = target_thin * output_thin_dil
    fp = output_thin - output_thin * target_thin_dil
    fn = target_thin - tp

    output, target = tp + fp, tp + fn
    return output, target



def ltIoU(pred, target, tol=4):
    """ Computes line-based, tolerant intersection-over-union (IoU). """
    pred = pred.cpu()
    target = target.cpu()

    pred[pred < 6] = 0
    target[target < 6] = 0

    pred[pred == 6] = 1
    target[target == 6] = 1

    output_thin = np.copy(pred)
    target_thin = np.copy(target)

    # thin predictions and labels
    for i in range(len(pred)):
      [output_thin[i, ...], target_thin[i, ...]] = transform2lines(pred[i, ...], target[i, ...], tol)

    iou = BinaryJaccardIndex()(from_numpy(output_thin), from_numpy(target_thin))

    return iou

# -------------------------------------------
# functions to document and save model
# -------------------------------------------


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    The log directory structure is as follows:

    results/runs/YYYY-MM-DD/experiment_name/model_name[/extra]

    Args:
        experiment_name (str): The name of the experiment.
        model_name (str): The name of the model.
        extra (str, optional): An optional extra identifier for further categorization within
            the log directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter: A SummaryWriter instance configured to
        log data to the specified directory.
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("results/runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("results/runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")

    return SummaryWriter(log_dir=log_dir)


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    #print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)



# ------------------------------------------
# functions to convert RGB labels to grayscale
# -------------------------------------------

def plt_to_tensor(plt):
    # Save the Matplotlib figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Open the BytesIO object as an Image and convert it to a PyTorch tensor
    image = Image.open(buf).convert('RGB')
    return transforms.ToTensor()(image)


def rgb2mask(img: np.ndarray):
    """
    Convert an RGB image to a semantic mask using a predefined color-to-label mapping.

    Args:
        img (numpy.ndarray): The input RGB image to be converted to a mask.

    Returns:
        numpy.ndarray: A 2D numpy array representing the semantic mask, where each pixel's
        value corresponds to a specific category label.

    Note:
        - The input image 'img' should be a numpy array with three channels (RGB).
    """
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    #colormap according to s2ds repository
    cmap = {
        (0, 0, 0): 0,  # Background
        (0, 0, 255): 1,  # Control Point
        (0, 255, 0): 2,  # Vegetation
        (0, 255, 255): 3,  # Efflorescence
        (255, 255, 0): 4,  # Corrosion
        (255, 0, 0): 5,  # Spalling
        (255, 255, 255): 6  # Crack
    }

    mask = np.zeros((height, width), dtype=np.uint8)

    for color, label in cmap.items():
        mask[np.all(img == np.array(color), axis=-1)] = label

    return mask


def convert_rgb_to_grayscale(input_dir, output_dir):
    """
    Convert RGB images in an input directory to grayscale and save them in an output directory.

    Args:
        input_dir (str): The directory containing the RGB images to be converted.
        output_dir (str): The directory where the grayscale images will be saved.

    Note:
        - Grayscale images will be saved in the output directory with the same filenames as the
          corresponding RGB images.
    """
    # Directory containing the images
    input_directory = input_dir

    # Output directory for saving grayscale images
    output_directory = output_dir

    # Loop through each image in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.png'):  # You can change the extension as needed
            # Read the image
            image_path = os.path.join(input_directory, filename)
            pil_image = cv2.imread(image_path)
            pil_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2RGB)

            # Apply the rgb2mask function
            result = rgb2mask(pil_image)

            # Save the result as a grayscale image with the same name
            output_path = os.path.join(output_directory, filename)
            cv2.imwrite(output_path, result)


# -------------------------------------------
# visualize the augmented images from the dataloader
# -------------------------------------------

def visualize_data(images, masks):
    num_samples = len(images)
    sample_indices = random.sample(range(num_samples), 10)  # Select 10 random samples

    f, axs = plt.subplots(10, 2, figsize=(8, 20))

    for i, idx in enumerate(sample_indices):
        image = images[idx]
        mask = masks[idx]

        axs[i, 0].imshow(image)
        axs[i, 1].imshow(mask, cmap="gray")

        axs[i, 0].axis('off')
        axs[i, 1].axis('off')

        unique_values = np.unique(mask)
        axs[i, 1].set_title(f"Unique Values: {unique_values}")

    plt.show()

