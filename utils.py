import io
import math
import os
import random
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.morphology import disk, thin
from torch import from_numpy
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex, ConfusionMatrix, \
    MulticlassPrecision, MulticlassRecall, \
    MulticlassF1Score
from torchvision import transforms


def transform2lines(output, target, tol=4):
    """
        Transform predictions and labels to a thinned representation for line-based, tolerant IoU calculation.
        This method is inspired by the approach described by Benz and Rodehorst in their 2022 paper on structural defect detection.

        Parameters:
            output (tensor): The predicted segmentation mask.
            target (tensor): The ground truth segmentation mask.
            tol (int): The tolerance level for dilating the thinned lines.

        Returns:
            tuple: The transformed prediction and target masks ready for tolerant IoU calculation.

        Reference:
            Benz, Christian and Rodehorst, Volker. "Image-Based Detection of Structural Defects Using Hierarchical Multi-scale Attention."
            DAGM German Conference on Pattern Recognition. Springer, 2022.
        """

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
    """
        Computes the line-based, tolerant intersection-over-union (IoU) metric, based on a method inspired by Benz and Rodehorst's work.
        This function specifically caters to the evaluation of structural defect detection models.

        Parameters:
            pred (tensor): The predicted segmentation mask.
            target (tensor): The ground truth segmentation mask.
            tol (int): The tolerance level for dilating the thinned lines.

        Returns:
            float: The computed line-based, tolerant IoU value.

        Reference:
            Benz, Christian and Rodehorst, Volker. "Image-Based Detection of Structural Defects Using Hierarchical Multi-scale Attention."
            DAGM German Conference on Pattern Recognition. Springer, 2022.
        """

    pred = pred.cpu()
    target = target.cpu()

    pred[pred < 6] = 0
    target[target < 6] = 0

    pred[pred > 6] = 0
    target[target > 6] = 0

    pred[pred == 6] = 1
    target[target == 6] = 1

    output_thin = np.copy(pred)
    target_thin = np.copy(target)

    # thin predictions and labels
    for i in range(len(pred)):
        [output_thin[i, ...], target_thin[i, ...]] = transform2lines(pred[i, ...], target[i, ...], tol)

    iou = BinaryJaccardIndex()(from_numpy(output_thin), from_numpy(target_thin))

    return iou


def initialize_metrics(device):
    metrics = {
        "jaccard": MulticlassJaccardIndex(num_classes=8, average="weighted", ignore_index=7).to(device),
        "confmat": ConfusionMatrix(task="multiclass", num_classes=8).to(device),
        "precision": MulticlassPrecision(num_classes=8, average='weighted', ignore_index=7).to(device),
        "recall": MulticlassRecall(num_classes=8, average='weighted', ignore_index=7).to(device),
        "f1_score": MulticlassF1Score(num_classes=8, average='weighted', ignore_index=7).to(device)
    }
    return metrics


def update_running_means(running_means, new_value):
    if new_value is not None and new_value.item() != 0. and not math.isnan(new_value.item()):
        running_means.append(new_value.item())
    return running_means


def visualize_confusion_matrix(confmat, epoch=None):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    classes = ["Background", "Control Point", "Vegetation", "Efflorescence", "Corrosion", "Spalling", "Crack",
               "Boundary"]
    row_sums = confmat.sum(axis=1, keepdims=True)
    normalized_confmat = (confmat / row_sums) * 100
    df_cm = pd.DataFrame(normalized_confmat, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='.1f', cmap='Greens')  # Adjust the colormap as needed
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if epoch:
        plt.title(f'Val Confusion Matrix, Epoch {epoch}')
    else:
        plt.title("Test Confusion Matrix")
    image_val_confmat = plt_to_tensor(plt)
    plt.close()

    return image_val_confmat


def calculate_metrics(preds, labels, metrics):
    iou_values = metrics["jaccard"](preds, labels)
    metrics["confmat"].update(preds, labels)
    metrics["precision"].update(preds, labels)
    metrics["recall"].update(preds, labels)
    metrics["f1_score"].update(preds, labels)
    lt_iou = ltIoU(preds, labels)
    return iou_values, lt_iou


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    Creates a tensorboard writer for logging experiments. The log directory structure is organized by date, experiment name, model name, and an optional extra identifier.

    Args:
        experiment_name (str): The name of the experiment.
        model_name (str): The name of the model being used.
        extra (str, optional): An additional string to further specify the logs. Defaults to None.

    Returns:
        SummaryWriter: An instance of `torch.utils.tensorboard.writer.SummaryWriter` configured for the specified log directory.

    Reference:
        Bourke, Daniel. "Learn PyTorch for Deep Learning". Available at: https://github.com/mrdbourke/pytorch-deep-learning
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
    """
        Saves a PyTorch model's `state_dict` to a specified directory with the given model name. The model is saved with either a ".pth" or ".pt" file extension.

        Args:
            model (torch.nn.Module): The model to be saved.
            target_dir (str): The directory path where the model should be saved.
            model_name (str): The name of the file to save the model as. Must end with '.pth' or '.pt'.

        Raises:
            AssertionError: If `model_name` doesn't end with '.pth' or '.pt'.

        Example:
            save_model(model=my_model, target_dir="saved_models", model_name="best_model.pth")

        Reference:
            Bourke, Daniel. "Learn PyTorch for Deep Learning". Available at: https://github.com/mrdbourke/pytorch-deep-learning
        """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    # print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


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

    # colormap according to s2ds repository
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
