import argparse
import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import data_setup
import deeplab_model
from utils import plt_to_tensor, calculate_metrics, update_running_means, \
    initialize_metrics

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


def test_step(model, dataloader, loss_fn, metrics, device):
    """
        Executes a single test step over the entire dataset.

        Parameters:
            model (torch.nn.Module): The trained model for inference.
            dataloader (DataLoader): DataLoader for the test dataset.
            loss_fn (callable): Loss function used for evaluation.
            metrics (dict): Dictionary containing metric functions for evaluation.
            device (torch.device): The device tensors will be transferred to.

        Returns:
            tuple: Contains test loss, accuracy, clt-IoU accuracy, confusion matrix, precision, recall, and F1 score.
        """
    running_iou_means = []
    running_ltiou_means = []

    # Setup test loss and test accuracy values
    test_loss = 0

    with torch.inference_mode():
        for i_batch, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)["out"]
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            iou_values, lt_iou = calculate_metrics(preds, labels, metrics)
            running_iou_means.append(iou_values)
            running_ltiou_means = update_running_means(running_ltiou_means, lt_iou)

        # Compute average metrics
        test_acc = torch.mean(torch.stack(running_iou_means), dim=0)
        lt_iou_acc = np.mean(running_ltiou_means) if running_ltiou_means else 0.

    final_precision = metrics["precision"].compute()
    final_recall = metrics["recall"].compute()
    final_f1 = metrics["f1_score"].compute()

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    return test_loss, test_acc, lt_iou_acc, metrics[
        "confmat"].compute().cpu().numpy(), final_precision, final_recall, final_f1


def format_percentage(value):
    #Converts a decimal value to a percentage with two decimal places.
    return round(value*100, 2)


def format_classes(value):
    #Converts a decimal value to a percentage with two decimal places.
    return round(value, 2)


def test(test_dir, weights, pruned_model, use_pruned, model_name):
    """
        Main function to execute the testing pipeline, writing the results into an excel file.

        Parameters:
            test_dir (str): Directory containing the test dataset.
            weights (str): Path to the saved model weights.
            pruned_model (str): Path to the pruned model file.
            use_pruned (bool): Flag indicating whether to use the pruned model.
            model_name (str): Name of the model being tested.

        Returns:
            None
        """
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    print("Initializing Datasets and Dataloaders...")


    test_transform = A.Compose([
            A.LongestMaxSize(max_size=768, interpolation=1),
            A.CenterCrop(512, 512),
            A.PadIfNeeded(min_height=512, min_width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),])
    test_data = data_setup.DataLoaderSegmentation(test_dir, transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=16, num_workers=NUM_WORKERS)

    # Load the model
    print("Initializing Model...")
    if use_pruned:
        model = torch.load(pruned_model)
    else:
        model = deeplab_model.initialize_model(NUM_CLASSES, keep_feature_extract=True)
    model.load_state_dict(torch.load(weights))
    model = model.to(device)
    model.eval()

    # Define the loss function and metrics
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=7)
    metrics = initialize_metrics(device)

    test_loss, test_iou, test_lt_iou, test_confmat, precision, recall, f1 = test_step(model=model,
                                                                                      dataloader=test_dataloader,
                                                                                      loss_fn=loss_fn,
                                                                                      metrics=metrics,
                                                                                      device=device)

    print(
        f"test_loss: {test_loss:.4f} | "
        f"test_iou: {test_iou:.4f} | "
        f"test_lt_iou: {test_lt_iou:.4f} | "
        f"precison: {precision: .4f} | "
        f"recall: {recall: .4f} | "
        f"F1: {f1: .4f} |"
    )

    #extract the performance per class
    classes = ["Background", "Control Point", "Vegetation", "Efflorescence", "Corrosion", "Spalling", "Crack", "Boundary"]
    row_sums = test_confmat.sum(axis=1, keepdims=True)
    normalized_confmat = (test_confmat / row_sums) * 100
    df_cm = pd.DataFrame(normalized_confmat, index=classes, columns=classes)
    class_percentages = {class_name: df_cm.loc[class_name, class_name] for class_name in classes}

    # Prepare data for Excel with formatted values, including class-specific percentages
    data = {
        "Name": model_name,  # Assuming model_name is a string
        "IoU": format_percentage(test_iou.detach().cpu().item()),  # Format IoU
        "lt IoU": format_percentage(test_lt_iou),  # Format lt IoU similarly if it's a tensor
        "F1": format_percentage(f1.detach().cpu().item()),  # Format F1
    }
    data.update({class_name: format_classes(class_percentages[class_name]) for class_name in classes})
    results_df = pd.DataFrame([data])

    # Excel file path
    excel_path = 'model_test_results.xlsx'

    # Check if the Excel file exists to append or create a new one
    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Read existing data
            existing_df = pd.read_excel(excel_path)
            # Append new data
            updated_df = pd.concat([existing_df, results_df], ignore_index=True)
            updated_df.to_excel(writer, index=False, sheet_name='Test Results')
    else:
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Test Results')

    print(f"Results appended to {excel_path}")


def args_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help="Directory for the test data")
    parser.add_argument("--weights", help='Path and name of the state dict for vanilla model')
    parser.add_argument("--pruned_model", help='Path to the pruned model file')
    parser.add_argument("--use_pruned", action='store_true', help='Flag to use the pruned model')
    args = parser.parse_args()

    model_name = os.path.basename(args.pruned_model) if args.use_pruned else 'vanilla_model'
    test(args.test_dir, args.weights, args.pruned_model, args.use_pruned, model_name)

if __name__ == "__main__":
    args_preprocess()
