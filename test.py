from utils import ltIoU, plt_to_tensor
import numpy as np
import os
import torch
import data_setup, deeplab_model
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassJaccardIndex
import math
import seaborn as sn
import pandas as pd

def test_step(model, dataloader, loss_fn, state_dict, device):
    running_iou_means = []
    running_ltiou_means = []

    # Setup test loss and test accuracy values
    test_loss = 0

    jaccard_metric = MulticlassJaccardIndex(num_classes=7, ignore_index=6).to(device)
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=7).to(device)
    confmat = torch.zeros((7, 7), device=device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for i_batch, sample_batched in enumerate(dataloader):
            #print(f"Batch {i_batch + 1}/{len(dataloader)}")

            inputs, labels = sample_batched
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)["out"]

            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            iou_values = jaccard_metric(preds, labels)

            confmat_batch = confmat_metric(preds, labels)
            confmat += confmat_batch

            running_iou_means.append(iou_values)

            lt_iou = ltIoU(preds, labels)
            running_ltiou_means.append(lt_iou)
            test_acc = torch.mean(torch.stack(running_iou_means), dim=0)

    #epoch_loss = running_loss / len(dataloader)
    if running_ltiou_means is not None:
        res = []
        for val in running_ltiou_means:
            if val.item() != 0. or not math.isnan(val.item()):
                res.append(val)
        lt_iou_acc = np.nanmean(np.array(res))
    else:
        lt_iou_acc = 0.

    # Adjust metrics to get average loss and accuracy per batch
    test_loss =test_loss / len(dataloader)
    #test_acc =test_acc / len(dataloader)
    return test_loss,test_acc, lt_iou_acc, confmat.cpu().numpy()


def test(test_dir, state_dict):
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 7

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    #state_dict = torch.load(state_dict, map_location=device)

    print("Initializing Datasets and Dataloaders...")

    test_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=768, interpolation=1),
            A.CenterCrop(512, 512),
            A.PadIfNeeded(min_height=512, min_width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_data = data_setup.DataLoaderSegmentation(folder_path=test_dir, transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)

    print("Initializing Model...")
    model = deeplab_model.initialize_model(NUM_CLASSES, keep_feature_extract=True, print_model=False)
    state_dict = "results/models/e400_baseline.pth"
    #model = torch.load("results/models/p50_magnitude.pth")

    #state_dict = "results/models/e50_50mag.pth"

    model.load_state_dict(torch.load(state_dict, map_location=device))

    model = model.to(device)
    model.eval()


    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    test_loss, test_iou, test_lt_iou, test_confmat = test_step(model=model,
                                               dataloader=test_dataloader,
                                               loss_fn=loss_fn,
                                               state_dict=state_dict,
                                               device=device)

    print(
                f"test_loss: {test_loss:.4f} | "
                f"test_iou: {test_iou:.4f} | "
                f"test_lt_iou: {test_lt_iou:.4f} | "
            )

    classes = ["Background", "Control Point", "Vegetation", "Efflorescence", "Corrosion", "Spalling", "Crack"]

    row_sums = test_confmat.sum(axis=1, keepdims=True)
    normalized_confmat = (test_confmat / row_sums) * 100
    df_cm = pd.DataFrame(normalized_confmat, index=classes, columns=classes)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='.1f', cmap='Reds')  # Adjust the colormap as needed
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Test Confusion Matrix')
    image_test_confmat = plt_to_tensor(plt)
    plt.show()

def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help="Directory for the test data")
    parser.add_argument("--state_dict", help='Path and name of the state dict')

    args = parser.parse_args()
    test(args.test_dir, args.state_dict)

if __name__ == "__main__":
    args_preprocess()