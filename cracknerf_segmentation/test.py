from utils import iou, ltIoU
import numpy as np
import os
import torch
import data_setup, deeplab_model
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

def test_step(model, dataloader, loss_fn, state_dict, device):

    # Put model in eval mode
    state_dict = torch.load(state_dict, map_location=device)

    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    running_iou_means = []
    running_ltiou_means = []

    # Setup test loss and test accuracy values
    test_loss = 0

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

            iou_mean = iou(preds, labels, 7).mean()
            running_iou_means.append(iou_mean)

            lt_iou = ltIoU(preds, labels).mean()
            running_ltiou_means.append(lt_iou)

    #epoch_loss = running_loss / len(dataloader)
    if running_iou_means is not None:
        test_acc = np.array(running_iou_means).mean()
        lt_iou_acc = np.array(running_ltiou_means).mean()
    else:
        test_acc = 0.

    # Adjust metrics to get average loss and accuracy per batch
    test_loss =test_loss / len(dataloader)
    #test_acc =test_acc / len(dataloader)
    return test_loss,test_acc, lt_iou_acc


def test(test_dir, state_dict):
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 7

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    state_dict = torch.load(state_dict, map_location=device)

    print("Initializing Datasets and Dataloaders...")

    test_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.CenterCrop(256, 256),
            A.PadIfNeeded(min_height=256, min_width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_data = data_setup.DataLoaderSegmentation(folder_path=test_dir, transform=test_transform)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)

    print("Initializing Model...")
    model = deeplab_model.initialize_model(NUM_CLASSES, keep_feature_extract=True, print_model=False)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()

    test_loss, test_iou, test_lt_iou = test_step(model=model,
                                               dataloader=test_dataloader,
                                               loss_fn=loss_fn,
                                               state_dict=state_dict,
                                               device=device)

    print(
                f"test_loss: {test_loss:.4f} | "
                f"test_iou: {test_iou:.4f} | "
                f"test_lt_iou: {test_lt_iou:.4f} | "
            )

def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help="Directory for the test data")
    parser.add_argument("state_dict", help='Path and name of the state dict')

    args = parser.parse_args()
    test(args.test_dir, args.state_dict)

if __name__ == "__main__":
    args_preprocess()