import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from utils import iou, ltIoU, save_model, plt_to_tensor
import numpy as np
from torchmetrics.classification import MulticlassJaccardIndex, ConfusionMatrix
import math
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    # Put model in train mode
    model.train()

    running_loss = 0.0
    running_iou_means = []
    running_ltiou_means = []

    jaccard_metric = MulticlassJaccardIndex(num_classes=7, ignore_index=6).to(device)
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=7).to(device)
    confmat = torch.zeros((7, 7), device=device)

    # Loop through data loader data batches
    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()

        inputs, labels = sample_batched
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)["out"]

        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        iou_values = jaccard_metric(preds, labels)

        confmat_batch = confmat_metric(preds, labels)
        confmat += confmat_batch

        running_iou_means.append(iou_values)

        lt_iou = ltIoU(preds,labels)
        running_ltiou_means.append(lt_iou)
        train_acc = torch.mean(torch.stack(running_iou_means), dim=0)

    if running_ltiou_means is not None:
        res = []
        for val in running_ltiou_means:
            if val.item() != 0. or not math.isnan(val.item()):
                res.append(val)
        lt_iou_acc = np.array(res).mean()
    else:
        lt_iou_acc = 0.

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = running_loss / len(dataloader)
    return train_loss, train_acc, lt_iou_acc, confmat.cpu().numpy()


def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):

    model.eval()

    running_iou_means = []
    running_ltiou_means = []
    val_loss = 0

    jaccard_metric = MulticlassJaccardIndex(num_classes=7).to(device)
    confmat_metric = ConfusionMatrix(task="multiclass", num_classes=7).to(device)
    confmat = torch.zeros((7, 7), device=device)

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for i_batch, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)["out"]

            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            iou_values = jaccard_metric(preds, labels)

            confmat_batch = confmat_metric(preds, labels)
            confmat += confmat_batch

            running_iou_means.append(iou_values)

            lt_iou = ltIoU(preds, labels)
            running_ltiou_means.append(lt_iou)

            val_acc = torch.mean(torch.stack(running_iou_means), dim=0)

        if running_ltiou_means is not None:
            res = []
            for val in running_ltiou_means:
                if val.item() != 0. or not math.isnan(val.item()):
                    res.append(val)
            lt_iou_acc = np.array(res).mean()
        else:
            lt_iou_acc = 0.

    val_loss = val_loss / len(dataloader)

    return val_loss, val_acc, lt_iou_acc, confmat.cpu().numpy()


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          name: str# new parameter to take in a writer
        ):


    # Make sure model on target device
    model.to(device)

    best_val_loss = float('inf')

    # Loop through training and testing steps for a number of epochs
    with writer:
        for epoch in tqdm(range(epochs)):
            train_loss, train_iou, train_lt_iou, train_confmat = train_step(model=model,
                                               dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
            val_loss, val_iou, val_lt_iou, val_confmat = val_step(model=model,
                                            dataloader=val_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)

            # Print out what's happening
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_iou: {train_iou:.4f} | "
                f"train_lt_iou: {train_lt_iou:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_iou: {val_iou:.4f} | "
                f"val_lt_iou: {val_lt_iou:.4f} | "
            )


            # Add results to SummaryWriter

            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": val_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="IoU",
                               tag_scalar_dict={"train_iou": train_iou,
                               "val_iou": val_iou,
                               "train_lt_iou": train_lt_iou,
                               "val_lt_iou": val_lt_iou},
                               global_step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the current best model
                save_model(model=model,
                           target_dir="results/models",
                           model_name=f"{name}.pth")

            if epoch % 50 == 0:
                classes = ["Background", "Control Point", "Vegetation", "Efflorescence", "Corrosion", "Spalling", "Crack"]

                row_sums = train_confmat.sum(axis=1, keepdims=True)
                normalized_confmat = (train_confmat / row_sums) * 100
                df_cm = pd.DataFrame(normalized_confmat, index=classes, columns=classes)

                plt.figure(figsize=(10, 7))
                sn.heatmap(df_cm, annot=True, fmt='.1f', cmap='Blues')  # Adjust the colormap as needed
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Train Confusion Matrix, Epoch {epoch}')
                image_train_confmat = plt_to_tensor(plt)
                plt.close()  # Close the figure to free up resources

                # Add the image to TensorBoard
                writer.add_image('Train Confusion Matrix', image_train_confmat, global_step=epoch)

                row_sums = val_confmat.sum(axis=1, keepdims=True)
                normalized_confmat = (val_confmat / row_sums) * 100
                df_cm = pd.DataFrame(normalized_confmat, index=classes, columns=classes)

                plt.figure(figsize=(10, 7))
                sn.heatmap(df_cm, annot=True, fmt='.1f', cmap='Greens')  # Adjust the colormap as needed
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Val Confusion Matrix, Epoch {epoch}')
                image_val_confmat = plt_to_tensor(plt)
                plt.close()

                writer.add_image('Val Confusion Matrix', image_val_confmat, global_step=epoch)