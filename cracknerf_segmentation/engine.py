import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from utils import iou, ltIoU
import numpy as np


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):

    # Put model in train mode
    model.train()

    running_loss = 0.0
    running_iou_means = []

    # Setup train loss and train accuracy values
    train_loss = 0

    # Loop through data loader data batches
    for i_batch, sample_batched in enumerate(dataloader):
        #print(f"Batch {i_batch + 1}/{len(dataloader)}")

        #model.to(device)
        optimizer.zero_grad()

        inputs, labels = sample_batched
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)["out"]

        loss = loss_fn(outputs, labels)
        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        iou_mean = iou(preds, labels, 7).mean()
        running_loss += loss.item() * inputs.size(0)
        running_iou_means.append(iou_mean)

        lt_iou = ltIoU(preds, labels).mean()

    #epoch_loss = running_loss / len(dataloader)
    if running_iou_means is not None:
        train_acc = np.array(running_iou_means).mean()
    else:
        train_acc = 0.

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    #train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, lt_iou


def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):

    # Put model in eval mode
    model.eval()

    running_loss = 0.0
    running_iou_means = []

    # Setup test loss and test accuracy values
    val_loss = 0

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
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            iou_mean = iou(preds, labels, 7).mean()
            running_loss += loss.item() * inputs.size(0)
            running_iou_means.append(iou_mean)

            lt_iou = ltIoU(preds, labels).mean()

    if running_iou_means is not None:
        test_acc = np.array(running_iou_means).mean()
    else:
        test_acc = 0.

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    #test_acc = test_acc / len(dataloader)
    return val_loss, test_acc, lt_iou


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter  # new parameter to take in a writer
        ):


    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_iou, train_lt_iou = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_iou, test_lt_iou = val_step(model=model,
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
            f"val_iou: {val_iou:.4f}"
            f"test_lt_iou: {test_lt_iou:.4f} | "
        )

        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": val_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="iou",
                               tag_scalar_dict={"train_iou": train_iou,
                                                "val_iou": val_iou,
                                                "train_lt_iou": train_iou,
                                                 "test_lt_iou": val_iou},
                               global_step=epoch)

            # Close the writer
            writer.close()
        else:
            pass

