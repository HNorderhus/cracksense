import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from utils import iou, ltIoU, save_model
import numpy as np
from torchmetrics.classification import MulticlassJaccardIndex



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

    # Setup train loss and train accuracy values
    train_loss = 0

    jaccard_metric = MulticlassJaccardIndex(num_classes=7, ignore_index=6).to(device)

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
       # train_loss += loss.item()
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        #iou_mean = iou(preds, labels, 7).mean()
        #running_iou_means.append(iou_mean)

        iou_values = jaccard_metric(preds, labels)
        #print(f"iou_values: {iou_values}")

        running_iou_means.append(iou_values)

        lt_iou = ltIoU(preds, labels).mean()
        running_ltiou_means.append(lt_iou)

    #epoch_loss = running_loss / len(dataloader)

    if running_iou_means is not None:
        #train_acc = np.array(running_iou_means).mean()
        #iou_tensor = torch.cat(running_iou_means)
        train_acc = torch.mean(torch.stack(running_iou_means), dim=0)

        lt_iou_acc = np.array(running_ltiou_means).mean()
    else:
        train_acc = 0.

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = running_loss / len(dataloader)
    #train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, lt_iou_acc


def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):

    # Put model in eval mode
    model.eval()

    running_loss = 0.0
    running_iou_means = []
    running_ltiou_means = []

    # Setup test loss and test accuracy values
    val_loss = 0

    jaccard_metric = MulticlassJaccardIndex(num_classes=7, ignore_index=6).to(device)

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
            #val_loss += loss.item()
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            #iou_mean = iou(preds, labels, 7).mean()
            #running_iou_means.append(iou_mean)

            iou_values = jaccard_metric(preds, labels)
            running_iou_means.append(iou_values)

            lt_iou = ltIoU(preds, labels).mean()
            running_ltiou_means.append(lt_iou)

    #epoch_loss = running_loss / len(dataloader)
    if running_iou_means is not None:
#        val_acc = np.array(running_iou_means).mean()
        val_acc = torch.mean(torch.stack(running_iou_means), dim=0)
        lt_iou_acc = np.array(running_ltiou_means).mean()
    else:
        val_acc = 0.

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    #val_acc = val_acc / len(dataloader)
    return val_loss, val_acc, lt_iou_acc


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
        patience = 10
        epochs_without_improvement = 0

        # Loop through training and testing steps for a number of epochs
        try:
            with writer:
                for epoch in tqdm(range(epochs)):
                    train_loss, train_iou, train_lt_iou = train_step(model=model,
                                                       dataloader=train_dataloader,
                                                       loss_fn=loss_fn,
                                                       optimizer=optimizer,
                                                       device=device)
                    val_loss, val_iou, val_lt_iou = val_step(model=model,
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
                        epochs_without_improvement = 0
                        # Save the current best model
                        save_model(model=model,
                                   target_dir="results/models",
                                   model_name=f"{name}.pth")
                    else:
                        epochs_without_improvement += 1
                        # If validation loss doesn't improve for 'patience' epochs, stop training
                        if epochs_without_improvement >= patience:
                            print("Early stopping. Restoring model from epoch: ", epoch - patience)
                            break

        except KeyboardInterrupt:
            # Handle manual interruption by saving the current state of the model
            print("Training interrupted. Saving model.")
            torch.save(model.state_dict(), f'{name}_stopped.pth')

