import os
import torch
import data_setup, deeplab_model, engine
from torch.utils.data import DataLoader
from utils import create_writer,save_model


# Setup hyperparameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

NUM_WORKERS = os.cpu_count()
NUM_CLASSES = 7
BATCH_SIZE = 32
MOMENTUM = 0.9


def main(train_dir, val_dir):

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    print("Initializing Datasets and Dataloaders...")

    train_data = data_setup.DataLoaderSegmentation(folder_path=train_dir)
    val_data = data_setup.DataLoaderSegmentation(folder_path=val_dir)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print("Initializing Model...")
    model = deeplab_model.initialize_model(NUM_CLASSES, keep_feature_extract=True, print_model=False)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    tensorboard_writer = create_writer(experiment_name="deeplabv3_resnet101",
                                   model_name="50_epochs",
                                   extra="modified_ltiou")

    print("Begin training...")

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 val_dataloader=val_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device,
                 writer=tensorboard_writer)

    # Save the model with help from utils.py
    save_model(model=model,
                     target_dir="results/models",
                     model_name="50epochs_ltiou.pth")


if __name__ == "__main__":
    # Setup directories
    train_dir = "dataset/train/"
    test_dir = "dataset/val/"

    main(train_dir, test_dir)

