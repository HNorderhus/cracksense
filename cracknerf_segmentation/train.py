import os
import torch
import data_setup, deeplab_model, engine
from torch.utils.data import DataLoader
from utils import create_writer,save_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse


def main(train_dir, val_dir, batchsize, epochs, learning_rate, name):
    # Setup hyperparameters
    NUM_EPOCHS = epochs
    LEARNING_RATE = learning_rate

    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 7
    BATCH_SIZE = batchsize
    #MOMENTUM = 0.9

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    print("Initializing Datasets and Dataloaders...")

    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.RandomCrop(256, 256),
            A.PadIfNeeded(min_height=256, min_width=256),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.6),
            A.RandomRotate90(p=0.6),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.4),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2),
                A.GridDistortion(p=0.3),
                A.Transpose(p=0.5)
            ], p=0.8),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.CenterCrop(256, 256),
            A.PadIfNeeded(min_height=256, min_width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_data = data_setup.DataLoaderSegmentation(folder_path=train_dir, transform=train_transform)
    val_data = data_setup.DataLoaderSegmentation(folder_path=val_dir, transform=val_transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print("Initializing Model...")
    model = deeplab_model.initialize_model(NUM_CLASSES, keep_feature_extract=True, print_model=False)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    tensorboard_writer = create_writer(experiment_name="deeplabv3_resnet101",
                                   model_name=f"{name}")

    print("Begin training...")

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 val_dataloader=val_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device,
                 writer=tensorboard_writer,
                 name=name)


def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help='Directory path, should contain train/Images, train/Labels_grayscale')
    parser.add_argument("val_dir", help='Directory path, should contain val/Images and val/Labels_grayscale')
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Optimizer learning rate")
    parser.add_argument("--name", type=str, help="Name of the current training variant")


    args = parser.parse_args()

    main(args.train_dir, args.val_dir, args.batch_size, args.epochs, args.learning_rate, args.name)

if __name__ == '__main__':
    args_preprocess()

