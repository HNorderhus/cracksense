import os
import torch
import data_setup, deeplab_model, engine
from torch.utils.data import DataLoader
from utils import create_writer,save_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse


def main(train_dir, val_dir, epochs, name, pruned_model):
    # Setup hyperparameters
    NUM_EPOCHS = epochs
    LEARNING_RATE = 0.01

    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 7
    BATCH_SIZE = 32

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    print("Initializing Datasets and Dataloaders...")

    train_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=512, interpolation=1),
                A.RandomCrop(224, 224),
                A.PadIfNeeded(min_height=224, min_width=224),
                A.VerticalFlip(p=0.4),  # Increased intensity
                A.HorizontalFlip(p=0.4),  # Increased intensity
                A.RandomRotate90(p=0.4),  # Increased intensity
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.33),
                    A.GridDistortion(p=0.33),
                    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=4, min_height=16, min_width=16,
                                    fill_value=0, p=0.33),
                ], p=0.8),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.33),
                    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.33),
                    A.CLAHE(p=0.33),
                ], p=0.8),
                A.OneOf([
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.33),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.33),
                    A.ColorJitter(p=0.33),
                ], p=0.8),
                A.OneOf([
                    A.RandomFog(p=0.33),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.33),
                    A.Blur(blur_limit=(3, 7), p=0.33),  # Increased intensity
                ], p=0.8),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=768, interpolation=1),
            A.CenterCrop(512, 512),
            A.PadIfNeeded(min_height=512, min_width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_data = data_setup.DataLoaderSegmentation(folder_path=train_dir, transform=train_transform)
    val_data = data_setup.DataLoaderSegmentation(folder_path=val_dir, transform=val_transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print("Initializing Model...")

    if pruned_model:
        model_path = f'results/models/{pruned_model}'
        model = torch.load(model_path)

        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        model = deeplab_model.initialize_model(NUM_CLASSES, keep_feature_extract=True, print_model=False)

    # Set loss and optimizer
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=7, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

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
    parser.add_argument("epochs", type=int, help="Number of epochs to train for")
    parser.add_argument("name", type=str, help="Name of the current training variant")
    parser.add_argument("--pruned_model", type=str, help="Load the pruned model")

    args = parser.parse_args()
    main(args.train_dir, args.val_dir, args.epochs, args.name, args.pruned_model)

if __name__ == '__main__':
    args_preprocess()

