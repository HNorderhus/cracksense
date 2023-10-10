import glob
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import random
from utils import visualize_data

class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path, transform = None):

        #from pathlib import Path

        self.img_files = glob.glob(os.path.join(folder_path, 'Images', '*.*'))
        self.mask_files = glob.glob(os.path.join(folder_path, 'Labels_grayscale', '*.*'))
        self.transform = transform

    def dilate_cracks(self, mask):
        # Define a kernel for dilation (you can adjust the size of the kernel as needed)
        kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel, adjust the size based on your requirements
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        return dilated_mask

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if the mask contains crack pixels (class 6)
        if np.any(mask == 6):
            # Perform dilation on the mask (cracks) only when there are crack pixels in the mask
            mask = self.dilate_cracks(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)  # Pass the dilated mask to transform
            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.long()

        return image, mask

    def __len__(self):
        return len(self.img_files)



debug_mode = False

if debug_mode:
    data_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.RandomCrop(256, 256),
            A.PadIfNeeded(min_height=256, min_width=256),
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.3),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                A.GridDistortion(p=0.3),
                A.Transpose(p=0.3)
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Create an instance of the DataLoader
    folder_path = "dataset/train/"
    batch_size = 32  # Choose your desired batch size

    dataset = DataLoaderSegmentation(folder_path, transform=data_transform)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(len(dataset))

    num_samples_to_display = 10
    sample_indices = random.sample(range(len(dataset)), num_samples_to_display)
    images_to_display = []
    masks_to_display = []

    for idx in sample_indices:
        image, mask = dataset[idx]
        images_to_display.append(np.transpose(image, (1, 2, 0)))
        masks_to_display.append(mask)

    # Call the visualize function with the selected images/masks
    visualize_data(images_to_display, masks_to_display)
