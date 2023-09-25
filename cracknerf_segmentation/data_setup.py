from PIL import Image
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



class DataLoaderSegmentation(Dataset):
    def __init__(self, folder_path):
        self.img_files = glob.glob(os.path.join(folder_path, 'Images', '*.*'))
        self.label_files = glob.glob(os.path.join(folder_path, 'Labels_grayscale', '*.*'))
        #self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        image = Image.open(img_path)
        label = Image.open(label_path)

        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.RandomResizedCrop((512, 512)),
            #transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ])

        # Convert the image to a PyTorch tensor

        image = self.transforms(image)
        label = self.transforms(label)

        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = normalize_transform(image)

        # label = np.array(label)
        label = label * 255
        label = label.long().squeeze()

        return image, label

    def __len__(self):
        return len(self.img_files)



debug_mode = False

if debug_mode:
    # Define the data transformation for the image (e.g., toTensor)
    data_transform = transforms.Compose([transforms.ToTensor()])

    # Create an instance of the DataLoader
    folder_path = "dataset/train/"
    batch_size = 32  # Choose your desired batch size

    dataset = DataLoaderSegmentation(folder_path, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(len(dataset))