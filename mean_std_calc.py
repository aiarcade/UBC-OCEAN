import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from settings import *
import pandas as pd

import os

class UBCDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        # Get all file names in the root directory and subdirectories
        all_files = [os.path.join(root, file) for root, dirs, files in os.walk(root_dir) for file in files]
        self.file_list = all_files
        self.length = len(self.file_list)

        # Shuffle the file list
        self.shuffle()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img_name = os.path.basename(img_path)
        id_str = img_name.split('_')[0]  # Assuming the id is before the first underscore
        image_id = int(id_str)

        # Fetch class from train.csv based on image_id
        label = self.data.loc[self.data['image_id'] == image_id, 'label'].values[0]

        # Convert label to categorical if needed
        # You can use a dictionary to map label strings to categorical values
        label_categorical = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4,'Other': 5}[label]

        # Load the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label_categorical

    def shuffle(self):
        import random
        random.shuffle(self.file_list)
def calculate_mean_std(dataset):
    # Initialize variables to accumulate mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)

    # Iterate through the dataset and accumulate mean and std
    for sample,id in dataset:
        mean += sample.mean(dim=[1, 2])
        std += sample.std(dim=[1, 2])

    # Calculate the mean and std over the entire dataset
    mean /= len(dataset)
    std /= len(dataset)

    return mean, std

# Instantiate your dataset
dataset =UBCDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_PROCESSED_DIR)

# Calculate mean and std
mean, std = calculate_mean_std(dataset)

print(f'Mean: {mean}')
print(f'Std: {std}')
