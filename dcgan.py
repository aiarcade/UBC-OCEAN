import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from settings import *

import PIL

from settings import *
from common import *
import os
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List
from typing import Optional
from torch.utils.data import random_split
import torch.nn.functional as F

PIL.Image.MAX_IMAGE_PIXELS = 933120000


class GANDataset(Dataset):
    def __init__(self, csv_file, root_dir,label, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir+'/'+label+'/'

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                #transforms.Normalize([0.8004, 0.6944, 0.7964], [0.1013, 0.1176, 0.0917]),
            ])

        # Filter the DataFrame to include only 'CC' images where is_tma is True
        filtered_df = self.data[(self.data['is_tma']) & (self.data['label'] == label)]

        all_files = [os.path.join(root, file) for root, dirs, files in os.walk(self.root_dir ) for file in files]
        self.file_list = all_files
        self.length = len(self.file_list)

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
        label_mapping = {'CC': 0}
        label_categorical = label_mapping.get(label, -1)  # Default to -1 if label is not found

        # Load the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label_categorical

class GANDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_file:str,data_root:str,label:str):
        super().__init__()
        self.batch_size = batch_size
        self.train_file=data_file
        self.root_dir=data_root
        self.label=label

    def setup(self, stage: Optional[str] = None) -> None:
        dataset=GANDataset(csv_file=TRAIN_CSV, root_dir=TRAIN_GAN_DIR,label=self.label)
        train_size = int(len(dataset))
        #val_size = (len(dataset) - train_size) 
        #test_size = len(dataset) - train_size - val_size
        self.no_workers=24
        self.train_dataset=dataset#, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    #def val_dataloader(self) -> DataLoader:
    #    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    


from pl_bolts.models.gans import DCGAN

if __name__ == "__main__":
    pl.seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    datamodule = GANDataModule(batch_size=256,data_file=TRAIN_CSV,data_root=TRAIN_PROCESSED_DIR,label='CC')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        filename='GAN_CC-model-epoch_{epoch:02d}',
        every_n_epochs=1,
        dirpath=MODEL_DIR_PREFIX
    )
    lmodel = DCGAN(image_channels=3)
    trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback],
        accelerator=ACCELERATION, devices=DEVICES, 
        strategy="ddp")

    trainer.fit(lmodel, datamodule=datamodule)