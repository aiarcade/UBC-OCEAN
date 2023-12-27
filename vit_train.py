import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List
from typing import Optional
from torch.utils.data import random_split
import torch.nn.functional as F
from linformer import Linformer
import torch
from torchmetrics.classification import MulticlassConfusionMatrix
import warnings
import PIL
from vit_pytorch.efficient import ViT

from settings import *
from common import *

PIL.Image.MAX_IMAGE_PIXELS = 933120000




class VitModel(pl.LightningModule):
    def __init__(self,learning_rate=1e-6):#,d_model, nhead, num_encoder_layers, num_decoder_layers,learning_rate=None):
        super(VitModel, self).__init__()
        self.efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )
        self.model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=6,
            transformer=self.efficient_transformer,
            channels=3,
        )
        self.lr=learning_rate
        self.criteria=nn.CrossEntropyLoss()
        self.save_hyperparameters()
        
    def forward(self, src):
        output=self.model(src)
        return output

    def torch_balanced_accuracy(self,
        y_true: torch.tensor,
        y_pred: torch.tensor,
        adjusted: bool = False
    ) -> torch.float:

        C = self.metric(y_pred, y_true)
        per_class = torch.diag(C) / C.sum(axis=1)
        if torch.any(torch.isnan(C)):
            warnings.warn("y_pred contains classes not in y_true")
            per_class = per_class[~torch.isnan(per_class)]
        score = per_class.mean()

        if adjusted:
            n_classes = len(per_class)
            chance = 1 / n_classes
            score -= chance
            score /= 1 - chance
        return score

    def loss(self,pred,target):
        loss =  self.criteria(pred,target)#self.torch_balanced_accuracy(target,predicted_classes_batch )
        #loss = loss[~torch.isnan(loss)].mean()
        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
        loss = self.loss(outputs,y)
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
        loss =  self.loss(outputs,y)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        return loss
    def configure_optimizers(self):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class UBCDataset(Dataset):
    def __init__(self, file_list, root_dir, transform=None,train=False,val=False):
        self.file_list = file_list
        self.root_dir = root_dir
        #Mean: tensor([0.8004, 0.6944, 0.7964])
        #Std: tensor([0.1013, 0.1176, 0.0917])
        if transform is None:
            # self.transform = transforms.Compose([
            #     transforms.Resize([224,224]),
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.8004, 0.6944, 0.7964], [0.1013, 0.1176, 0.0917]), 
            # ])
            if train:
                self.transform=transforms.Compose([
                    transforms.Resize([224,224]),
                    transforms.RandomRotation(45, fill=255),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.8004, 0.6944, 0.7964], [0.1013, 0.1176, 0.0917])])
            if val:
                 self.transform=transforms.Compose([
                    transforms.Resize([224,224]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.8004, 0.6944, 0.7964], [0.1013, 0.1176, 0.0917])])
                
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img_name = os.path.basename(img_path)
        id_str = img_name.split('_')[0]  # Assuming the id is before the first underscore
        label=img_path.split("/")[-2]
        # Convert label to categorical if needed
        label_categorical = {'CC': 0, 'EC': 1, 'HGSC': 2, 'LGSC': 3, 'MC': 4,'Other': 5}[label]

        # Load the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label_categorical

    def shuffle(self):
        import random
        random.shuffle(self.file_list)


class UBCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_file:str,data_root:str):
        super().__init__()
        self.batch_size = batch_size
        self.train_file=data_file
        self.root_dir=data_root

    def setup(self, stage: Optional[str] = None) -> None:
        all_files = [os.path.join(root, file) for root, dirs, files in os.walk(self.root_dir) for file in files]
        train_size = int(0.9 * len(all_files))
        val_size = (len(all_files) - train_size)

        train_files=all_files[0:train_size]
        val_files=all_files[train_size:]

        self.train_dataset=UBCDataset(file_list=train_files, root_dir=self.root_dir,train=True)
        self.val_dataset=UBCDataset(file_list=val_files, root_dir=self.root_dir,val=True)
        
        #test_size = len(dataset) - train_size - val_size
        self.no_workers=24


    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)

    


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    lmodel=VitModel()
    datamodule = UBCDataModule(batch_size=TRAIN_BATCH_SIZE,data_file=TRAIN_CSV,data_root=TRAIN_PROCESSED_DIR)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='vit-model-epoch_{epoch:02d}_val_loss_{val_loss:.2f}',
        every_n_epochs=1,
        dirpath=MODEL_DIR_PREFIX
    )

    trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],
        accelerator=ACCELERATION, devices=DEVICES, 
        strategy="ddp")

    trainer.fit(lmodel, datamodule=datamodule)