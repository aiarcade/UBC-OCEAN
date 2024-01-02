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
import timm
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from settings import *
from common import *

PIL.Image.MAX_IMAGE_PIXELS = 933120000


class LitCancerSubtype(pl.LightningModule):

    def __init__(self, net, lr: float = 1e-4):
        super().__init__()
        self.net = net
        self.arch = net.pretrained_cfg.get('architecture')
        self.num_classes = net.num_classes
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self.learn_rate = lr

    def forward(self, x):
        y = F.softmax(self.net(x))
        if y.isnan().any():
            y = torch.ones_like(y) / self.num_classes
        return y

    def compute_loss(self, y_hat, y):
        #print(y)
        return F.cross_entropy(y_hat, y.to(y_hat.dtype))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        lbs = torch.argmax(y, axis=1)
        #print(f"{lbs=} ?= {y_hat=}")
        loss = self.compute_loss(y_hat, y)
        #print(f"{y=} ?= {y_hat=} -> {loss=}")
        self.log("train_loss", loss, logger=True, prog_bar=True)
        #print(f"{lb=} ?= {y_hat=} -> {self.train_accuracy(y_hat, lbs)}")
        self.log("train_acc", self.train_accuracy(y_hat, lbs), logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        lbs = torch.argmax(y, dim=0)
        loss = self.compute_loss(y_hat, y)
        self.log("valid_loss", loss, logger=True, prog_bar=True)
        #self.log("valid_acc", self.val_accuracy(y_hat, lbs), logger=True, prog_bar=False)
        #self.log("valid_f1", self.val_f1_score(y_hat, lbs), logger=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        #optimizer = AdaBound(self.parameters(), lr=self.learn_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learn_rate)
        #optimizer = Lion(self.parameters(), lr=self.learn_rate, weight_decay=1e-2)
        #optimizer = Adan(self.parameters(), lr=self.learn_rate, betas=(0.02, 0.08, 0.01), weight_decay=0.02)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6, verbose=True)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(
        #  optimizer, base_lr=self.learn_rate, max_lr=self.learn_rate * 5,
        #  step_size_up=5, cycle_momentum=False, mode="triangular2", verbose=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learn_rate * 5, steps_per_epoch=1, epochs=self.trainer.max_epochs)
        return [optimizer], [scheduler]





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
        onehot=[0,0,0,0,0,0]
        onehot[label_categorical]=1
        # Load the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(onehot).to(int)
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
        self.no_workers=16


    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)

    


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    net = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True, num_classes=6)
    lmodel = LitCancerSubtype(net=net, lr=1e-4)
    datamodule = UBCDataModule(batch_size=TRAIN_BATCH_SIZE,data_file=TRAIN_CSV,data_root=TRAIN_PROCESSED_DIR)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='valid_loss',
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