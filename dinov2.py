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

import torch
from torchmetrics.classification import MulticlassConfusionMatrix
import warnings

from settings import *
from common import *

PIL.Image.MAX_IMAGE_PIXELS = 933120000

#dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

class LinearClassifierWrapper(nn.Module):
    def __init__(self, *, backbone: nn.Module, linear_head: nn.Module, layers: int = 4):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.layers = layers

    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14',pretrained=False)
        self.layers=4
        #self.transformer.train()
        self.linear_head = nn.Linear((1 + self.layers) * self.transformer.embed_dim, 6)
        # self.classifier = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 6)
        # )
        self.classifier=LinearClassifierWrapper(backbone=self.transformer, linear_head=self.linear_head, layers=self.layers)
    
    def forward(self, x):
        # x = self.transformer(x)
        # x = self.transformer.norm(x)
        x = self.classifier(x)
        return x


class DinoV2Model(pl.LightningModule):
    def __init__(self,learning_rate=8e-4):#,d_model, nhead, num_encoder_layers, num_decoder_layers,learning_rate=None):
        super(DinoV2Model, self).__init__()
        self.dinvov2 = DinoVisionTransformerClassifier()#RNA_Model()
        self.lr=learning_rate
        self.metric=MulticlassConfusionMatrix(num_classes=6)
        self.criteria=nn.CrossEntropyLoss()
        self.save_hyperparameters()
        
    def forward(self, src):
        output=self.dinvov2(src)
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
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
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


class UBCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_file:str,data_root:str):
        super().__init__()
        self.batch_size = batch_size
        self.train_file=data_file
        self.root_dir=data_root

    def setup(self, stage: Optional[str] = None) -> None:
        dataset=UBCDataset(csv_file=self.train_file, root_dir=self.root_dir)
        train_size = int(0.9 * len(dataset))
        val_size = (len(dataset) - train_size) 
        #test_size = len(dataset) - train_size - val_size
        self.no_workers=24
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        return  DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.no_workers,pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.no_workers,pin_memory=True)
    


if __name__ == "__main__":
    pl.seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    lmodel=DinoV2Model()
    datamodule = UBCDataModule(batch_size=TRAIN_BATCH_SIZE,data_file=TRAIN_CSV,data_root=TRAIN_PROCESSED_DIR)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='dinov2-model-epoch_{epoch:02d}_val_loss_{val_loss:.2f}',
        every_n_epochs=1,
        dirpath=MODEL_DIR_PREFIX
    )

    trainer = pl.Trainer(max_epochs=TRAIN_EPOCHS, callbacks=[checkpoint_callback],
        accelerator=ACCELERATION, devices=DEVICES, 
        strategy="fsdp")

    trainer.fit(lmodel, datamodule=datamodule)