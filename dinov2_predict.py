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
from torchmetrics.classification import MulticlassConfusionMatrix
import warnings
import PIL 
import gc

pl.seed_everything(42)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
PIL.Image.MAX_IMAGE_PIXELS = 999999933120000


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
        #self.linear_head = nn.Linear((1 + self.layers) * self.transformer.embed_dim, 6)
        self.linear_head = nn.Sequential(
            nn.Linear((1 + self.layers) * self.transformer.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )
        
        self.classifier=LinearClassifierWrapper(backbone=self.transformer, linear_head=self.linear_head, layers=self.layers)
    
    def forward(self, x):
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
        return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x) 
        loss = self.loss(outputs,y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)  
        loss =  self.loss(outputs,y)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        return loss
    def configure_optimizers(self):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"
print("Model is deploying into",device)
lmodel=DinoV2Model.load_from_checkpoint("../out/dinov2-model-epoch_epoch=12_val_loss_val_loss=1.38.ckpt",map_location=torch.device(device))


class UBCTestDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        #self.data = df.head(5)
        self.root_dir = root_dir
        # Get all file names in the root directory and subdirectories
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_id = self.data.iloc[idx, 0]
        img_path = self.root_dir+str(img_id) +".jpg"
        image = Image.open(img_path).convert('RGB')
        base_width=2240
        wpercent = (base_width / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
        return image,img_id
class ImageTiler():
    def __init__(self,tile_size=224):
        self.tile_size = tile_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def split_image(self, img):
        # Split the image into tiles of size self.tile_size x self.tile_size
        tiles = []
        width, height = img.size

        for i in range(0, width, self.tile_size):
            for j in range(0, height, self.tile_size):
                box = (i, j, i + self.tile_size, j + self.tile_size)

                # Adjust box coordinates for the last row and column
                box = (
                    box[0], box[1],
                    min(box[2], width), min(box[3], height)
                )

                tile = img.crop(box)
                if self.percent_black(tile)>90:
                    continue
                # If the tile size is smaller than specified, resize it
                if tile.size[0] < self.tile_size or tile.size[1] < self.tile_size:
                    tile = tile.resize((self.tile_size, self.tile_size), Image.BICUBIC)


                tiles.append(tile)

        return tiles

    def percent_black(self, img):
        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Count the number of black pixels (assumed to be [0, 0, 0] in RGB)
        black_pixels = np.sum((img_array == [0, 0, 0]).all(axis=-1))

        # Calculate the percentage of black pixels
        total_pixels = img_array.shape[0] * img_array.shape[1]
        percent_black = (black_pixels / total_pixels) * 100

        return percent_black
    
class TileDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        

        if self.transform:
            image = self.transform(image)

        return image

lmodel.eval()
test_dataset=UBCTestDataset("../train.csv","../train/")
tiler=ImageTiler()
class_labels=['CC','EC','HGSC','LGSC','MC','Other'] 
if device=='cpu':
    batch_size=16
else:
    batch_size = 1
submission_file=open("submission.csv","w")
submission_file.write('image_id,label\n')

test_no=0
for image,img_id in test_dataset:
    tiles=tiler.split_image(image)
    tile_dataset = TileDataset(tiles)
    data_loader = DataLoader(tile_dataset , batch_size=batch_size, shuffle=False)
    tiles_class_count=[0,0,0,0,0,0]
    for batch in data_loader:
        input_set=batch.to(device)
        logits=lmodel(input_set)
        probs = F.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probs, dim=1)
        print(predicted_classes)
        for pred_class in predicted_classes:
            tiles_class_count[pred_class]=tiles_class_count[pred_class]+1
    del data_loader
    del tile_dataset
    del tiles
    gc.collect()
    print(tiles_class_count)
    image_class = tiles_class_count.index(max(tiles_class_count))
    label=class_labels[image_class]
    submission_file.write(str(img_id)+','+label+'\n')
    print(img_id,label)
    if test_no<25:
        test_no=test_no+1
    else:
        break

submission_file.close()    
