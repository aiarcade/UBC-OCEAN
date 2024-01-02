import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from multiprocessing import Process
import PIL
from settings import *
import numpy as np
PIL.Image.MAX_IMAGE_PIXELS = 933120000

class TiledImageDataset(Dataset):
    def __init__(self, data, root_dir, tile_size=256):
        self.data = data
        self.root_dir = root_dir
        self.tile_size = tile_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imid = self.data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, str(imid) + ".jpg")
        label = self.data.iloc[idx, 1]
        img = Image.open(img_name).convert('RGB')
        tiles = self.split_image(img)
        return tiles, label, imid

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
                if self.percent_black(tile)>80:
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

def save_tiles(dataset):
    for tiles, label, id in dataset:
        label_dir = os.path.join(output_dir, label)
        # Create directory for the label if it doesn't exist
        os.makedirs(label_dir, exist_ok=True)
        for i, tile in enumerate(tiles):
            tile.save(os.path.join(label_dir, str(id) + "_" + str(i) + ".png"))

def process_chunk(df_chunk):
    dataset = TiledImageDataset(df_chunk, root_dir=images_root_dir, tile_size=tile_size)
    save_tiles(dataset)

if __name__ == "__main__":
    csv_file_path = TRAIN_CSV
    images_root_dir = TRAIN_RAW_DIR
    output_dir = TRAIN_PROCESSED_DIR
    tile_size = 224

    df = pd.read_csv(csv_file_path)
    total_rows = len(df)

    # Calculate the approximate number of rows per split
    rows_per_split = total_rows // 50

    # Initialize an empty list to store the DataFrames
    df_splits = []

    # Split the DataFrame into 50 pieces row-wise
    start_idx = 0
    for _ in range(50):
        end_idx = start_idx + rows_per_split
        df_split = df.iloc[start_idx:end_idx, :]
        df_splits.append(df_split)
        start_idx = end_idx

    # If there are remaining rows, add them to the last split
    if start_idx < total_rows:
        df_split = df.iloc[start_idx:, :]
        df_splits[-1] = pd.concat([df_splits[-1], df_split])

    print(len(df_splits))
    processes = []

    for df_p in df_splits:
        print("Starting for ", len(df_p))
        process_c = Process(target=process_chunk, args=(df_p,))
        process_c.start()
        processes.append(process_c)

    for p in processes:
        p.join()
