""" 
    Common library imports for training
        a model through pytorch 
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config  # Contatining vars relevant to the Dataloader


def retTorchTensor(array):
    array = np.array(array)
    array = array[:, :, np.newaxis]
    array = torch.FloatTensor((np.array(array)) / 255.0).permute(2, 0, 1)

    return array


def collate_fn(batch):
    im = [item[0] for item in batch]
    mask_r = [item[1] for item in batch]
    mask_c = [item[2] for item in batch]
    mask_rh = [item[3] for item in batch]
    mask_ch = [item[4] for item in batch]
    mask_s = [item[5] for item in batch]

    return [im, mask_r, mask_c, mask_rh, mask_ch, mask_s]

"""
    Setting up the Dataset class in the way
        as per instructed in pytorch 
                documentation
"""

class Dataset(nn.Module):
    def __init__(self, df, isTrain=True, transform=None):
        super(Dataset, self).__init__()
        self.df = df

        # Normalizing the dataset
        if transform is None:
            self.transform = A.Compose(
                [
                    A.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        max_pixel_value=255,
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        (
            img_path,
            table_row,
            table_column,
            table_row_header,
            table_column_header,
            table_spanning,
        ) = (
            self.df.iloc[index, 0],
            self.df.iloc[index, 1],
            self.df.iloc[index, 2],
            self.df.iloc[index, 3],
            self.df.iloc[index, 4],
            self.df.iloc[index, 5],
        )

        image = np.array(Image.open(os.path.join(config.base_dir, img_path)))

        table_row = ImageOps.grayscale(
            Image.open(os.path.join(config.base_dir, table_row))
        )
        table_row = retTorchTensor(table_row)

        table_column = ImageOps.grayscale(
            Image.open(os.path.join(config.base_dir, table_column))
        )
        table_column = retTorchTensor(table_column)

        table_row_header = ImageOps.grayscale(
            Image.open(os.path.join(config.base_dir, table_row_header))
        )
        table_row_header = retTorchTensor(table_row_header)

        table_column_header = ImageOps.grayscale(
            Image.open(os.path.join(config.base_dir, table_column_header))
        )
        table_column_header = retTorchTensor(table_column_header)

        table_spanning = ImageOps.grayscale(
            Image.open(os.path.join(config.base_dir, table_spanning))
        )
        table_spanning = retTorchTensor(table_spanning)

        image = self.transform(image=image)["image"]

        return [
            image,
            table_row,
            table_column,
            table_row_header,
            table_column_header,
            table_spanning,
        ]


"""
    !python ___.py 
    read the processed csv file
    containing image locations and the respective masks
"""
if __name__ == "__main__":

    df = pd.read_csv(config.data_dir)
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=config.seed
    )

    dataset = Dataset(train_data)
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for img_list in train_loader:
        (
            image,
            table_row,
            table_column,
            table_row_header,
            table_column_header,
            table_spanning,
        ) = (
            img_list[0],
            img_list[1],
            img_list[2],
            img_list[3],
            img_list[4],
            img_list[5],
        )

        for i in image:  # To verify loading proc
            print(i.shape)
