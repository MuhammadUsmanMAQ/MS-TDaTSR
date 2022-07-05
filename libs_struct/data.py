""" 
    Common library imports for training
        a model through pytorch 
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded
from albumentations.pytorch import ToTensorV2
import config  # Contatining vars relevant to the Dataloader


def padding(img, expected_size):
    desired_width, desired_height = expected_size
    delta_width = desired_width - img.size[0]
    delta_height = desired_height - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )

    return np.array(ImageOps.expand(img, padding))


def retTorchTensor(array):
    array = np.array(array)

    if array.shape[1] > 384 and array.shape[0] > 512:
        array = cv2.resize(array, (384, 512), interpolation=cv2.INTER_AREA)

    elif array.shape[0] > 512:
        array = cv2.resize(array, (array.shape[1], 512), interpolation=cv2.INTER_AREA)

    elif array.shape[1] > 384:
        array = cv2.resize(array, (384, array.shape[0]), interpolation=cv2.INTER_AREA)

    else:
        pass

    array = padding(Image.fromarray(array), (512, 384))
    array = array[:, :, np.newaxis]
    array = torch.FloatTensor((np.array(array)) / 255.0).permute(2, 0, 1)

    return array


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
                    # PadIfNeeded(384, 512, cv2.BORDER_CONSTANT, 0),
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

        image = np.array(image)

        if image.shape[1] > 384 and image.shape[0] > 512:
            image = cv2.resize(image, (384, 512), interpolation=cv2.INTER_AREA)

        elif image.shape[0] > 512:
            image = cv2.resize(
                image, (image.shape[1], 512), interpolation=cv2.INTER_AREA
            )

        elif image.shape[1] > 384:
            image = cv2.resize(
                image, (384, image.shape[0]), interpolation=cv2.INTER_AREA
            )

        else:
            pass

        image = padding(Image.fromarray(image), (512, 384))

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
    train_loader = DataLoader(dataset, batch_size=config.batch_size)

    for i, img_list in zip(range(3), train_loader):
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

        print("Image: ", image.shape)
        print("Mask: ", table_row.shape)
        print("Mask: ", table_column.shape)
        print("Mask: ", table_row_header.shape)
        print("Mask: ", table_column_header.shape)
        print("Mask: ", table_spanning.shape)
        print("\n")
        if i == 3:
            break
