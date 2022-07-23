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
from configs import config

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
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255,
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path, table_mask_path = self.df.iloc[index, 0], self.df.iloc[index, 1]
        image = np.array(Image.open(os.path.join(config.base_dir, img_path)))
        table_mask = (
            np.array(
                ImageOps.grayscale(
                    Image.open(os.path.join(config.base_dir, table_mask_path))
                )
            )
            / 255.0
        )
        table_mask = np.expand_dims(table_mask, axis=2)
        table_mask = torch.FloatTensor(table_mask).permute(2, 0, 1)

        image = self.transform(image=image)["image"]

        return {"image": image, "table_mask": table_mask}


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

    print("Loading first three images & masks.")

    for (i, img_dict) in zip(range(3), train_loader):  # Iterate only thrice
        image, table_image = img_dict["image"], img_dict["table_mask"]

        # Helps in knowing that data is being loaded correctly
        print("Image Shape: ", image.shape)
        print("Mask Shape: ", table_image.shape, "\n")
