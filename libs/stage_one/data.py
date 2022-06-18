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
import config   # Contatining vars relevant to the Dataloader

"""
    Setting up the Dataset class in the way
        as per instructed in pytorch 
                documentation
"""
class MarmotDataset(nn.Module):
    def __init__(self, df, isTrain = True, transform = None):
        super(MarmotDataset, self).__init__()
        self.df = df

            # Normalizing the dataset
        if transform is None:
            self.transform = A.Compose([
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            max_pixel_value = 255,
                        ),
                        ToTensorV2()
                    ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path, table_mask_path = self.df.iloc[index, 0], self.df.iloc[index, 1]
        image = np.array(Image.open(os.path.join(config.base_dir, img_path)))
        table_mask = torch.FloatTensor(np.array(ImageOps.grayscale(Image.open(os.path.join(config.base_dir, table_mask_path))))/255.0).reshape(1,1024,768)

        image = self.transform(image = image)['image']

        return {"image": image, "table_mask": table_mask}


"""
    !python ___.py 
    read the processed csv file
    containing image locations and the respective masks
"""
if __name__ == '__main__':

    df = pd.read_csv(config.data_dir)
    train_data, test_data  = train_test_split(df, test_size = 0.2, random_state = config.seed)

    dataset = MarmotDataset(train_data)
    train_loader = DataLoader(dataset, batch_size = config.batch_size)

    for img_dict in train_loader:
        image, table_image = img_dict['image'], img_dict['table_mask']
        
        # Helps in knowing that data is being loaded correctly
        print(image.shape)                  
        print(table_image.shape)