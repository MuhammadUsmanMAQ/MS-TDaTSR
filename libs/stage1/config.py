
""" 
    To check if a GPU is available
    we import torch for torch.device()
"""
import torch

""" 
    Parameters for stage 1
"""
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
epochs = 25
batch_size = 2
weight_decay = 3e-4

"""
    Path to Data Locating .csv
"""
data_dir = "/content/drive/MyDrive/Colab Notebooks/Research/MS-TDaSR/datasets/stage1/locate.csv"