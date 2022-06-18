
""" 
    To check if a GPU is available
    we import torch for torch.device()
"""
import torch
from datetime import datetime

""" 
    Parameters for stage 1
"""
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
epochs = 25
batch_size = 2
weight_decay = 3e-4
base_dir = 'DIR_TO_BE_CONFIGURED'
run_id = datetime.now().strftime('%Y_%m_%d_%H%M%S')

"""
    Available Options:
    Encoder: ResNet-50, ResNet-101, ConvNext-tiny, ConvNext-small, ConvNext-base, ConvNext-large
    Decoder: CNDecoder, RNDecoder
"""
encoder = 'ResNet-50'
decoder = 'RNDecoder'

"""
    Base Directory to locate images/masks
"""
data_dir = f"{base_dir}/datasets/stage_one/locate.csv"