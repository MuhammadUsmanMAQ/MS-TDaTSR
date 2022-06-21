import config
from train import test_on_epoch
from loss import TDLoss
from model import TDModel

from utils import (
    load_checkpoint,
    save_fig,
    get_TableMasks
)

import json
import argparse
import torch
import csv
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datetime import datetime

"""
    Evaluate Model
"""

def load_model(saved_model):
    model = TDModel(use_pretrained_model = True, basemodel_requires_grad = True)
    model = model.to(config.device)
    model = model.float()   # Convert from float16 to float32
    last_epoch, tr_metrics, te_metrics = load_checkpoint(torch.load(saved_model, map_location = config.device), model)

    return model

def res_with_gt(path_to_img, path_to_gt, path_to_out):
    save_id = datetime.now().strftime('%d_%M%S')
    out = os.path.join(path_to_out, save_id)
    os.makedirs(out, exist_ok = True)

    test_img = np.array(Image.open(str(path_to_img)))
    test_table = np.array(Image.open(str(path_to_gt)))

    save_fig(test_img, test_table, str(out), save_id, 'groundtruth', title = 'Original')
    table_out = get_TableMasks(test_img, model)
    save_fig(test_img, np.squeeze(table_out), str(out), save_id, 'test', title = '')

def res_without_gt(path_to_img, path_to_out):
    save_id = datetime.now().strftime('%d_%M%S')
    out = os.path.join(path_to_out, save_id)
    os.makedirs(out, exist_ok = True)

    test_img = np.array(Image.open(str(path_to_img)))

    table_out = get_TableMasks(test_img, model)
    save_fig(test_img, np.squeeze(table_out), str(out), save_id, 'test', title = '')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluation Metrics')
    parser.add_argument("--input_img", help = "Load an input image.", required = True)
    parser.add_argument("--gt_dir", help = "Load ground truth mask.", required = False)
    parser.add_argument("--model_dir", help = "Load pretrained model.", required = True)
    parser.add_argument("--output_dir", help = "Path to directory where masks/bbox will be saved.", required = True)
    args = parser.parse_args()
    
    saved_model = str(args.model_dir)
    img_dir = str(args.input_img)
    output_dir = str(args.output_dir)

    plt.rcParams["figure.figsize"] = (15,15)
    model = load_model(saved_model)

    if args.gt_dir is None:
        res_without_gt(img_dir, output_dir)
    else:
        gt_dir = str(args.gt_dir)
        res_with_gt(img_dir, gt_dir, output_dir)