import config
from train import test_on_epoch
from loss import TDLoss
from utils import get_data_loaders
from model import TDModel

from utils import (
    get_data_loaders,
    load_checkpoint,
    save_fig,
    save_fig_gt,
    get_masks,
    get_bbox,
)

import json
import argparse
import torch
import csv
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm

"""
    Inference on Batches
"""

def load_model(saved_model):
    model = TDModel(use_pretrained_model=True, basemodel_requires_grad=True)
    model = model.to(config.device)
    model = model.float()  # Convert from float16 to float32
    last_epoch, tr_metrics, te_metrics = load_checkpoint(
        torch.load(saved_model, map_location=config.device), model
    )

    return model


def res_with_gt(model, path_to_img, path_to_gt, path_to_out, run_id):
    save_id = "batch_pred_" + str(run_id)
    out = os.path.join(path_to_out, save_id)
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(path_to_img)

    test_img = np.array(Image.open(str(path_to_img)).convert("RGB"))
    test_table = np.array(Image.open(str(path_to_gt)))
    table_out = get_masks(test_img, model)

    image, table_boundRect = get_bbox(test_img, table_out)

    # draw bounding boxes of Table Coordinates
    color = (0, 255, 0)
    thickness = 3

    bbox_image = test_img.copy()

    for x, y, w, h in table_boundRect:
        bbox_image = cv2.rectangle(bbox_image, (x, y), (x + w, y + h), color, thickness)

    save_fig_gt(
        test_img, test_table, np.squeeze(table_out), bbox_image, str(out), "p_" + name
    )


def res_without_gt(model, path_to_img, path_to_out, run_id):
    save_id = "batch_pred_" + str(run_id)
    out = os.path.join(path_to_out, save_id)
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(path_to_img)

    test_img = np.array(Image.open(str(path_to_img)).convert("RGB"))
    table_out = get_masks(test_img, model)

    outputs = get_bbox(test_img, table_out)
    _, table_boundRect = outputs

    # draw bounding boxes of Table Coordinates
    color = (0, 255, 0)
    thickness = 3

    bbox_image = test_img.copy()

    for x, y, w, h in table_boundRect:
        bbox_image = cv2.rectangle(bbox_image, (x, y), (x + w, y + h), color, thickness)

    save_fig(test_img, np.squeeze(table_out), bbox_image, str(out), "p_" + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Metrics")
    parser.add_argument("--test_dir", help="Path to test images.", required=True)
    parser.add_argument("--gt_dir", help="Path to ground truth masks.", required=False)
    parser.add_argument("--model_dir", help="Load pretrained model.", required=True)
    parser.add_argument(
        "--output_dir",
        help="Path to directory where masks/bbox will be saved.",
        required=True,
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%M%S")
    saved_model = str(args.model_dir)
    test_dir = str(args.test_dir)
    output_dir = str(args.output_dir)

    iter_list = os.listdir(test_dir)

    plt.rcParams["figure.figsize"] = (15, 15)
    model = load_model(saved_model)

    if args.gt_dir is None:
        for i in iter_list:
            print("Inferring on " + i)
            td = os.path.join(test_dir, i)
            res_without_gt(model, td, output_dir, str(run_id))

    else:
        for i in iter_list:
            print("Inferring on " + i)
            td = os.path.join(test_dir, i)
            gd = os.path.join(args.gt_dir, i)
            res_with_gt(model, td, gd, output_dir, str(run_id))
