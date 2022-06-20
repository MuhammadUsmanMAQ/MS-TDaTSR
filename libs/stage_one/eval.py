import config
from train import test_on_epoch
from loss import TDLoss
from utils import get_data_loaders
from model import TDModel

from utils import (
    get_data_loaders,
    load_checkpoint
)

import json
import argparse
import torch
import csv
import os

"""
    Evaluate Model
"""

def evaluate_model(saved_model, data_dir):
    model = TDModel(use_pretrained_model = True, basemodel_requires_grad = True)
    model = model.to(config.device)
    model = model.float()   # Convert from float16 to float32
    last_epoch, tr_metrics, te_metrics = load_checkpoint(torch.load(saved_model, map_location = config.device), model)

    _, test_loader = get_data_loaders(data_dir)
    metrics = test_on_epoch(test_loader, model, TDLoss(), threshold = 0.6)

    return metrics

def print_metrics(metrics):
    print(f"Evaluation Metrics\n\
            Table Loss - Test: {metrics['table_loss']:.3f}\n\
            Table Accuracy - Test: {metrics['table_acc']:.3f}\n\
            Table F1 - Test: {metrics['table_f1']:.3f}\n\
            Table Precision - Test: {metrics['table_precision']:.3f}\n\
            Table Recall - Test: {metrics['table_recall']:.3f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Evaluation Metrics')
    parser.add_argument("--output_dir", help = "Dump metrics into a CSV file.", required = True)
    parser.add_argument("--model_dir", help = "Load pretrained model.", required = True)
    parser.add_argument("--data_dir", help = "Path to locate.csv file.", required = True)
    args = parser.parse_args()
    
    saved_model = str(args.model_dir)
    data_dir = str(args.data_dir)
    output_dir = str(args.output_dir)

    os.makedirs(output_dir, exist_ok = True) # Create output directory

    data = evaluate_model(saved_model, data_dir)
    r_data = dict()

    # Round off metrics to 3 decimal points
    for key in data:
        r_data[key] = round(data[key], 3)

    # Add title rows
    up_dict = {"Metric": "Value"}
    up_dict.update(r_data)

    # Write .csv file
    w = csv.writer(open(os.path.join(output_dir, "metrics.csv"), "w"))
    for key, val in up_dict.items():
        w.writerow([key, val])