import config
from train import test_on_epoch
from loss import TDLoss
from utils import get_data_loaders
from model import TDModel

from utils import get_data_loaders, load_checkpoint

import json
import argparse
import torch
import csv
import os
import pandas as pd
from data import Dataset
from torch.utils.data import DataLoader
from tabulate import tabulate

"""
    Evaluate Model
"""

def evaluate_model_split(saved_model, data_dir, threshold):
    model = TDModel(use_pretrained_model=True, basemodel_requires_grad=True)
    model = model.to(config.device)
    model = model.float()  # Convert from float16 to float32
    last_epoch, tr_metrics, te_metrics = load_checkpoint(
        torch.load(saved_model, map_location=config.device), model
    )

    _, test_loader = get_data_loaders(data_dir)
    metrics = test_on_epoch(test_loader, model, TDLoss(), threshold=threshold)

    return metrics


def evaluate_model_test(saved_model, test_dir, threshold):
    model = TDModel(use_pretrained_model=True, basemodel_requires_grad=True)
    model = model.to(config.device)
    model = model.float()  # Convert from float16 to float32
    last_epoch, tr_metrics, te_metrics = load_checkpoint(
        torch.load(saved_model, map_location=config.device), model
    )

    df = pd.read_csv(test_dir)
    test_dataset = Dataset(df, isTrain=False, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    metrics = test_on_epoch(test_loader, model, TDLoss(), threshold=threshold)

    return metrics


def print_metrics(metrics):
    print(
        f"Evaluation Metrics\n\
            Table Loss - Test: {metrics['table_loss']:.3f}\n\
            Table Accuracy - Test: {metrics['table_acc']:.3f}\n\
            Table F1 - Test: {metrics['table_f1']:.3f}\n\
            Table Precision - Test: {metrics['table_precision']:.3f}\n\
            Table Recall - Test: {metrics['table_recall']:.3f}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Metrics")
    parser.add_argument(
        "--output_dir", help="Dump metrics into a CSV file.", required=True
    )
    parser.add_argument("--model_dir", help="Load pretrained model.", required=True)
    parser.add_argument(
        "--data_dir",
        help="Path to complete data's locate.csv file. (Applies train/test split.)",
        required=False,
    )
    parser.add_argument(
        "--test_dir", help="Path to test data's locate.csv file. ", required=False
    )
    parser.add_argument(
        "--no_csv", type=bool, help="Do not write the CSV file.", required=False
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Confidence score for calculating metrics. (Default = 0.6)",
        required=False,
    )
    args = parser.parse_args()

    if args.threshold is None:
        threshold = 0.6
    else:
        threshold = args.threshold

    if args.data_dir is None and args.test_dir is None:
        parser.error("At least one of --data_dir and --test_dir is required.")

    saved_model = str(args.model_dir)
    data_dir = str(args.data_dir)
    test_dir = str(args.test_dir)
    output_dir = str(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)  # Create output directory
    tempdir = os.path.basename(saved_model)
    size = len(str(tempdir))
    newdir = tempdir[: size - 8]

    if args.data_dir is not None:
        data = evaluate_model_split(saved_model, data_dir, threshold)
    else:
        data = evaluate_model_test(saved_model, test_dir, threshold)

    r_data = dict()

    # Round off metrics to 3 decimal points
    for key in data:
        r_data[key] = round(data[key], 3)

    # Add title rows
    up_dict = {"Metric": "Value"}
    up_dict.update(r_data)

    csv_dir = os.path.join(output_dir, f"{newdir}_metrics.csv")

    if args.no_csv is None:
        # Write .csv file
        w = csv.writer(open(os.path.join(output_dir, f"{newdir}_metrics.csv"), "w"))
        for key, val in up_dict.items():
            w.writerow([key, val])

    elif args.no_csv is True:
        pass

    elif args.no_csv is False:
        # Write .csv file
        w = csv.writer(open(os.path.join(output_dir, f"{newdir}_metrics.csv"), "w"))
        for key, val in up_dict.items():
            w.writerow([key, val])

    print("\n")
    # Print data to console
    for key, val in up_dict.items():
        print("{:<20} {:<20}".format(key, val))
