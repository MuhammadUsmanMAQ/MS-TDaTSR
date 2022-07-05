import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import config
import os
from termcolor import colored

from utils import (
    get_data_loaders,
    load_checkpoint,
    save_checkpoint,
    display_metrics,
    write_summary,
    compute_metrics,
    seed_all,
)

from loss import TDLoss
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from model import TDModel
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

"""
    Common train functions imported/edited from TabNet-pytorch Git repo
    https://github.com/asagar60/TableNet-pytorch/blob/main/Training/train.py
"""


def train_on_epoch(data_loader, model, optimizer, loss, scaler, threshold=0.5):

    combined_loss = []
    tr_loss, tr_acc, tr_precision, tr_recall, tr_f1 = [], [], [], [], []
    tc_loss, tc_acc, tc_precision, tc_recall, tc_f1 = [], [], [], [], []
    trh_loss, trh_acc, trh_precision, trh_recall, trh_f1 = [], [], [], [], []
    tch_loss, tch_acc, tch_precision, tch_recall, tch_f1 = [], [], [], [], []
    ts_loss, ts_acc, ts_precision, ts_recall, ts_f1 = [], [], [], [], []

    loop = tqdm(data_loader, leave=True)

    for batch_idx, img_list in enumerate(loop):
        image = img_list[0].to(config.device)
        table_row = img_list[1].to(config.device)
        table_column = img_list[2].to(config.device)
        table_row_header = img_list[3].to(config.device)
        table_column_header = img_list[4].to(config.device)
        table_spanning = img_list[5].to(config.device)

        with torch.cuda.amp.autocast():
            tr_out, tc_out, trh_out, tch_out, ts_out = model(image)

            tr_l = loss(tr_out, table_row)
            tc_l = loss(tc_out, table_column)
            trh_l = loss(trh_out, table_row_header)
            tch_l = loss(tch_out, table_column_header)
            ts_l = loss(ts_out, table_spanning)

        tr_loss.append(tr_l.item())
        tc_loss.append(tc_l.item())
        trh_loss.append(trh_l.item())
        tch_loss.append(tch_l.item())
        ts_loss.append(ts_l.item())

        combined_loss.append((tr_l + tc_l + trh_l + tch_l + ts_l).item())

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(tr_l + tc_l + trh_l + tch_l + ts_l).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(combined_loss) / len(combined_loss)
        loop.set_postfix(loss=mean_loss)

        cal_metrics_tr = compute_metrics(tr_out, table_row, threshold)
        cal_metrics_tc = compute_metrics(tc_out, table_column, threshold)
        cal_metrics_trh = compute_metrics(trh_out, table_row_header, threshold)
        cal_metrics_tch = compute_metrics(tch_out, table_column_header, threshold)
        cal_metrics_ts = compute_metrics(ts_out, table_spanning, threshold)

        tr_f1.append(cal_metrics_tr["f1"])
        tr_precision.append(cal_metrics_tr["precision"])
        tr_acc.append(cal_metrics_tr["acc"])
        tr_recall.append(cal_metrics_tr["recall"])

        tc_f1.append(cal_metrics_tc["f1"])
        tc_precision.append(cal_metrics_tc["precision"])
        tc_acc.append(cal_metrics_tc["acc"])
        tc_recall.append(cal_metrics_tc["recall"])

        trh_f1.append(cal_metrics_trh["f1"])
        trh_precision.append(cal_metrics_trh["precision"])
        trh_acc.append(cal_metrics_trh["acc"])
        trh_recall.append(cal_metrics_trh["recall"])

        tch_f1.append(cal_metrics_tch["f1"])
        tch_precision.append(cal_metrics_tch["precision"])
        tch_acc.append(cal_metrics_tch["acc"])
        tch_recall.append(cal_metrics_tch["recall"])

        ts_f1.append(cal_metrics_ts["f1"])
        ts_precision.append(cal_metrics_ts["precision"])
        ts_acc.append(cal_metrics_ts["acc"])
        ts_recall.append(cal_metrics_ts["recall"])

        metrics = {
            "tr_loss": np.mean(tr_loss),
            "tr_f1": np.mean(tr_f1),
            "tr_precision": np.mean(tr_precision),
            "tr_recall": np.mean(tr_recall),
            "tr_acc": np.mean(tr_acc),
            "tc_loss": np.mean(tc_loss),
            "tc_f1": np.mean(tc_f1),
            "tc_precision": np.mean(tc_precision),
            "tc_recall": np.mean(tc_recall),
            "tc_acc": np.mean(tc_acc),
            "trh_loss": np.mean(trh_loss),
            "trh_f1": np.mean(trh_f1),
            "trh_precision": np.mean(trh_precision),
            "trh_recall": np.mean(trh_recall),
            "trh_acc": np.mean(trh_acc),
            "tch_loss": np.mean(tch_loss),
            "tch_f1": np.mean(tch_f1),
            "tch_precision": np.mean(tch_precision),
            "tch_recall": np.mean(tch_recall),
            "tch_acc": np.mean(tch_acc),
            "ts_loss": np.mean(ts_loss),
            "ts_f1": np.mean(ts_f1),
            "ts_precision": np.mean(ts_precision),
            "ts_recall": np.mean(ts_recall),
            "ts_acc": np.mean(ts_acc),
        }

    return metrics


def test_on_epoch(data_loader, model, loss, threshold=0.5):

    combined_loss = []
    tr_loss, tr_acc, tr_precision, tr_recall, tr_f1 = [], [], [], [], []
    tc_loss, tc_acc, tc_precision, tc_recall, tc_f1 = [], [], [], [], []
    trh_loss, trh_acc, trh_precision, trh_recall, trh_f1 = [], [], [], [], []
    tch_loss, tch_acc, tch_precision, tch_recall, tch_f1 = [], [], [], [], []
    ts_loss, ts_acc, ts_precision, ts_recall, ts_f1 = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        loop = tqdm(data_loader, leave=True)

        for batch_idx, img_list in enumerate(loop):
            image = img_list[0].to(config.device)
            table_row = img_list[1].to(config.device)
            table_column = img_list[2].to(config.device)
            table_row_header = img_list[3].to(config.device)
            table_column_header = img_list[4].to(config.device)
            table_spanning = img_list[5].to(config.device)

            image = image.float()
            table_image = table_image.float()

            with torch.cuda.amp.autocast():
                tr_out, tc_out, trh_out, tch_out, ts_out = model(image)

                tr_l = loss(tr_out, table_row)
                tc_l = loss(tc_out, table_column)
                trh_l = loss(trh_out, table_row_header)
                tch_l = loss(tch_out, table_column_header)
                ts_l = loss(ts_out, table_spanning)

            tr_loss.append(tr_l.item())
            tc_loss.append(tc_l.item())
            trh_loss.append(trh_l.item())
            tch_loss.append(tch_l.item())
            ts_loss.append(ts_l.item())

            combined_loss.append((tr_l + tc_l + trh_l + tch_l + ts_l).item())

            mean_loss = sum(combined_loss) / len(combined_loss)
            loop.set_postfix(loss=mean_loss)

            cal_metrics_tr = compute_metrics(tr_out, table_row, threshold)
            cal_metrics_tc = compute_metrics(tc_out, table_column, threshold)
            cal_metrics_trh = compute_metrics(trh_out, table_row_header, threshold)
            cal_metrics_tch = compute_metrics(tch_out, table_column_header, threshold)
            cal_metrics_ts = compute_metrics(ts_out, table_spanning, threshold)

            tr_f1.append(cal_metrics_tr["f1"])
            tr_precision.append(cal_metrics_tr["precision"])
            tr_acc.append(cal_metrics_tr["acc"])
            tr_recall.append(cal_metrics_tr["recall"])

            tc_f1.append(cal_metrics_tc["f1"])
            tc_precision.append(cal_metrics_tc["precision"])
            tc_acc.append(cal_metrics_tc["acc"])
            tc_recall.append(cal_metrics_tc["recall"])

            trh_f1.append(cal_metrics_trh["f1"])
            trh_precision.append(cal_metrics_trh["precision"])
            trh_acc.append(cal_metrics_trh["acc"])
            trh_recall.append(cal_metrics_trh["recall"])

            tch_f1.append(cal_metrics_tch["f1"])
            tch_precision.append(cal_metrics_tch["precision"])
            tch_acc.append(cal_metrics_tch["acc"])
            tch_recall.append(cal_metrics_tch["recall"])

            ts_f1.append(cal_metrics_ts["f1"])
            ts_precision.append(cal_metrics_ts["precision"])
            ts_acc.append(cal_metrics_ts["acc"])
            ts_recall.append(cal_metrics_ts["recall"])

    metrics = {
        "tr_loss": np.mean(tr_loss),
        "tr_f1": np.mean(tr_f1),
        "tr_precision": np.mean(tr_precision),
        "tr_recall": np.mean(tr_recall),
        "tr_acc": np.mean(tr_acc),
        "tc_loss": np.mean(tc_loss),
        "tc_f1": np.mean(tc_f1),
        "tc_precision": np.mean(tc_precision),
        "tc_recall": np.mean(tc_recall),
        "tc_acc": np.mean(tc_acc),
        "trh_loss": np.mean(trh_loss),
        "trh_f1": np.mean(trh_f1),
        "trh_precision": np.mean(trh_precision),
        "trh_recall": np.mean(trh_recall),
        "trh_acc": np.mean(trh_acc),
        "tch_loss": np.mean(tch_loss),
        "tch_f1": np.mean(tch_f1),
        "tch_precision": np.mean(tch_precision),
        "tch_recall": np.mean(tch_recall),
        "tch_acc": np.mean(tch_acc),
        "ts_loss": np.mean(ts_loss),
        "ts_f1": np.mean(ts_f1),
        "ts_precision": np.mean(ts_precision),
        "ts_recall": np.mean(ts_recall),
        "ts_acc": np.mean(ts_acc),
    }

    model.train()
    return metrics


"""
    Configure for Model Params/Architecture
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Metrics")
    parser.add_argument("--resume", help="Resume training a model.", required=False)
    args = parser.parse_args()

    seed_all(SEED_VALUE=config.seed)

    if args.resume is not None:
        checkpoint_name = str(args.resume)
    else:
        checkpoint_name = f"{config.base_dir}/models/{config.encoder}_{config.decoder}_SR/{config.run_id}_checkpoint.pth.tar"

    model = TDModel(use_pretrained_model=True, basemodel_requires_grad=True)

    print(colored("Model Architecture and Trainable Paramerters", "green"))
    print(colored("=" * 45, "green"))
    print(
        colored(
            summary(
                model,
                torch.zeros((1, 3, 1024, 768)),
                show_input=False,
                show_hierarchical=True,
            ),
            "green",
        )
    )

    model = model.to(config.device)
    optimizer = optim.NAdam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    loss = TDLoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader = get_data_loaders(data_path=config.data_dir)

    nl = "\n"

    # load checkpoint
    if os.path.exists(checkpoint_name):
        last_epoch, tr_metrics, te_metrics = load_checkpoint(
            torch.load(checkpoint_name), model, optimizer
        )
        last_tr_f1 = te_metrics["tr_f1"]
        last_tc_f1 = te_metrics["tc_f1"]
        last_trh_f1 = te_metrics["trh_f1"]
        last_tch_f1 = te_metrics["tch_f1"]
        last_ts_f1 = te_metrics["ts_f1"]

        print("Loading Checkpoint")
        display_metrics(last_epoch, tr_metrics, te_metrics)
        print()

    else:
        last_epoch = 0
        last_tr_f1 = 0
        last_tc_f1 = 0
        last_trh_f1 = 0
        last_tch_f1 = 0
        last_ts_f1 = 0

    # Train Network
    print("Training Model\n")
    writer = SummaryWriter(
        f"{config.base_dir}/models/{config.encoder}_{config.decoder}_SR/{config.run_id}_train"
    )

    # for early stopping
    i = 0

    for epoch in range(last_epoch + 1, config.epochs):
        print("=" * 30)
        start = time.time()

        tr_metrics = train_on_epoch(train_loader, model, optimizer, loss, scaler, 0.6)
        te_metrics = test_on_epoch(test_loader, model, loss, threshold=0.6)

        write_summary(writer, tr_metrics, te_metrics, epoch)

        end = time.time()

        display_metrics(epoch, tr_metrics, te_metrics)

        if last_tr_f1 < te_metrics["tr_f1"] or last_tc_f1 < te_metrics["tc_f1"]:

            last_tr_f1 = te_metrics["tr_f1"]
            last_tc_f1 = te_metrics["tc_f1"]
            last_trh_f1 = te_metrics["trh_f1"]
            last_tch_f1 = te_metrics["tch_f1"]
            last_ts_f1 = te_metrics["ts_f1"]

            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_metrics": tr_metrics,
                "test_metrics": te_metrics,
            }
            save_checkpoint(checkpoint, checkpoint_name)
