import time
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from configs import config
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
from utils.loss import TDLoss
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from architectures.model import TDModel
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
import argparse

"""
    Common train functions imported/edited from TabNet-pytorch Git repo
    https://github.com/asagar60/TableNet-pytorch/blob/main/Training/train.py
"""


def train_on_epoch(data_loader, model, optimizer, loss, scaler, threshold=0.5):

    combined_loss = []
    table_loss, table_acc, table_precision, table_recall, table_f1 = [], [], [], [], []
    loop = tqdm(data_loader, leave=True)

    for batch_idx, img_dict in enumerate(loop):
        image = img_dict["image"].to(config.device)
        table_image = img_dict["table_mask"].to(config.device)

        with torch.cuda.amp.autocast():
            table_out = model(image)
            t_loss = loss(table_out, table_image)

        table_loss.append(t_loss.item())

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(t_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(table_loss) / len(table_loss)
        loop.set_postfix(loss=mean_loss)

        cal_metrics_table = compute_metrics(table_image, table_out, threshold)

        table_f1.append(cal_metrics_table["f1"])
        table_precision.append(cal_metrics_table["precision"])
        table_acc.append(cal_metrics_table["acc"])
        table_recall.append(cal_metrics_table["recall"])

        metrics = {
            "table_loss": np.mean(table_loss),
            "table_f1": np.mean(table_f1),
            "table_precision": np.mean(table_precision),
            "table_recall": np.mean(table_recall),
            "table_acc": np.mean(table_acc),
        }

    return metrics


def test_on_epoch(data_loader, model, loss, threshold=0.5):

    combined_loss = []
    table_loss, table_acc, table_precision, table_recall, table_f1 = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        loop = tqdm(data_loader, leave=True)

        for batch_idx, img_dict in enumerate(loop):
            image = img_dict["image"].to(config.device)
            table_image = img_dict["table_mask"].to(config.device)

            image = image.float()
            table_image = table_image.float()

            with torch.cuda.amp.autocast():
                table_out = model(image)
                t_loss = loss(table_out, table_image)

            table_loss.append(t_loss.item())

            mean_loss = sum(table_loss) / len(table_loss)
            loop.set_postfix(loss=mean_loss)

            cal_metrics_table = compute_metrics(table_image, table_out, threshold)

            table_f1.append(cal_metrics_table["f1"])
            table_precision.append(cal_metrics_table["precision"])
            table_acc.append(cal_metrics_table["acc"])
            table_recall.append(cal_metrics_table["recall"])

    metrics = {
        "table_loss": np.mean(table_loss),
        "table_f1": np.mean(table_f1),
        "table_precision": np.mean(table_precision),
        "table_recall": np.mean(table_recall),
        "table_acc": np.mean(table_acc),
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
        checkpoint_name = f"{config.base_dir}/models/{config.encoder}_{config.decoder}/{config.run_id}_checkpoint.pth.tar"

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
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
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
        last_table_f1 = te_metrics["table_f1"]

        print("Loading Checkpoint")
        display_metrics(last_epoch, tr_metrics, te_metrics)
        print()

    else:
        last_epoch = 0
        last_table_f1 = 0.0

    # Train Network
    print("Training Model\n")
    writer = SummaryWriter(
        f"{config.base_dir}/models/{config.encoder}_{config.decoder}/{config.run_id}_train"
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

        if last_table_f1 < te_metrics["table_f1"]:

            last_table_f1 = te_metrics["table_f1"]
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_metrics": tr_metrics,
                "test_metrics": te_metrics,
            }
            save_checkpoint(checkpoint, checkpoint_name)
