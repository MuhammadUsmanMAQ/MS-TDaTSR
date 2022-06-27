import torch
import random
from PIL import Image
import numpy as np
import os
import pandas as pd
import config
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
    Utility functions imported/edited from TabNet-pytorch Git repo
    https://github.com/asagar60/TableNet-pytorch/blob/main/Training/utils.py
"""

TRANSFORM = A.Compose(
    [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ]
)


def seed_all(SEED_VALUE=config.seed):
    random.seed(SEED_VALUE)
    os.environ["PYTHONHASHSEED"] = str(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_data_loaders(data_path=config.data_dir):
    df = pd.read_csv(data_path)
    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=config.seed
    )

    train_dataset = Dataset(train_data, isTrain=True, transform=None)
    test_dataset = Dataset(test_data, isTrain=False, transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# Checkpoint
def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint Saved at ", filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    last_epoch = checkpoint["epoch"]
    tr_metrics = checkpoint["train_metrics"]
    te_metrics = checkpoint["test_metrics"]
    print("Model loaded successfully.\n")

    return last_epoch, tr_metrics, te_metrics


def write_summary(writer, tr_metrics, te_metrics, epoch):
    writer.add_scalar("Table loss/Train", tr_metrics["table_loss"], global_step=epoch)
    writer.add_scalar("Table loss/Test", te_metrics["table_loss"], global_step=epoch)

    writer.add_scalar("Table Acc/Train", tr_metrics["table_acc"], global_step=epoch)
    writer.add_scalar("Table Acc/Test", te_metrics["table_acc"], global_step=epoch)

    writer.add_scalar("Table F1/Train", tr_metrics["table_f1"], global_step=epoch)
    writer.add_scalar("Table F1/Test", te_metrics["table_f1"], global_step=epoch)

    writer.add_scalar(
        "Table Precision/Train", tr_metrics["table_precision"], global_step=epoch
    )
    writer.add_scalar(
        "Table Precision/Test", te_metrics["table_precision"], global_step=epoch
    )

    writer.add_scalar(
        "Table Recall/Train", tr_metrics["table_recall"], global_step=epoch
    )
    writer.add_scalar(
        "Table Recall/Test", te_metrics["table_recall"], global_step=epoch
    )


def display_metrics(epoch, tr_metrics, te_metrics):
    nl = "\n"

    print(
        f"Epoch: {epoch} {nl}\
            Table Loss -- Train: {tr_metrics['table_loss']:.3f} Test: {te_metrics['table_loss']:.3f}{nl}\
            Table Acc -- Train: {tr_metrics['table_acc']:.3f} Test: {te_metrics['table_acc']:.3f}{nl}\
            Table F1 -- Train: {tr_metrics['table_f1']:.3f} Test: {te_metrics['table_f1']:.3f}{nl}\
            Table Precision -- Train: {tr_metrics['table_precision']:.3f} Test: {te_metrics['table_precision']:.3f}{nl}\
            Table Recall -- Train: {tr_metrics['table_recall']:.3f} Test: {te_metrics['table_recall']:.3f}{nl}\
        "
    )


"""
    Metrics to compare proposed model with well-performing models
"""

def compute_metrics(ground_truth, prediction, threshold=0.5):
    ground_truth = ground_truth.int()
    prediction = (torch.sigmoid(prediction) > threshold).int()

    TP = torch.sum(prediction[ground_truth == 1] == 1)
    TN = torch.sum(prediction[ground_truth == 0] == 0)
    FP = torch.sum(prediction[ground_truth == 1] == 0)
    FN = torch.sum(prediction[ground_truth == 0] == 1)

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (FP + TP + 1e-4)
    recall = TP / (FN + TP + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)

    metrics = {
        "acc": acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }

    return metrics


def display(img, table, title="Original"):
    f, ax = plt.subplots(1, 2, figsize=(20, 20))
    ax[0].imshow(img)
    ax[0].set_title(f"{title} Image")
    ax[1].imshow(table)
    ax[1].set_title(f"{title} Table Mask")
    plt.show()


def save_fig_gt(orig_img, orig_mask, table_mask, bbox_image, out_dir, n):
    out_dir = os.path.join(out_dir, n)

    fig = plt.figure(figsize=(20, 20))  # Notice the equal aspect ratio

    ax = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    show_list = [orig_img, orig_mask, table_mask, bbox_image]
    title_list = ["Original Image", "Original Mask", "Predicted Mask", "Predicted BBox"]

    for a, b, c in zip(ax, show_list, title_list):
        a.set_aspect("auto")
        a.imshow(b)
        a.set_title(c)

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(out_dir, bbox_inches="tight")


def save_fig(orig_img, table_mask, bbox_image, out_dir, n):
    out_dir = os.path.join(out_dir, n)

    f, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(orig_img)
    ax[0].set_title(f"Original Image")

    ax[1].imshow(bbox_image, cmap="gray")
    ax[1].set_title(f"Predicted BBox")

    plt.savefig(out_dir, bbox_inches="tight")


def get_masks(test_img, model, transform=TRANSFORM, device=config.device):
    image = transform(image=test_img)["image"]
    # get predictions
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        table_out = model(image)
        table_out = torch.sigmoid(table_out)

    # remove gradients
    table_out = (
        table_out.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5
    ).astype(int)
    table_out = table_out.reshape(1024, 768).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    table_out = cv2.erode(table_out, kernel, iterations=2)
    table_out = cv2.dilate(table_out, kernel, iterations=1)

    return table_out


def is_contour_bad(c):
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    return not len(approx) == 4


def get_bbox(image, table_mask):
    table_mask = table_mask.reshape(1024, 768).astype(np.uint8)
    image = image[..., 0].reshape(1024, 768).astype(np.uint8)

    contours, table_heirarchy = cv2.findContours(
        table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    table_contours = []

    for c in contours:
        if cv2.contourArea(c) > 2000:
            table_contours.append(c)

    if len(table_contours) == 0:
        return None

    table_boundRect = [None] * len(table_contours)

    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 3, True)
        table_boundRect[i] = cv2.boundingRect(polygon)

    table_boundRect.sort()

    color = (0, 255, 0)
    thickness = 4

    for x, y, w, h in table_boundRect:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    return image, table_boundRect

def resize_padding(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    old_h, old_w, channels = img.shape

    new_w = 768
    new_h = 1024
    color = (255, 255, 255)

    result = np.full((new_h, new_w, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_w - old_w) // 2
    y_center = (new_h - old_h) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_h,
           x_center:x_center + old_w] = img

    return result
