# Common Libraries
import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision

# OpenMMLabs
import mmdet
from mmdet.apis import inference_detector
from mmdet.apis.inference import init_detector
import keras_ocr

import subprocess
import argparse
from config import base_dir
from utils.utils import getCoords, solve, getArea
from termcolor import colored
import warnings

warnings.simplefilter("ignore", UserWarning)

"""
    Inference Model
"""


def upscale_image(upscale_model, input_dir, device):
    image = cv2.imread(str(input_dir), cv2.IMREAD_COLOR)
    path = os.path.split(str(input_dir))[0]
    base_name = os.path.split(str(input_dir))[1]

    os.makedirs("input/", exist_ok=True)
    os.makedirs("upscale/", exist_ok=True)

    cv2.imwrite(
        f"input/{base_name}", image[:, :, :3]
    )  # Removing possible alpha channel

    if device == "cpu":
        proc = subprocess.run(f"python upscale.py -m {upscale_model} -cpu", shell=True,)
    else:
        proc = subprocess.run(f"python upscale.py -m {upscale_model}", shell=True,)

    return proc.returncode


def detect_structure(image, config_file, checkpoint_file, device, thresh):
    print(colored("Initializing MMDetection Pipeline.", "red"))
    model = init_detector(config_file, checkpoint_file, device=device)
    result = inference_detector(model, image)  # Prints array of bbox
    print(colored("Inference completed successfully.", "green"))

    det_boxes = []
    for r in result[0]:
        if r[4] > thresh:
            det_boxes.append(r.astype(int))

    det_boxes = np.array(det_boxes)[
        :, :4
    ]  # Each element is of format [xmin, ymin, xmax, ymax]
    det_boxes = det_boxes.tolist()
    struct_boxes = det_boxes.copy()

    for i in range(len(det_boxes)):
        for k in range(len(det_boxes)):
            if (k != i) and (solve(det_boxes[i], det_boxes[k]) == True):
                if (det_boxes[i] in struct_boxes) and (
                    getArea(det_boxes[i]) < getArea(det_boxes[k])
                ):
                    struct_boxes.remove(det_boxes[i])
            else:
                pass

    return struct_boxes


def detect_text(image):
    print(colored("Initializing KerasOCR Pipeline.", "red"))
    pipeline = keras_ocr.pipeline.Pipeline()
    prediction_groups = pipeline.recognize([image])
    print(colored("Inference completed successfully.", "green"))

    boxes = list()
    for i in range(len(prediction_groups[0])):
        _, box = prediction_groups[0][i]
        boxes.append(box)

    temp_boxes = list()
    for i in range(len(boxes)):
        temp_boxes.append(getCoords(boxes[i]))

    keras_boxes = temp_boxes.copy()

    for i in range(len(temp_boxes)):
        for k in range(len(struct_boxes)):
            if solve(struct_boxes[k], temp_boxes[i]) == True:
                if temp_boxes[i] in keras_boxes:
                    keras_boxes.remove(temp_boxes[i])
            else:
                pass

    return keras_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform Table Detection > Table Structure Extraction"
    )
    parser.add_argument("--input_img", help="Path to input image.", required=True)
    parser.add_argument(
        "--upscale",
        help="Upscale / Yields better results in most scenarios.",
        dest="upscale",
        action="store_true",
    )
    parser.add_argument(
        "--no-upscale", help="Disable upscaling.", dest="upscale", action="store_false"
    )
    parser.set_defaults(upscale=True)
    parser.add_argument(
        "--upscale_model",
        help="Change upscaling model, Options: 'animesharp', 'nkmd_typescale'. Read: http://upscale.wiki/",
        default="animesharp",
        required=False,
    )
    parser.add_argument(
        "--struct_weights", help="Load structure recognition model.", required=True,
    )
    args = parser.parse_args()

    if osp.exists(args.input_img):
        pass
    else:
        print(colored("Input image does not exist. Recheck file directory.", "red",))
        exit()

    base_name = os.path.split(str(args.input_img))[1]
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # MMDet Pipeline
    if device == "cpu":
        print(colored("Using CPU runtime, this may take a while.", "red"))
        print(
            colored(
                "Not what is expected? -> Ensure that CUDA-Ready version of PyTorch is installed.",
                "red",
            )
        )
    else:
        print(
            colored(
                "Using CUDA runtime to perform upscaling & structure recognition.",
                "red",
            )
        )
    if args.upscale:
        stat = upscale_image(args.upscale_model, args.input_img, device)

        if stat == 0:
            print(colored("Upscaled input successfully.", "green"))
            ori_img = cv2.imread(f"upscale/{base_name[:-4]}.png", cv2.IMREAD_COLOR)
        elif stat == 1:
            print(colored("Recheck argument directiories and try again.", "red"))
            exit()
        else:
            print(
                colored(
                    "Upscale did not complete successfully. Proceeding without upscaling.",
                    "red",
                )
            )
            ori_img = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
            ori_img = ori_img[:, :, :3]  # Removing possible alpha channel

    else:
        print(colored("Upscale set to False. Proceeding without upscaling.", "green"))
        ori_img = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
        ori_img = ori_img[:, :, :3]  # Removing possible alpha channel

    struct_img = ori_img.copy()

    checkpoint_file = args.struct_weights
    print(checkpoint_file)

    # MMDet Pipeline
    struct_boxes = detect_structure(
        struct_img, "config.py", checkpoint_file, device, 0.3
    )

    # Keras Pipeline
    keras_boxes = detect_text(struct_img)

    for i in range(len(struct_boxes)):
        struct_img = cv2.rectangle(
            struct_img,
            (struct_boxes[i][0], struct_boxes[i][1]),
            (struct_boxes[i][2], struct_boxes[i][3]),
            (0, 255, 0),
            2,
        )

    os.makedirs("output/", exist_ok=True)

    for i in range(len(keras_boxes)):
        struct_img = cv2.rectangle(
            struct_img,
            (keras_boxes[i][0], keras_boxes[i][1]),
            (keras_boxes[i][2], keras_boxes[i][3]),
            (0, 255, 0),
            2,
        )
    print(colored("\nSaving output image with bounding boxes.", "green",))
    cv2.imwrite(f"output/{base_name}", struct_img)
