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
from craft_text_detector import Craft

import subprocess
import argparse
from utils.utils import getCoords, solve, getArea, getWidth, getHeight, getNearness
from termcolor import colored
import warnings
from sys import exit

warnings.simplefilter("ignore", UserWarning)

"""
    Inference Model
"""


def upscale_image(upscale_model, device):
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


def detect_text(image, device, struct_boxes):
    print(colored("Initializing Craft Text Detector.", "red"))

    if device == "cpu":
        craft = Craft(output_dir=None, crop_type="box", cuda=False, long_size=1200,)
    elif device == "cuda":
        craft = Craft(output_dir=None, crop_type="box", cuda=True, long_size=1200,)

    prediction_result = craft.detect_text(image)
    print(colored("Inference completed successfully.", "green"))
    regions = prediction_result["boxes"]

    craft_boxes = list()
    for i in range(len(regions)):
        craft_boxes.append(getCoords(regions[i]))

    return craft_boxes


def remove_overlap(struct_boxes, craft_boxes):
    structtemp = struct_boxes.copy()
    crafttemp = craft_boxes.copy()
    # To remove overlap between craft and structure detector, uncomment
    for i in range(len(craft_boxes)):
        for k in range(len(struct_boxes)):
            if solve(struct_boxes[k], craft_boxes[i]) == True:
                if getNearness(getHeight(struct_boxes[k]), getHeight(craft_boxes[i])):
                    if getArea(struct_boxes[k]) <= getArea(craft_boxes[i]):
                        if craft_boxes[i] in crafttemp:
                            crafttemp.remove(craft_boxes[i])
                    else:
                        if struct_boxes[k] in structtemp:
                            structtemp.remove(struct_boxes[k])
                else:
                    if craft_boxes[i] in crafttemp:
                        crafttemp.remove(craft_boxes[i])
            else:
                pass

    return structtemp, crafttemp


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
        help="Change upscaling model, Options: '4x_animesharp', 'nkmd_typescale'. Read: http://upscale.wiki/",
        default="4x_animesharp",
        required=False,
    )
    parser.add_argument(
        "--struct_weights", help="Load structure recognition model.", required=True,
    )
    args = parser.parse_args()

    if osp.exists(args.input_img):
        image = cv2.imread(str(args.input_img), cv2.IMREAD_COLOR)
        path = os.path.split(str(args.input_img))[0]
        base_name = os.path.split(str(args.input_img))[1]
    else:
        print(colored("Input image does not exist. Recheck file directory.", "red",))
        exit()

    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs("input/", exist_ok=True)
    os.makedirs("upscale/", exist_ok=True)

    cv2.imwrite(
        f"input/{base_name}", image[:, :, :3]
    )  # Removing possible alpha channel

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
        stat = upscale_image(args.upscale_model, device)

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
    config_name = osp.basename(checkpoint_file)[:-13] + ".py"

    # MMDet Pipeline
    struct_boxes = detect_structure(
        struct_img, osp.join("configs/", config_name), checkpoint_file, device, 0.6
    )

    # Keras Pipeline
    craft_boxes = detect_text(struct_img, device, struct_boxes)

    struct_final, craft_final = remove_overlap(struct_boxes, craft_boxes)
    # Save Output Image with BBoxes
    os.makedirs(f"output/{base_name[:-4]}", exist_ok=True)

    for i in range(len(struct_final)):
        struct_img = cv2.rectangle(
            struct_img,
            (struct_final[i][0], struct_final[i][1]),
            (struct_final[i][2], struct_final[i][3]),
            (0, 255, 0),
            2,
        )

    for i in range(len(craft_final)):
        struct_img = cv2.rectangle(
            struct_img,
            (craft_final[i][0], craft_final[i][1]),
            (craft_final[i][2], craft_final[i][3]),
            (255, 0, 255),
            2,
        )

    merged_boxes = list()
    # merged_boxes = struct_boxes.copy()
    for i in range(len(craft_boxes)):
        merged_boxes.append(craft_boxes[i])

    file = open(f"output/{base_name[:-4]}/bbox_coords.txt", "w")
    for k in range(len(merged_boxes)):
        file.write(
            str(merged_boxes[k][0])
            + ","
            + str(merged_boxes[k][1])
            + ","
            + str(merged_boxes[k][2])
            + ","
            + str(merged_boxes[k][3])
            + "\n"
        )
    file.close()

    print(
        colored(
            f"Saving output image {base_name} with bounding boxes.",
            "blue",
            None,
            ["bold"],
        )
    )
    cv2.imwrite(f"output/{base_name[:-4]}/bbox_detections.png", struct_img)
