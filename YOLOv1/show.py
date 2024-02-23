#!/usr/bin/env python

"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from Yolov1 import Model
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    plot_image,
    load_checkpoint,
)
import pdb

import os, sys
import argparse
import pathlib
import logging, logging.config


logging.config.fileConfig("loggin.conf")
logger = logging.getLogger("testLog")

for handler in logger.handlers:
    if hasattr(handler, "baseFilename"):
        logfile_name = getattr(handler, 'baseFilename')
        logfile_name = os.path.abspath(logfile_name)
        logfile_dir = os.path.dirname(logfile_name)
        if not os.path.exists(logfile_dir):
            print(f"Create {logfile_dir}")
            os.makedirs(logfile_dir)

parser = argparse.ArgumentParser(prog='show',
            description='invovate model and show predictions')
parser.add_argument('-t', '--test-data', type=pathlib.Path, required = True)
parser.add_argument('-d', '--data-dir', type=pathlib.Path, required = False)
parser.add_argument('-m', '--model', type=pathlib.Path, required = True)
parser.add_argument('-b', '--batch', default=32, help='batch size')
parser.add_argument('--no-cuda',
                    action='store_const',
                    const=True,
                    default=False,
                    help='disable GPU')
parser.add_argument('-c', '--count', default=1, help='number of examples')
parser.add_argument('--iou', default=0.5, type = float, help='iou threshold (0.5)')
parser.add_argument('--nms', default=0.4, type = float, help='non max suppression threshold (0.4)')
parser.add_argument('--ground-true',
                    action='store_const',
                    const=True,
                    default=False,
                    help='show round true boxes')

args = parser.parse_args()

model_path = args.model
if not os.path.exists(model_path):
    print("Model in not found at {model_path}")
    sys.exit(1)

test_data_path = args.test_data
if not os.path.exists(test_data_path):
    print("Test data in not found at {test_data_path}")
    sys.exit(1)

iou_threshold = float(args.iou)
nms_hreshold = float(args.nms)
is_ground_true = args.ground_true

seed = 123
torch.manual_seed(seed)


DEVICE = "cuda" if torch.cuda.is_available and not args.no_cuda else "cpu"
BATCH_SIZE = int(args.batch)
# Hyperparameters etc. 
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
NUM_WORKERS = 2
PIN_MEMORY = True

data_dir_path = args.data_dir
if data_dir_path:
    if not os.path.exists(data_dir_path):
        print(f"Data path is not exist {data_dir_path}")
        sys.exit(1)
else:
    data_dir_path = "data"

IMG_DIR = os.path.join(data_dir_path, "images")
LABEL_DIR = os.path.join(data_dir_path, "labels")

print(f"Torch device is {DEVICE}")

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def main():

    model = Model(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    load_checkpoint(torch.load(model_path, map_location=torch.device(DEVICE)),
                    model,
                    optimizer)
    model.eval()

    test_dataset = VOCDataset(
        test_data_path,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory =  True if DEVICE == "cuda" else False,
        shuffle=True,
        drop_last=True,
    )

    cls = ['aeroplane','bicycle','bird','boat',
            'bottle','bus','car','cat',
            'chair','cow','diningtable','dog',
            'horse','motorbike','person','pottedplant',
            'sheep','sofa','train','tvmonitor']

    cont = int(args.count)
    for images, exp_label in test_loader:
        images = images.to(DEVICE)
        boxes = cellboxes_to_boxes(model(images))
        if is_ground_true:
            boxes = cellboxes_to_boxes(exp_label.reshape(exp_label.shape[0],-1))

        for idx in range(images.shape[0]):
            best_boxes = non_max_suppression(boxes[idx],
                                             iou_threshold = iou_threshold,
                                             threshold = nms_hreshold)
            for i in range(len(best_boxes)): print(best_boxes[i])

            plot_image(images[idx].permute(1,2,0).to("cpu"), best_boxes, cls)

            cont -= 1
            if cont <= 0:
                sys.exit()

if __name__ == "__main__":
    main()
