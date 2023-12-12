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
from model import Yolov1
from dataset import VOCDataset
from utils import (
    get_bboxes,
    mean_average_precision,
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

parser = argparse.ArgumentParser(prog='test',
            description='model tester')
parser.add_argument('-t', '--test-data', type=pathlib.Path, required = True)
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

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

print(f"Tourch device is {DEVICE}")

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def main():

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
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

    for iou_threshold in [i*0.1 for i in range(4,10)]:
        for conf in [i*0.1 for i in range(1,10)]:
            pred_boxes, target_boxes = get_bboxes(
                test_loader,
                model,
                iou_threshold=iou_threshold,
                threshold=conf,
                device=DEVICE)

            mean_avg_prec = mean_average_precision(
                pred_boxes,
                target_boxes,
                iou_threshold=0.5)
            
            print(f'mean. avg. precision: {mean_avg_prec*100.:3.2f}% for confidence {conf:.2f} iou {iou_threshold:.2f}')

if __name__ == "__main__":
    main()
