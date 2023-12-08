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
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

import os, sys
from datetime import datetime
import argparse
import pathlib

parser = argparse.ArgumentParser(prog='test',
            description='model tester')
parser.add_argument('-c', '--count', default=1, help='number of examples')
parser.add_argument('-b', '--batch', default=32, help='batch size')
parser.add_argument('--no-cuda',
                    action='store_const',
                    const=True,
                    default=False,
                    help='disable GPU')
parser.add_argument('-t', '--test_data', type=pathlib.Path, required = True)

parser.add_argument('-m', '--model', type=pathlib.Path, required = True)
args = parser.parse_args()

model_path = args.model
if not os.path.exists(model_path):
    print("Model in not found at {model_path}")
    sys.exit(1)

test_data_path = args.test_data
if not os.path.exists(test_data_path):
    print("Test data in not found at {test_data_path}")
    sys.exit(1)

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

    load_checkpoint(torch.load(model_path), model, optimizer)

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
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    cont = args.count
    for x, y in test_loader:
        x = x.to(DEVICE)
        for idx in range(x.shape[0]):
            bboxes = cellboxes_to_boxes(model(x))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4)
            for i in range(len(bboxes)):
                print(bboxes[i])
            plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

            cont -= 1
            if cont <= 0:
                sys.exit()

if __name__ == "__main__":
    main()
