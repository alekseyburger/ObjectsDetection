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
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    save_checkpoint,
    model_file_name
)
from loss import YoloLoss

import os, sys
import argparse
import pathlib
import logging, logging.config


parser = argparse.ArgumentParser(prog='train',
            description='model trainer')
parser.add_argument('-e', '--epoch', default=100, help='epohs number')
parser.add_argument('-b', '--batch', default=32, help='batch size')
parser.add_argument('--no-cuda',
                    action='store_const',
                    const=True,
                    default=False,
                    help='disable GPU')
parser.add_argument('-i', '--input-data', type=pathlib.Path, required = True)
parser.add_argument('-t', '--test-data', type=pathlib.Path, required = True)
parser.add_argument('-d', '--data-dir', type=pathlib.Path, required = False)
parser.add_argument('-m', '--model', type=pathlib.Path)
parser.add_argument('--feature-extraction', type=pathlib.Path,
                    help='feature extraction: freeze CNN , replace classification layer')
parser.add_argument('--lrate', default=2e-5, type = float, help='learning rate (2e-5)')
parser.add_argument('--no-log',
                    action='store_const',
                    const=True,
                    default=False,
                    help='disable logging to file')

args = parser.parse_args()

train_data_path = args.input_data
if not os.path.exists(train_data_path):
    print("Train data in not found at {train_data_path}")
    sys.exit(1)

test_data_path = args.test_data
if not os.path.exists(test_data_path):
    print("Test data in not found at {test_data_path}")
    sys.exit(1)

model_path = args.model
if model_path and not os.path.exists(model_path):
    print("Model in not found at {model_path}")
    sys.exit(1)

feature_extraction_model = args.feature_extraction
if feature_extraction_model and not os.path.exists(feature_extraction_model):
    print("Model in not found at {feature_extraction_model}")
    sys.exit(1)

logging.config.fileConfig("loggin.conf")
logger = logging.getLogger("trainLog" if not args.no_log else "debugLog")

for handler in logger.handlers:
    if hasattr(handler, "baseFilename"):
        logfile_name = getattr(handler, 'baseFilename')
        logfile_name = os.path.abspath(logfile_name)
        logfile_dir = os.path.dirname(logfile_name)
        if not os.path.exists(logfile_dir):
            print(f"Create {logfile_dir}")
            os.makedirs(logfile_dir)

seed = 123
torch.manual_seed(seed)

DEVICE = "cuda" if torch.cuda.is_available and not args.no_cuda else "cpu"
BATCH_SIZE = int(args.batch)
# Hyperparameters etc.
LEARNING_RATE = float(args.lrate)
WEIGHT_DECAY = 0
EPOCHS = int(args.epoch)
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

data_dir_path = args.data_dir
if data_dir_path:
    if not os.path.exists(data_dir_path):
        print(f"Data path is not exist {data_dir_path}")
        sys.exit(1)
else:
    data_dir_path = "data"

IMG_DIR = os.path.join(data_dir_path, "images")
LABEL_DIR = os.path.join(data_dir_path, "labels")

logger.info(f"train torch device is {DEVICE}")

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    logger.info(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

# def nested_children(m: torch.nn.Module):
#     children = dict(m.named_children())
#     output = {}
#     if children == {}:
#         # if module has no children; m is last child! :O
#         return m
#     else:
#         # look for children from children... to the last child!
#         for name, child in children.items():
#             try:
#                 output[name] = nested_children(child)
#             except TypeError:
#                 output[name] = nested_children(child)
#     return output

def main():

    logger.info(f"Start train: {train_data_path} test: {test_data_path} batch: {BATCH_SIZE} learning rate {LEARNING_RATE}")

    if feature_extraction_model :
        model = Model(split_size=7,
                    num_boxes=2,
                    num_classes=20,
                    model_type="pretraining").to("cpu")
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        load_checkpoint(torch.load(feature_extraction_model,map_location=torch.device("cpu")),
                            model,
                            optimizer)
        logger.info(f"Load model {feature_extraction_model}")
        logger.info(f'Original classifier {model.fcs}')
        model.classifier_to_detection(split_size=7,
                    num_boxes=2,
                    num_classes=20)
        model.fcs.train()
        logger.info(f'Target classifier {model.fcs}')

        model.cnn_freeze(True)
        # model.cnn.eval()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model.to(DEVICE)
    else:
        model = Model(split_size=7,
                    num_boxes=2,
                    num_classes=20).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        if model_path:
            load_checkpoint(torch.load(model_path,map_location=torch.device(DEVICE)),
                            model,
                            optimizer)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            model.train()
            logger.info(f"Load model {model_path}")

    loss_fn = YoloLoss()

    train_dataset = VOCDataset(
        train_data_path,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        is_training_mode = True
    )

    test_dataset = VOCDataset(
        test_data_path,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory =  True if DEVICE == "cuda" else False,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory =  True if DEVICE == "cuda" else False,
        shuffle=True,
        drop_last=True,
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    best_mean_avg = 0.8
    reason = 'final'

    try:
        for epoch in range(EPOCHS):

            with torch.no_grad():
                pred_boxes, target_boxes = get_bboxes(
                    test_loader,
                    model,
                    iou_threshold=0.5,
                    threshold=0.4,
                    device=DEVICE)

                mean_avg_prec = mean_average_precision(
                    pred_boxes,
                    target_boxes,
                    iou_threshold=0.5)

                logger.info(f"EPOCH {epoch}/{EPOCHS}")
                logger.info(f"Train mAP: {mean_avg_prec}")

                if mean_avg_prec > best_mean_avg:
                    best_mean_avg = mean_avg_prec
                    cp_filename = model_file_name(f'best-mAP{mean_avg_prec:.2f}')
                    save_checkpoint(checkpoint,filename = cp_filename)
                    logger.info(f"Save model {cp_filename}")

            train_fn(train_loader, model, optimizer, loss_fn)

    except KeyboardInterrupt:
        logger.info('Interrupted')
        reason = "Interrupted"
        pass

    cp_filename = model_file_name(f'{reason}-mAP{mean_avg_prec:.2f}')
    save_checkpoint(checkpoint, filename = cp_filename)
    logger.info(f"Save model {cp_filename}")

if __name__ == "__main__":
    main()
