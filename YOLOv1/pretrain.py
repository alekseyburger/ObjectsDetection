#!/usr/bin/env python
"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import ClassificationDataset
from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    save_checkpoint,
)
from loss import YoloLoss
from model_output import CLASSES_NUM

import os, sys
from datetime import datetime
import argparse
import pathlib
import logging, logging.config

import pdb


parser = argparse.ArgumentParser(prog='pretrain',
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
parser.add_argument('-m', '--model', type=pathlib.Path)
parser.add_argument('--lrate', default=2e-5, type = float, help='learning rate (2e-5)')
parser.add_argument('--no-log',
                    action='store_const',
                    const=True,
                    default=False,
                    help='disable logging to file')
parser.add_argument('--show',
                    action='store_const',
                    const=True,
                    default=False,
                    help='Show images with classes')


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

logging.config.fileConfig("loggin.conf")
logger = logging.getLogger("pretrainLog" if not args.no_log else "debugLog")
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

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

print(f"Tourch device is {DEVICE}")

def model_file_name(pref : str = ""):
    dir_name = "model"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    current_time = datetime.now()
    sfx = f'-{current_time.date()}'+current_time.strftime(".%H.%M.%S")
    return dir_name + "/model-" + pref + sfx + ".pth.tar"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

def pretarin_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y) * x.shape[1]
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    logger.info(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            predictions = model(x)
            num_correct += ((predictions * y) > .9).sum()
            #num_correct +=  (((y < 1.) * predictions) < 0.1).sum()
            num_samples += y.sum()
            #num_samples += y.shape[0] * y.shape[1]

    model.train()
    return num_correct / (num_samples + 1.e-11)


def main():

    logger.info(f"Start train: {train_data_path} test: {test_data_path} learning rate {LEARNING_RATE}")

    model = Yolov1(split_size=7,
                   num_boxes=2,
                   num_classes=20,
                   model_type="pretraining").to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    loss_fn = nn.MSELoss()
    if model_path:
        load_checkpoint(torch.load(model_path, map_location=torch.device(DEVICE)), model, optimizer)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        logger.info(f"Load model {model_path}")
    logger.info(f'Classifier {model.fcs}')

    train_dataset = ClassificationDataset(
        train_data_path,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = ClassificationDataset(
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

    best_accuracy = 0.9
    reason = 'final'

    try:
        for epoch in range(EPOCHS):

            current_accuracy = accuracy(test_loader, model, device=DEVICE)

            logger.info(f"EPOCH {epoch}/{EPOCHS}")
            logger.info(f"Accuracy: {current_accuracy}")

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                cp_filename = model_file_name(f'CNN-best-mAP{current_accuracy:.2f}')
                save_checkpoint(checkpoint,filename = cp_filename)
                logger.info(f"Save model {cp_filename}")
            
            pretarin_fn(train_loader, model, optimizer, loss_fn)

    except KeyboardInterrupt:
        logger.info('Interrupted')
        reason = "Interrupted"
        pass

    cp_filename = model_file_name(f'CNN-{reason}-mAP{current_accuracy:.2f}')
    save_checkpoint(checkpoint, filename = cp_filename)
    logger.info(f"Save model {cp_filename}")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
def test():

    model = Yolov1(split_size=7,
                   num_boxes=2,
                   num_classes=20,
                   model_type="pretraining").to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if model_path:
        load_checkpoint(torch.load(model_path, map_location=torch.device(DEVICE)), model, optimizer)
        logger.info(f"Load model {model_path}")

    test_dataset = ClassificationDataset(
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

    for images, exp_label in test_loader:
        images = images.to(DEVICE)
        class_prediction = (model(images))

        for idx in range(images.shape[0]):
            name_list = [cls[i] for i in range(CLASSES_NUM)  if class_prediction[idx][i] > .5]
            exp_list = [cls[i] for i in range(CLASSES_NUM)  if exp_label[idx][i] > .9]
            im = np.array(images[idx].permute(1,2,0).to("cpu"))
            # Create figure and axes
            fig, ax = plt.subplots(1)
            # Display the image
            ax.imshow(im)
            plt.title(','.join(name_list)+':'+','.join(exp_list))
            plt.show()


if __name__ == "__main__":
    if not args.show: main()
    else: test()
