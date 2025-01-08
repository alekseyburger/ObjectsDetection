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
from Yolov1 import Model
from dataset import ClassificationDataset
from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    save_checkpoint,
    model_file_name,
    true_positive,
    true_negative,
    false_positive,
    true_positive_per_class,
    true_negative_per_class,
    false_positive_per_class,
    accuracy
)
from loss import YoloLoss
from model_output import CLASSES_NUM
from model_output import IMAGE_HEIGHT, IMAGE_WIDTH

import os, sys

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
parser.add_argument('-d', '--data-dir', type=pathlib.Path, required = False)
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
                    help='run program for Show images with classes')
parser.add_argument('--accuracy',
                    action='store_const',
                    const=True,
                    default=False,
                    help='run program for calculate true positive/negative accuracy per class')

args = parser.parse_args()

train_data_path = args.input_data
if not os.path.exists(train_data_path):
    print(f"Train data in not found at {train_data_path}")
    sys.exit(1)

test_data_path = args.test_data
if not os.path.exists(test_data_path):
    print(f"Test data in not found at {test_data_path}")
    sys.exit(1)

model_path = args.model
if model_path and not os.path.exists(model_path):
    print(f"Model in not found at {model_path}")
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

data_dir_path = args.data_dir
if data_dir_path:
    if not os.path.exists(data_dir_path):
        print(f"Data path is not exist {data_dir_path}")
        sys.exit(1)
else:
    data_dir_path = "data"

IMG_DIR = os.path.join(data_dir_path, "images")
LABEL_DIR = os.path.join(data_dir_path, "labels")

logger.info(f"pretrain: torch device is {DEVICE}")

cls = ['aeroplane','bicycle','bird','boat',
        'bottle','bus','car','cat',
        'chair','cow','diningtable','dog',
        'horse','motorbike','person','pottedplant',
        'sheep','sofa','train','tvmonitor']

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), transforms.ToTensor(),])

def pretarin_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y) # * y.shape[1]
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    logger.info(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def train():

    logger.info(f"Start with image {IMAGE_HEIGHT}x{IMAGE_WIDTH} train: {train_data_path} test: {test_data_path} batch: {BATCH_SIZE} learning rate:{LEARNING_RATE}")

    model = Model(num_cells=7,
                   num_boxes=2,
                   num_classes=len(cls),
                   model_type="pretraining").to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    loss_fn = nn.CrossEntropyLoss()
    if model_path:
        load_checkpoint(torch.load(model_path, map_location=torch.device(DEVICE)), model, optimizer)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model.train()
        logger.info(f"Load model {model_path}")
    logger.info(f'CNN {model.cnn}')
    logger.info(f'Reduction {model.reduction}')
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

    best_accuracy = 0.90
    reason = 'final'

    try:
        for epoch in range(EPOCHS):

            _accuracy = accuracy(test_loader, model, device=DEVICE)
            epoch_accuracy = (_accuracy[0] + _accuracy[1]) / 2.

            logger.info(f"EPOCH {epoch}/{EPOCHS}")
            logger.info(f"Accuracy: positive {_accuracy[0]:5.3f} negative {_accuracy[1]:5.3f} med: {epoch_accuracy:5.3f}")

            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                cp_filename = model_file_name(f'CNN-best-{IMAGE_HEIGHT}x{IMAGE_WIDTH}-mAP{epoch_accuracy:.2f}')
                save_checkpoint(checkpoint,filename = cp_filename)
                logger.info(f"Save model {cp_filename}")
            
            pretarin_fn(train_loader, model, optimizer, loss_fn)

    except KeyboardInterrupt:
        logger.info('Interrupted')
        reason = "Interrupted"
        pass

    cp_filename = model_file_name(f'CNN-{reason}-{IMAGE_HEIGHT}x{IMAGE_WIDTH}-mAP{epoch_accuracy:.2f}')
    save_checkpoint(checkpoint, filename = cp_filename)
    logger.info(f"Save model {cp_filename}")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def run_show():

    model = Model(num_cells=7,
                   num_boxes=2,
                   num_classes=len(cls),
                   model_type="pretraining").to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
 
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


    for images, exp_label in test_loader:
        images = images.to(DEVICE)
        class_predictions = (model(images))

        for idx in range(images.shape[0]):
            name_list = [cls[i] for i in range(CLASSES_NUM)  if class_predictions[idx][i] > .9]
            exp_list = [cls[i] for i in range(CLASSES_NUM)  if exp_label[idx][i] > .9]
            im = np.array(images[idx].permute(1,2,0).to("cpu"))
            # Create figure and axes
            fig, ax = plt.subplots(1)
            # Display the image
            ax.imshow(im)
            plt.title(','.join(name_list)+'('+','.join(exp_list)+')')
            plt.show()

def report_per_class_accuracy(class_correct, class_samples):
    for c in range(CLASSES_NUM):
        logger.info( f'{cls[c]} ({c}): \t{class_correct[c]} / {class_samples[c]} = {class_correct[c]*100/(class_samples[c] + 1.e-11):3.0f}%' )
    logger.info( f'Commulative {sum(class_correct)} / {sum(class_samples)} = {sum(class_correct)*100/(sum(class_samples) + 1.e-11):3.0f}%' )
   
def run_accuracy():

    logger.info(f"Start accuracy check with image {IMAGE_HEIGHT}x{IMAGE_WIDTH} train: {train_data_path} test: {test_data_path}")

    model = Model(num_cells=7,
                   num_boxes=2,
                   num_classes=len(cls),
                   model_type="pretraining").to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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

    train_dataset = ClassificationDataset(
        train_data_path,
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
    
    logger.info(f'Test Data:  --- True positive --- ')
    class_correct, class_samples = true_positive_per_class(test_loader, model, device=DEVICE)
    report_per_class_accuracy(class_correct, class_samples)

    logger.info(f'Train Data: --- True positive ---')
    class_correct, class_samples = true_positive_per_class(train_loader, model, device=DEVICE)
    report_per_class_accuracy(class_correct, class_samples)

    logger.info(f'Test Data: ---  False positive --- ')
    class_correct, class_samples = false_positive_per_class(test_loader, model, device=DEVICE)
    report_per_class_accuracy(class_correct, class_samples)

    logger.info(f'Train Data: ---  False positive --- ')
    class_correct, class_samples = false_positive_per_class(train_loader, model, device=DEVICE)
    report_per_class_accuracy(class_correct, class_samples)    

    logger.info(f'Test Data:  --- True negative --- ')
    class_correct, class_samples = true_negative_per_class(test_loader, model, device=DEVICE)
    report_per_class_accuracy(class_correct, class_samples)

    logger.info(f'Train Data: ---  True negative --- ')
    class_correct, class_samples = true_negative_per_class(train_loader, model, device=DEVICE)
    report_per_class_accuracy(class_correct, class_samples)


if __name__ == "__main__":
    if args.show:
        run_show()
    elif args.accuracy:
        run_accuracy()
    else:
        train()
