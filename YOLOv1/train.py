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
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
)
from loss import YoloLoss

import os, sys
from datetime import datetime
import argparse
import pathlib
import logging, logging.config


logging.config.fileConfig("loggin.conf")
logger = logging.getLogger("trainLog")

parser = argparse.ArgumentParser(prog='train',
            description='model trainer')
parser.add_argument('-e', '--epoch', default=100, help='epohs number')
parser.add_argument('-b', '--batch', default=32, help='batch size')
parser.add_argument('--no_cuda', default=False, help='disable GPU')
parser.add_argument('-i', '--input_data', type=pathlib.Path, required = True)
parser.add_argument('-t', '--test_data', type=pathlib.Path, required = True)
parser.add_argument('-m', '--model', type=pathlib.Path)

args = parser.parse_args()

train_data_path = args.input_data
if not os.path.exists(train_data_path):
    print("Train data in not found at {train_data_path}")
    sys.exit(1)

test_data_path = args.input_data
if not os.path.exists(test_data_path):
    print("Test data in not found at {test_data_path}")
    sys.exit(1)

model_path = args.model
if model_path and not os.path.exists(model_path):
    print("Model in not found at {model_path}")
    sys.exit(1)

seed = 123
torch.manual_seed(seed)

DEVICE = "cuda" if torch.cuda.is_available and not args.no_cuda else "cpu"
BATCH_SIZE = int(args.batch)
# Hyperparameters etc.
LEARNING_RATE = 2e-5
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


def main():

    logger.info(f"Start train: {test_data_path} test: {test_data_path}")

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if model_path:
        load_checkpoint(torch.load(model_path), model, optimizer)
        logger.info(f"Load model {model_path}")

    train_dataset = VOCDataset(
        test_data_path,
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
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
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

            pred_boxes, target_boxes = get_bboxes(
                test_loader,
                model,
                iou_threshold=0.5,
                threshold=0.4)

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
