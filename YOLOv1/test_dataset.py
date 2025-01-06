import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
import torch
from torch.utils.data import DataLoader
from dataset import VOCDataset
import torchvision.transforms as transforms

from model_output import CELLS_PER_DIM, CLASSES_NUM, BOXES_NUM, BOX_PROPERTIES_LEN, BOXES_AREA_LEN
from model_output import COORD_LOSS_WEIGHT
from model_output import moutput_box_center, moutput_box_h_w, moutput_box
from model_output import box_center, box_h_w
from model_output import obj_print
from model_output import soutput_box_probability, soutput_box

"""
python -m unittest test_dataset.py
"""
# LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 1 # 64 in original paper but I don't have that much vram, grad accum?
# WEIGHT_DECAY = 0
# EPOCHS = 1000
NUM_WORKERS = 1
PIN_MEMORY = True
# LOAD_MODEL = False
# LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
FILE_CSV = "data/1examples.csv"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

def nearly_equal (expected, result, threshold = 10e-3):
    # if expected > (threshold * 100.):
    #     threshold = expected / 100.
    return abs(expected - result) < threshold

import pdb

class TestDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.DEVICE = DEVICE
        print(f"Device is {self.DEVICE}")

        transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
        
        self.train_dataset = VOCDataset(
            FILE_CSV,
            transform=transform,
            img_dir=IMG_DIR,
            label_dir=LABEL_DIR,
        )

        return super().setUp()
     
    def test_one (self) -> None:
        
        # Input
        # 12 0.339 0.6693333333333333 0.402 0.42133333333333334
        # 14 0.379 0.5666666666666667 0.158 0.3813333333333333
        # 14 0.612 0.7093333333333333 0.084 0.3466666666666667
        # 14 0.555 0.7026666666666667 0.078 0.34933333333333333
        # Result class y, x, height, width
        # 3:2 14 tensor([0.9667, 0.6530, 0.3813, 0.1580])
        # 4:2 12 tensor([0.6853, 0.3730, 0.4213, 0.4020])
        # 4:3 14 tensor([0.9187, 0.8850, 0.3493, 0.0780])
        # 4:4 14 tensor([0.9653, 0.2840, 0.3467, 0.0840])

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )

        x_cell = int(0.3390*CELLS_PER_DIM)
        x = 0.3390*CELLS_PER_DIM - x_cell
        y_cell = int(0.6693*CELLS_PER_DIM)
        y = 0.6693*CELLS_PER_DIM - y_cell

        iterator = train_loader._get_iterator()
        try:

            _, boxes_list = iterator.__next__()
            # for target in boxes_list:
            soutput = boxes_list[0,y_cell,x_cell]

            self.assertEqual(soutput[12].item(),1)
            self.assertEqual(soutput_box_probability(soutput,0).item(), 1)
            box_pattern = torch.tensor([ x, y, 0.402, 0.42133333333333334])
            self.assertTrue(nearly_equal(0., torch.sum(torch.abs(soutput_box(soutput,0) - box_pattern))))

        except StopIteration:
            self.assertFalse(True, "Unexpected end of saples")

if __name__ == '__main__':
    unittest.main()