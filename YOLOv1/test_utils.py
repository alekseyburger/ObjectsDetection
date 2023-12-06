import unittest

import torch
# import torchvision.transforms as transforms
# import torch.optim as optim
# import torchvision.transforms.functional as FT
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from model import Yolov1
# from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    convert_cellboxes
)
from math import sqrt
from loss import YoloLoss
from model_output import NOOBJ_LOSS_WEIGHT, COORD_LOSS_WEIGHT
from model_output import CELLS_PER_DIM, CLASSES_NUM, BOXES_NUM, BOX_PROPERTIES_LEN, BOXES_AREA_LEN
from model_output import COORD_LOSS_WEIGHT
from model_output import moutput_box_center, moutput_box_h_w, moutput_box
from model_output import box_center, box_h_w

import pdb

"""
python -m unittest test_loss.py
"""

PRED0=CLASSES_NUM
BOX0=PRED0+BOXES_NUM


class TestUtils(unittest.TestCase):

    BATCH_SIZE = 1

    @classmethod
    def setUpClass(cls):
        cls.DEVICE = "cpu"
        if torch.cuda.is_available : cls.DEVICE = "cuda"
        print(f"Device is {cls.DEVICE}")

    def setUp(self) -> None:
        
        self.clean()
        return super().setUp()
    
    def clean(self):
        self.prediction = torch.zeros([self.BATCH_SIZE,
                          CELLS_PER_DIM,
                          CELLS_PER_DIM, 
                          (CLASSES_NUM + BOXES_NUM + BOXES_AREA_LEN)],
                          dtype=torch.float).to(self.DEVICE)
        
        self.target = torch.zeros([self.BATCH_SIZE,
                          CELLS_PER_DIM,
                          CELLS_PER_DIM, 
                          (CLASSES_NUM + BOXES_NUM + BOXES_AREA_LEN)],
                          dtype=torch.float).to(self.DEVICE)

        self.ouput_as_list_zero = [0 for i in range(CLASSES_NUM + BOXES_NUM + BOXES_AREA_LEN)]

    def test_convert_cellboxes(self):

        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[CLASSES_NUM-1] = 1. # class       
        ouput_as_list[PRED0:PRED0+2] = [1.,0.5]
        ouput_as_list[BOX0 : BOX0 + 2*BOX_PROPERTIES_LEN] = [0.11 , 0.12, 0.41, 0.42, 0.53, 0.54, 0.43, 0.44]        

        self.prediction[0,CELLS_PER_DIM-1,CELLS_PER_DIM-1] = torch.tensor(ouput_as_list)

        ret = convert_cellboxes(self.prediction, CELLS_PER_DIM)

    def test_ellboxes_to_boxes(self):

        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[CLASSES_NUM-1] = 1. # class       
        ouput_as_list[PRED0:PRED0+2] = [1.,0.5]
        ouput_as_list[BOX0 : BOX0 + 2*BOX_PROPERTIES_LEN] = [0.11 , 0.12, 0.41, 0.42, 0.53, 0.54, 0.43, 0.44]        

        self.prediction[0,CELLS_PER_DIM-1,CELLS_PER_DIM-1] = torch.tensor(ouput_as_list)

        ret = cellboxes_to_boxes(self.prediction, CELLS_PER_DIM)

if __name__ == '__main__':
    unittest.main()