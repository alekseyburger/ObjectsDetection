import unittest

import torch
# from math import sqrt
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

def nearly_equal (expected, result):
    threshold = 10e-3
    if expected > 10e-3:
        threshold = expected / 100.
    return abs(expected - result) < threshold

class TestLoss(unittest.TestCase):

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
        self.loss_func = YoloLoss(CELLS_PER_DIM, BOXES_NUM, CLASSES_NUM)

    
    def test_classes (self):

        # false neative confidence
        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()
        ouput_as_list[PRED0:PRED0+2] = [1.,1.]
        self.prediction[0,0,0] = torch.tensor(ouput_as_list)

        ouput_as_list[PRED0:PRED0+2] = [1.,0.]
        ouput_as_list[0] = 1. # class 0
        self.target[0,0,0] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)

        expected_loss = 1.
        self.assertTrue(nearly_equal(expected_loss, loss),
                        msg = f"Center Loss box 0 = {loss} expect {expected_loss}")
    
    def test_confidention (self):

        # false neative confidence
        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[PRED0] = 1.

        self.target[0,0,0] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)

        expected_loss = 2.0
        self.assertTrue(nearly_equal(expected_loss, loss),
                        msg = f"Center Loss box 0 = {loss} expect {expected_loss}")

        # false positive confidence
        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[PRED0] = 1.
        ouput_as_list[PRED0+1] = 1.

        self.prediction[0,0,0] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)

        expected_loss = 2.0 * NOOBJ_LOSS_WEIGHT
        self.assertTrue(nearly_equal(expected_loss, loss),
                        msg = f"Center Loss box 0 = {loss} expect {expected_loss}")
        
        # false positive confidence
        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[PRED0] = 1.
        self.target[0,0,0] = torch.tensor(ouput_as_list)

        ouput_as_list[PRED0+1] = 1.
        self.prediction[0,0,0] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)

        expected_loss = 0.
        self.assertTrue(nearly_equal(expected_loss, loss),
                        msg = f"Center Loss box 0 = {loss} expect {expected_loss}")        
     

    def test_center (self):
        # zero - zero case
        self.clean()
        loss = self.loss_func(self.prediction, self.target)

        self.assertTrue(nearly_equal(0.,loss),
                        msg = f"Zero:Zero Loss = {loss}")

        # self.clean()
        # false negative box 0 and 1
        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[PRED0:PRED0+2] = [1.,1.]
        self.prediction[0,0,0] = torch.tensor(ouput_as_list)

        ouput_as_list[PRED0:PRED0+2] = [1.,0.]
        ouput_as_list[BOX0 : BOX0 + BOX_PROPERTIES_LEN] = [0.5, 0.3, 0., 0.]

        self.target[0,0,0] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)
        expected_loss = COORD_LOSS_WEIGHT * (0.5**2 + 0.3**2) * 2

        self.assertTrue(nearly_equal(expected_loss, loss),
                        msg = f"Center Loss box 0 = {loss} expect {expected_loss}")

        #  box #2 false positive
        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[PRED0 : PRED0+2] = [1.,1.]
        ouput_as_list[BOX0  : BOX0 + BOX_PROPERTIES_LEN] = [0.8, 0.1, 0.4, 0.4]
        ouput_as_list[BOX0 + BOX_PROPERTIES_LEN : BOX0 + 2*BOX_PROPERTIES_LEN] = [0.8, 0.1, 0.4, 0.4]

        self.prediction[0,CELLS_PER_DIM-1, CELLS_PER_DIM-1] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)
        expected_loss = 0 + NOOBJ_LOSS_WEIGHT * (1. + 1.)

        self.assertTrue( nearly_equal(expected_loss, loss),
                        msg = f"Center Loss box #2 = {loss} expect {expected_loss}")
        
        # target and pred box #1 are equal
        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[CLASSES_NUM : CLASSES_NUM+1] = [1.,]
        ouput_as_list[CLASSES_NUM + BOXES_NUM : CLASSES_NUM + BOXES_NUM + BOX_PROPERTIES_LEN] = [0.1 , 0.1, 0.4, 0.4]

        y_cell, x_cell = CELLS_PER_DIM//2, CELLS_PER_DIM//2
        self.target[0, y_cell, x_cell] = torch.tensor(ouput_as_list)

        ouput_as_list[PRED0 : PRED0+2] = [1.,1.]  # privent no_object_loss
        self.prediction[0, y_cell, x_cell] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)
        expected_loss = COORD_LOSS_WEIGHT * (0.1**2 + 0.1**2 + 0.4 + 0.4 )
        self.assertTrue( nearly_equal(expected_loss, loss),
                        msg = f"target and pred are equal center Loss {loss} {expected_loss}")

        # target and pred are equal in different boxes
        self.clean()

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[CLASSES_NUM : CLASSES_NUM+1] = [1.,]
        ouput_as_list[CLASSES_NUM + BOXES_NUM : CLASSES_NUM + BOXES_NUM + BOX_PROPERTIES_LEN] = [0.1 , 0.1, 0.4, 0.4]

        y_cell, x_cell = CELLS_PER_DIM//2, CELLS_PER_DIM//2
        self.target[0, y_cell, x_cell] = torch.tensor(ouput_as_list)

        ouput_as_list = self.ouput_as_list_zero.copy()

        ouput_as_list[PRED0 : PRED0+2] = [1.,1.]  # privent no_object_loss        
        ouput_as_list[CLASSES_NUM + BOXES_NUM + BOX_PROPERTIES_LEN: CLASSES_NUM + BOXES_NUM + 2*BOX_PROPERTIES_LEN] = [0.1 , 0.1, 0.4, 0.4]

        self.prediction[0, y_cell, x_cell] = torch.tensor(ouput_as_list)

        loss = self.loss_func(self.prediction, self.target)
        # box #1 brings the loss
        expected_loss = COORD_LOSS_WEIGHT * (0.1**2 + 0.1**2 + 0.4 + 0.4 )
        self.assertTrue( nearly_equal(expected_loss, loss),
                        msg = f"target and pred are equal center Loss {loss} {expected_loss}")

if __name__ == '__main__':
    unittest.main()