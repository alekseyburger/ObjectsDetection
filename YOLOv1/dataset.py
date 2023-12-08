"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image

from model_output import CELLS_PER_DIM, CLASSES_NUM, BOXES_NUM, BOXES_AREA_LEN, BOX_PROPERTIES_LEN
from model_output import moutput_box, moutput_box_center, moutput_box_h_w
from model_output import box_center, box_h_w
from model_output import BoxIterator

import pdb

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self,
                csv_file,
                img_dir,
                label_dir,
                is_training_mode = False,
                transform=None,
                S=CELLS_PER_DIM,
                B=BOXES_NUM,
                C=CLASSES_NUM,):

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.training_mode = is_training_mode
 
        # self.debug_count = 0

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])

        image, boxes = Image.open(img_path), torch.tensor(boxes)

        if image and self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells: cell * cell * ( classes , boxes_confidence, boxes(in-cell-x, in-cell-y, in-image-width, in-image-hieght) )
        label_matrix = torch.zeros((self.S, self.S, self.C + self.B + (BOX_PROPERTIES_LEN * self.B)))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            y_in_img, x_in_img = self.S * y, self.S * x
            i_y, i_x = int(y_in_img), int(x_in_img)
            # y_in_cell, x_in_cell - box center in-cell offsets
            y_in_cell, x_in_cell  = y_in_img - i_y, x_in_img - i_x

            from model_output import soutput_box_probability, soutput_box, soutput_set_box
            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            for bidx in range(self.B):
                if soutput_box_probability(label_matrix[i_y, i_x], bidx) == 0.:
                    # 'Free' box - fill it
                    soutput_box_probability(label_matrix[i_y, i_x], bidx)[0] = 1.
                    soutput_set_box(label_matrix[i_y, i_x],
                                    bidx,
                                    torch.tensor([ x_in_cell, y_in_cell, width, height ]))
                    break
                # only one target per cell in training mode
                if self.training_mode: break

            label_matrix[i_y, i_x, class_label] = 1
            
        return image, label_matrix
