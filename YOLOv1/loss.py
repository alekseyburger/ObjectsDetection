"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union

from model_output import CELLS_PER_DIM, CLASSES_NUM, BOXES_NUM, BOXES_AREA_LEN
from model_output import NOOBJ_LOSS_WEIGHT, COORD_LOSS_WEIGHT
from model_output import moutput_confidences, moutput_classes
from model_output import moutput_box_confidence, moutput_box, moutput_box_center, moutput_box_h_w
from model_output import box_center, box_h_w 
import model_output as moutput

import pdb

class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=CELLS_PER_DIM, B=BOXES_NUM, C=CLASSES_NUM):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = NOOBJ_LOSS_WEIGHT
        self.lambda_coord = COORD_LOSS_WEIGHT

    def forward (self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        target_box = moutput_box(target,0)
        target_box_center = moutput_box_center(target,0)
        target_box_h_w = moutput_box_h_w(target,0)
        target_box_h_w_sqr = torch.sqrt(target_box_h_w)
        target_box_confidence = moutput_box_confidence(target,0)
    
        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_per_box_idx_list = []
        for bidx in range(BOXES_NUM):
            # compare with target box #0 only
            iou = intersection_over_union(moutput_box(predictions,bidx),
                                          target_box,
                                          hw_image_proportional = True)
            iou_per_box_idx_list.append(iou)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        _, resposaible_box_idx = torch.max(torch.cat(iou_per_box_idx_list, dim=-1), dim=-1)

        # during the training there is only one target box per cell.
        # So we are considering box #0 only
        cell_is_representative =  target_box_confidence # in paper this is Iobj_i
        resposaible_box_idx = torch.where(cell_is_representative[...,0]!=0., resposaible_box_idx, -1)

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        loss_center, loss_h_w = 0., 0.
        loss_no_object = 0.
        loss_obj_confidention = 0.
        loss_class = 0.

        for b in range(BOXES_NUM):
            
            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            resposaible_box = moutput_box(predictions,b)
            resposaible_box = (resposaible_box_idx == b).unsqueeze(-1) * resposaible_box
            resposaible_box = cell_is_representative * resposaible_box
            box_b_center = (resposaible_box_idx == b).unsqueeze(-1) * target_box_center
            box_b_h_w_sqr = (resposaible_box_idx == b).unsqueeze(-1) * target_box_h_w_sqr
            
            resposaible_box_center = box_center(resposaible_box)
            loss_center += self.mse(torch.flatten(box_b_center, end_dim=-2),
                                    torch.flatten(resposaible_box_center, end_dim=-2))

            resposaible_box_h_w = box_h_w(resposaible_box)
            resposaible_box_h_w_sqr = torch.sign(resposaible_box_h_w) * torch.sqrt(torch.abs(resposaible_box_h_w) + 1e-6 )

            loss_h_w +=  self.mse(torch.flatten(box_b_h_w_sqr, end_dim=-2),
                        torch.flatten(resposaible_box_h_w_sqr, end_dim=-2)) #
            
            # ==================== #
            #   FOR OBJECT LOSS    #
            # ==================== #
            box_confidence = moutput_box_confidence(predictions,b)
            box_confidence_with_obj =  (resposaible_box_idx == b).unsqueeze(-1) * box_confidence
            box_confidence_no_obj =  (resposaible_box_idx != b).unsqueeze(-1) * box_confidence
            box_b_confidence =  (resposaible_box_idx == b).unsqueeze(-1) * target_box_confidence

            loss_obj_confidention += self.mse(torch.flatten(box_b_confidence, end_dim=-2),
                            torch.flatten(box_confidence_with_obj, end_dim=-2))
                      
            loss_no_object += torch.sum(torch.square(box_confidence_no_obj))

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        loss_class = self.mse(
            torch.flatten(cell_is_representative * moutput_classes(predictions), end_dim=-2,),
            torch.flatten(cell_is_representative * moutput_classes(target), end_dim=-2,),
        )
        
        loss_coordinates = loss_center + loss_h_w

        loss = (
            self.lambda_coord * loss_coordinates  # first two rows in paper
            + loss_obj_confidention  # third row in paper
            + self.lambda_noobj * loss_no_object # forth row
            + loss_class  # fifth row
        )

        return loss
