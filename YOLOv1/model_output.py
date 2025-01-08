import torch

# IMAGE_HEIGHT =  112
# IMAGE_WIDTH =  112
IMAGE_HEIGHT =  448
IMAGE_WIDTH =  448

CELLS_PER_DIM = 7
CLASSES_NUM = 20
BOXES_NUM = 2
BOX_PROPERTIES_LEN = 4
BOXES_AREA_LEN = BOX_PROPERTIES_LEN*BOXES_NUM
OUTPUT_LEN = CLASSES_NUM + BOXES_NUM + BOXES_AREA_LEN

COORD_LOSS_WEIGHT = 5
NOOBJ_LOSS_WEIGHT = 0.5

# Single output attrubutes
def soutput_box_probability (soutput, idx):
        offset = CLASSES_NUM + idx
        return soutput[...,offset:offset+1]

def soutput_box (soutput, box_idx):
    offset = CLASSES_NUM + BOXES_NUM + box_idx*BOX_PROPERTIES_LEN
    return soutput[offset:offset+BOX_PROPERTIES_LEN]

def soutput_set_box (soutput, box_idx, tvalue):
    offset = CLASSES_NUM + BOXES_NUM + box_idx*BOX_PROPERTIES_LEN
    soutput[offset:offset+BOX_PROPERTIES_LEN] = tvalue

# Attrubutes from output Matrix batch * height * width
def moutput_classes(moutput):
    offset = CLASSES_NUM
    return moutput[...,:CLASSES_NUM]

def moutput_confidences(moutput):
    offset = CLASSES_NUM
    return moutput[...,offset:offset+BOXES_NUM]

def moutput_box_confidence(moutput, box_idx):
    offset = CLASSES_NUM + box_idx
    return moutput[...,offset:offset+1]

def moutput_box (moutput, box_idx):
    offset = CLASSES_NUM + BOXES_NUM + box_idx*BOX_PROPERTIES_LEN
    return moutput[...,offset:offset+BOX_PROPERTIES_LEN]

def moutput_box_center (moutput, box_idx):
    offset = CLASSES_NUM + BOXES_NUM + box_idx*BOX_PROPERTIES_LEN
    return moutput[...,offset:offset+2]

def moutput_box_h_w (moutput, box_idx):
    offset = CLASSES_NUM + BOXES_NUM  + box_idx*BOX_PROPERTIES_LEN  + 2
    return moutput[...,offset:offset+2]

# box attributes
def box_center (box):
    return box[...,0:2]

def box_h_w (box):
    return box[...,2:4]

class BoxIterator:
    def __init__(self, out_tensor):
        self._t = out_tensor
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < BOXES_NUM:
            probability_offset = CLASSES_NUM + self._index
            probability = self._t[probability_offset:probability_offset+1]
            box_offset = CLASSES_NUM + BOXES_NUM # classes and pobability
            box_offset += self._index*BOX_PROPERTIES_LEN
            box = self._t[box_offset:box_offset+BOX_PROPERTIES_LEN]
            self._index += 1
            return probability, box
        else:
            raise StopIteration


def obj_print (out_tensor):
    haight = out_tensor.shape[0]
    width = out_tensor.shape[1]

    box_num = (out_tensor.shape[2] - CLASSES_NUM - BOXES_NUM)/BOX_PROPERTIES_LEN
    if int(box_num) != box_num: raise Exception()
    else: box_num = int(box_num)
    for h in range(haight):
        for w in range(width):
            for probability, box in BoxIterator(out_tensor[h][w]):
                if probability: 
                    print(f"{h}:{w} {probability} {box}")
