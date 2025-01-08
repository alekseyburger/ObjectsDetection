"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

cnn_architecture_config = [
    (7, 64, 2, 3),    # CNN kernel_size, out_channels, stride, padding
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
]
reduction_architecture_config = [
     (3, 1024, 1, 1),
     (3, 1024, 2, 1),
     (3, 1024, 1, 1),
     (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Model(nn.Module):
    """
    Model input (batch, in_channels=3, 448,448) ABURGER: fix the size
    """

    def __init__(self, in_channels=3, model_type = "training", **kwargs):
        super(Model, self).__init__()

        self.in_channels = in_channels
        # it returns output chennals number in self.in_channels
        self.cnn = self._create_conv_layers(cnn_architecture_config, self.in_channels)
        self.reduction = None
        self.is_pretraining = model_type == "pretraining"
        #print(f"CNN input channels {self.in_channels} output channels {self.conv_channels}")

        if self.is_pretraining:
            self.fcs = self._create_pretraining_fcs(self.conv_channels, **kwargs)
        else:
            self.reduction = self._create_conv_layers(reduction_architecture_config, self.conv_channels)
            self.fcs = self._create_main_fcs(self.conv_channels, **kwargs)

    def forward(self, x):
        x = self.cnn(x)             #torch.Size([batch, 1024, 14, 14])
        if self.reduction:
            x = self.reduction(x)   #torch.Size([batch, 1024, 7, 7])
        return self.fcs(x)

    def _create_conv_layers(self, architecture, in_channels):
        layers = []

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        self.conv_channels = in_channels
        return nn.Sequential(*layers)

    def _create_main_fcs(self, in_channals, num_cells, num_boxes, num_classes):

        # In original paper this should be
        # nn.Linear(1024*num_cells*num_cells, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, num_cells*num_cells*(num_boxes*5+num_classes))

        lsize = num_classes * 4096//128   # Fit model to available GPU resources
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channals * num_cells * num_cells, lsize),
            # nn.Linear(in_channals * 2 * 2, lsize), # 112x112
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(lsize, num_cells * num_cells * (num_classes + num_boxes * 5)),
        )

    def _create_pretraining_fcs(self, in_channals, num_cells, num_boxes, num_classes):
        '''
        Creating classification layers for the pre-trained model runing as a classifier
        The output is an array of probabilities for each class.
        '''

        lsize = num_classes * 4096//32   # Fit model to available GPU resources
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = in_channals * 4 * num_cells * num_cells, out_features=lsize, bias=True),
            # nn.Linear(in_features = 1024 *3 *3,  out_features=lsize, bias=True),    # 112x112
            nn.Dropout(0.01),
            nn.LeakyReLU(0.1),
            nn.Linear(lsize, num_classes, bias=False),
            nn.Sigmoid()
        )

    def _create_pretraining_fcs_3linear(self, in_channals, num_cells, num_boxes, num_classes):
        '''
        Creating classification layers for the pre-trained model runing as a classifier
        The output is an array of probabilities for each class.
        '''

        # Fit model to available GPU resources
        lsize = num_classes * 32

        return nn.Sequential(
            nn.Flatten(),
            # nn.Linear(in_features = in_channals * 4 * num_cells * num_cells, out_features=lsize, bias=True),
            nn.Linear(in_features = in_channals *3 *3,  out_features=lsize, bias=True),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.01),
            nn.Linear(in_features = lsize,  out_features=lsize, bias=True),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.01),
            nn.Linear(lsize, num_classes, bias=False),
            nn.Sigmoid()
        )

    def classifier_to_detection (self, **kwargs):
        self.reduction = self._create_conv_layers(reduction_architecture_config, 1024)
        self.fcs = self._create_main_fcs(1024, **kwargs)

    def cnn_freeze (self, is_freez=True):
        requires_grad = not is_freez
        for p in self.cnn.parameters():
            p.requires_grad = requires_grad


if __name__ == "__main__":
    # test model output
    x = torch.randn((8, 3, 448,448))
    pretrain_model = Model(num_cells=7, num_boxes=2, num_classes=20, model_type="pretraining")
    print(pretrain_model(x).shape)
    train_model = Model(num_cells=7, num_boxes=2, num_classes=20, model_type="training")
    print(train_model(x).shape)