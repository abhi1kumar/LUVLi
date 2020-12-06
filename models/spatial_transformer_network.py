

"""
    Spatial Transformer Network
    References-
    Spatial Transformer Networks, Jaderberg et al, NeurIPS 2017
    https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html

    Version 1 2019-07-11 Abhinav Kumar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerNetwork(nn.Module):
    """
        A class which creates SpatialTransformer Network.
    """
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()

        # Convolution part of the Spatial Transformer Network
        # The parameters for conv2d are different from 
        # https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        # so that the output of localization is 10 x 3 x 3 image from 64 x 64 
        # image.
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 8, kernel_size= 7, stride= 2, padding= 3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels= 8, out_channels= 10, kernel_size= 5, stride= 2, padding=0),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        # Newer version
        # https://discuss.pytorch.org/t/torch-tensor-doesnt-work-provides-following-error-typeerror-module-object-is-not-callable/16903/8
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0.]))        
        self.fc_loc[2].bias.data.copy_(torch.FloatTensor([1, 0, 0, 0, 1, 0.]))

    def forward(self, x, process_channels_separately= True):
        """
            The channels of a single images are samples in the identical manner.
            https://pytorch.org/docs/stable/nn.html#torch.nn.functional.grid_sample

            If we want to have different transformations for each of the 
            channels or we want to process channels separately, we first reshape
            the channels to the batch dimensions and pass through the Spatial
            Transformer and then reshape again.
        """
        batch_size = x.shape[0]
        num_points = x.shape[1]
        height     = x.shape[2]
        width      = x.shape[3]

        if process_channels_separately:
            x = x.view(-1, 1, height, width)

        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        if process_channels_separately:
            x = x.view(-1, num_points, height, width)

        return x
