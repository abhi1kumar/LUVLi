

"""
    Uses a mask on the landmarks to calculate the loss. Each landmark in data 
    point is first masked by the number of visible landmarks in the mask.
    The two versions are provided -
    1) One which calculates average of each image by all landmarks
    2) Another which calculates average of each image by only visible 
        landmarks.

    Version 1 Abhinav Kumar 2019-07-23
"""

import torch
import torch.nn as nn

import pylib.Constants as constants


class MaskedMSELoss(nn.Module):

    def __init__(self, use_all_landmarks= True):
        super(MaskedMSELoss, self).__init__()
        self.use_all_landmarks = use_all_landmarks
        self.mse_loss = nn.MSELoss(reduce= self.use_all_landmarks)       

    def forward(self, prediction, target, mask):
        """
            Gets the MSE based on the mask. Each landmark in data point is first
            masked by the number of visible landmarks in the mask.
            The two versions are provided -
            1) One which calculates average of each image by all landmarks
            2) Another which calculates average of each image by only visible 
                landmarks.
            
            Once the error of each image is known, we average it across all the 
            images to get the final loss as a single number

            prediction and target are assumed to be of the same shape.
            mask first and second dimensions should match
            
            Input:
                prediction : Variable batch_size x num_points x ...
                target: Variable batch_size x num_points x ...
                mask  : Variable batch_size x num_points

            Outputs:
                loss  : Variable scalar
        """
        batch_size = prediction.shape[0]
        num_points = prediction.shape[1]

        # Reshape the prediction and target in the common 3-dimensions
        prediction = prediction.view (batch_size, num_points, -1)
        target     = target.view(batch_size, num_points, -1)

        mask   = mask.unsqueeze(-1)
        if self.use_all_landmarks:
            loss = self.mse_loss(prediction*mask, target*mask)       # simple number
        else:
            loss   = self.mse_loss(prediction*mask, target*mask)     # batch_size x num_points x 3rd dim

            # We do a mean over the 3rd dimension.
            loss   = loss.mean(2)

            # We then add for the 1st dimension since only non-zeroed landmark is 
            # going to be summed and divided by number of visible landmarks
            loss   = loss.sum(1)                                         # batch_size

            # Unsqueeze the mask again and get how many landmarks are visible
            # for each image
            mask_sum = mask.squeeze(dim= 2).sum(1) + constants.EPSILON   # batch_size

            # Finally get the mean over visible landmarks for each image and finally
            # average over all images
            loss   = torch.mean(loss/mask_sum)

        return loss
