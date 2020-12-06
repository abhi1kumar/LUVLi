

"""
    Computes the Gaussian Regularisation of the PROBABILITY heatmaps. In other 
    words, if a batch of heatmaps is passed through this loss, it outputs the 
    distance between the PROBABILITY heatmaps and the Gaussian heatmap obtained 
    from the means and covariance of the probability heatmaps. The distance is 
    specified by the measure chosen while initialization (default = KL-divergence)

    Version 6 2019-07-04 Abhinav Kumar means_other introduced for showing the means
    Version 5 2019-07-02 Abhinav Kumar Epsilon added to log, swapped version included
    Version 4 2019-06-27 Abhinav Kumar Wrong concatenation issue fixed
    Version 3 2019-06-23 Abhinav Kumar Prob compared instead of log probabilities.
    Version 2 2019-06-09 Abhinav Kumar 0.5 factor missing in log_prob_2 fixed.
    Version 1 2019-06-09 Abhinav Kumar
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

import math

from pylib.CommonOperations import *
import pylib.Constants as constants
from pylib.HeatmapStats import *

class GaussianRegularizationLoss(nn.Module):

    def __init__(self, measure="nothing"):
        super(GaussianRegularizationLoss, self).__init__()
        self.measure = measure

        # measures could be
        # nothing
        # kld
        # kld_swap
        # hellinger
        # l1
        # l2
        print("Using measure {}".format(measure))
        self.kldiv   = nn.KLDivLoss(reduce=False)   # For pytorch>= 0.4.0 use reduction=None

        if self.measure == "l1":
            self.pdist = nn.PairwiseDistance(p=1)
        elif self.measure == "l2":
            self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, heatmaps, means, covariance, is_covariance=True, display= False, means_other= None):
        """
            Calculates the distance between the PROBABILITY heatmaps and the
            Gaussian heatmap obtained from the means and covariance of the prob
            heatmaps. The distance is specified by the measure chosen while
            initialization.
            TODO:
            Sigma calculation from L values if is_covariance is False

            :param heatmaps:           batch_size x 68 x 64 x 64 Variable
            :param means:              batch_size x 68 x 2       Variable
            :param covariance:         batch_size x 68 x 4       Variable
            :param is_covariance:      a flag which says whether we are passing
                                       Cholesky coefficients or the covariance
                                       itself.

            :return loss:              Gaussian Regularization Loss scalar
        """

        if self.measure=="nothing":
            loss = Variable(torch.zeros(1,))
            if heatmaps.is_cuda:
                loss = loss.cuda()
            return loss
            
        batch_size = heatmaps.shape[0]
        num_points = heatmaps.shape[1]
        height     = heatmaps.shape[2]
        width      = heatmaps.shape[3]

        # Get the grid first
        xv, yv = generate_grid(height, width)

        xv = Variable(xv)
        yv = Variable(yv)
        if heatmaps.is_cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        if is_covariance:
            Sigma             = covariance.contiguous().view(batch_size, num_points, 2, 2)    # batch_size x 68 x 2    x 2

        # Determinant of the covariance matrix
        det_Sigma = Sigma[:, :, 0, 0]*Sigma[:, :, 1, 1] - Sigma[:, :, 0, 1]* Sigma[:, :, 1, 0] + constants.EPSILON # batch_size x 68

        Sigma_inv = get_zero_variable_like(Sigma)

        # Inverse of 2D matrix 
        # [a b]
        # [c d] is
        #
        # (ad-bc)^{-1} *  [d  -b]
        #                 [-c  a]

        Sigma_inv[:,:,0,0] =  Sigma[:,:,1,1]
        Sigma_inv[:,:,1,1] =  Sigma[:,:,0,0]
        Sigma_inv[:,:,1,0] = -Sigma[:,:,1,0]
        Sigma_inv[:,:,0,1] = -Sigma[:,:,0,1]
        Sigma_inverse = Sigma_inv / expand_two_dimensions_at_end(det_Sigma, 2, 2)              # batch_size x 68 x 2 x 2

        xmean         = means[:,:,0]
        xv_minus_mean = xv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(xmean, height, width)  # batch_size x 68 x 64 x 64
        ymean         = means[:,:,1]  
        yv_minus_mean = yv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(ymean, height, width)  # batch_size x 68 x 64 x 64

        xv_minus_mean = xv_minus_mean.view(batch_size*num_points, 1, height*width)             # batch_size * 68 x 1    x 4096
        yv_minus_mean = yv_minus_mean.view(batch_size*num_points, 1, height*width)             # batch_size * 68 x 1    x 4096
        vec_concat    = torch.cat((xv_minus_mean, yv_minus_mean), 1)                           # batch_size * 68 x 2    x 4096
        vec_concat    = vec_concat.transpose(1,2)                                              # batch_size * 68 x 4096 x 2
        vec_concat    = vec_concat.contiguous().view(batch_size*num_points*height*width, 1, 2) # batch_size * 68 * 4096 x 1 x 2

        # Bivariate Normal distribution is given by
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
        #
        # P(x/N) =      1                           ___-1 
        #         _________________exp [ -(x-\mu)^T \    (x-\mu)  ]
        #                 ___  0.5     [            /__           ]
        #          2 \pi |\   |        [ ________________________ ]
        #                |/__ |        [            2             ]
        #
        # KL Divergence in Torch expects input to be log-probabilities while target to be probabilities only.
        # https://pytorch.org/docs/stable/nn.html#torch.nn.KLDivLoss
        # 
        # Taking the log, we get
        #                                     ___                      ___-1
        # log P(x/N) = -log(2\pi) - 0.5*log( |\   | ) - 0.5 *(x-\mu)^T \    (x-\mu)
        #                                    |/__ |                    /__
        #
        #            = -log(2\pi) + log_prob_1 + log_prob_2

        # Torch batch matrix multiplication
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        Sigma_inverse = Sigma_inverse.view(batch_size*num_points, 2, 2)                        # batch_size * 68 x 2 x 2
        Sigma_inverse = Sigma_inverse.view(batch_size*num_points, 1, 2, 2)                     # batch_size * 68 x 1 x 2 x 2
        Sigma_inverse = Sigma_inverse.expand(batch_size*num_points, height*width, 2, 2)        # batch_size * 68 x 4096 x 2 x 2
        Sigma_inverse = Sigma_inverse.contiguous().view(batch_size*num_points*height*width, 2, 2)       # batch_size * 68 * 4096 x 2 x 2

        temp          = torch.bmm( vec_concat, Sigma_inverse)                                  # batch_size * 68 * 4096 x 1 x 2
        term_2        = - 0.5 * torch.bmm(temp, vec_concat.transpose(1,2))                     # batch_size * 68 * 4096 x 1 x 1
        term_2        = term_2.squeeze(-1).squeeze(-1)                                         # batch_size * 68 * 4096
        log_prob_2    = term_2.view(batch_size, num_points, height, width)                     # batch_size x 68 x 64 x 64

        prob          = torch.exp(log_prob_2) / (2*math.pi * expand_two_dimensions_at_end(torch.sqrt(det_Sigma), height, width) ) # batch_size x 68 x 64 x 64

        # prob is sampled version of the continuous Gaussian distribution and therefore might not sum to 1
        # Normalize them again
        sum_prob      = torch.sum(torch.sum(prob, dim=3), dim=2)
        sum_prob      = expand_two_dimensions_at_end(sum_prob, height, width)                  # batch_size x 68 x 64 x 64
        prob          = prob/sum_prob                                                          # batch_size x 68 x 64 x 64

        # ============================================
        # The following code is for testing purposes
        # ============================================
        if display:
            means_recomputed, covar_recomputed, _ = get_spatial_mean_and_covariance_improved(prob, use_softmax=False, tau=0.0003, postprocess="relu")
            print(means_recomputed[0,0].data)
            print(covar_recomputed[0,0].data)
            import matplotlib.pyplot as plt
            import numpy as np
            t = heatmaps[0,0].cpu().clone().data.numpy()
            vmax = np.max(t)
            vmin = np.min(t)
            plt.subplot(121)
            plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
            plt.plot(means[0,0,0].data.numpy(), means[0,0,1].data.numpy(), marker= 'x', markersize= 8, color='green')
            if means_other is not None:
                plt.plot(means_other[0,0,0].data.numpy(), means[0,0,1].data.numpy(), marker= 'x', markersize= 8, color='black')
            t = prob[0,0].cpu().clone().data.numpy()
            plt.subplot(122)
            plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
            plt.plot(means[0,0,0].data.numpy(), means[0,0,1].data.numpy(), marker= 'x', markersize= 8, color='green')
            if means_other is not None:
                plt.plot(means_other[0,0,0].data.numpy(), means[0,0,1].data.numpy(), marker= 'x', markersize= 8, color='black')
            plt.show()

        # KLDiv calculates kld along each rows. So reshape the two vectors
        heatmaps      = heatmaps.contiguous().view(batch_size*num_points, height*width)       # batch_size * 68 x 4096
        prob          = prob.view(batch_size*num_points, height*width)                        # batch_size * 68 x 4096

        loss = 0.
        if (self.measure == "kld"):
            # the input (x) given is expected to contain log-probabilities. The
            # targets (y) are given as probabilities
            # Loss = KL(x, y) = y(logy - x)
            # https://pytorch.org/docs/stable/nn.html#torch.nn.KLDivLoss
            loss = self.kldiv(torch.log(prob + constants.EPSILON), heatmaps.clone().detach())
            loss = torch.sum(loss, 1)                                                         # batch_size * 68
        elif (self.measure == "kld_swap"):
            # Using swapped version of the KLD

            # The input (x) given is expected to contain log-probabilities. The
            # targets (y) are given as probabilities
            # Loss = KL(x, y) = y(logy - x)
            # https://pytorch.org/docs/stable/nn.html#torch.nn.KLDivLoss
            loss = self.kldiv(torch.log(heatmaps + constants.EPSILON), prob.clone().detach())
            loss = torch.sum(loss, 1)                                                         # batch_size * 68
        elif (self.measure == "hellinger"):
            loss = 1.0 - torch.sum(torch.sqrt(heatmaps * prob), 1)                            # batch_size * 68
        elif (self.measure == "l1" or self.measure == "l2"):
            loss = self.pdist(prob, heatmaps)                                                 # batch_size * 68
    
        loss = loss.mean()

        return loss
