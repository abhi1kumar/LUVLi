

"""
    Calculates the Laplacian or Gaussian Log Likelihood Loss on the list of predicted
    landmarks and their ground truth. The list corresponds to different stages 
    of the network.

    Version 3 Abhinav Kumar 2019-06-11 torch.bmm introduced, CommonOperations used
    Version 2 Abhinav Kumar 2019-06-06 cuda().cpu() in log_probabilities removed, EPSILON introduced
    Version 1 Abhinav Kumar 2019-06-06 Support for covariance for heatmaps added
    Original  Wenxuan Mou
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from pylib.CommonOperations import *
import pylib.Constants as constants


class FaceAlignLoss(nn.Module):

    def __init__(self, laplacian= False, form= "simplified"):
        super(FaceAlignLoss, self).__init__()

        self.exclude_self_occluded_pts = False                                  # Flag to exclude self occluded points
        self.lambda_                   = 1                                      # The weight for the loss_term2

        self.laplacian = laplacian
        self.a         = 0.25
        self.form      = form       
        
        if self.laplacian:
            if self.form == "asymmetric":
                print("Using Asymmetric Laplacian Log Likelihood Loss, a= {}".format(self.a))
            elif self.form == "symmetric":
                print("Using Symmetric Laplacian Log Likelihood Loss, a= {}".format(self.a))
            else:
                print("Using Simplified Laplacian Log Likelihood")
        else:
            print("Using Gaussian Log Likelihood Loss")
            
    def get_mahalanobis_distance(self, x, Sigma_inverse, y):
        """
            Returns x^T Sigma_inverse y
            :param x:             batch_size x 68 x 2
            :param Sigma_inverse: batch_size x 68 x 2 x 2
            :param y:             batch_size x 68 x 2
            
            :return: product of size batch_size x 68
        """
        batch_size            = Sigma_inverse.shape[0]                          # batch_size
        num_points            = Sigma_inverse.shape[1]
        
        x_vec = x.unsqueeze(-1).contiguous().view(batch_size * num_points, 2, 1)             # batch_size * 68 x 2 x 1
        y_vec = y.unsqueeze(-1).contiguous().view(batch_size * num_points, 2, 1)             # batch_size * 68 x 2 x 1
        Sigma_inverse = Sigma_inverse.view(batch_size * num_points, 2, 2)       # batch_size * 68 x 2 x 2

        # Torch batch matrix multiplication
        # https://pytorch.org/docs/stable/torch.html#torch.bmm
        product       = torch.bmm( torch.bmm(x_vec.transpose(1,2), Sigma_inverse), y_vec)    # batch_size * 68 x 1 x 1        
        product = product.squeeze(-1).squeeze(-1)                               # batch_size * 68
        product = product.view(batch_size, num_points)                          # batch_size x 68
        
        Sigma_inverse = Sigma_inverse.view(batch_size, num_points, 2, 2)        # batch_size x 68 x 2 x 2
        x_vec = x.squeeze(-1).contiguous().view(batch_size, num_points, 2)
        y_vec = y.squeeze(-1).contiguous().view(batch_size, num_points, 2)
        return product

    def log_probabilities(self, landmarks, ground_truth, is_covariance=False):
        """
            Actual function that implements the Gaussian Log Likelihood loss (It returns positive log likelihood)

            :param landmarks:     batch_size x 68 x 5 (5 entries - 2 for spatial mean and 3 for entries of Cholesky decomposition of covrariance matrix)
                                  batch_size x 68 x 6 (6 entries - 2 for spatial mean and 4 for the covariance matrix itself) if is_covariance is True
            :param ground_truth:  batch_size x 68 x 3 (2 for coordinates and one for occluded or not)
            :param is_covariance: a flag which says whether we are passing Cholesky coefficients or the covariance itself.

            :return: log probabilities of groundtruth locations given estimated landmark distributions.  batch_size x 68
        """
        batch_size            = landmarks.shape[0]                              # batch_size
        num_points            = landmarks.shape[1]

        # Mean of Gaussian (x,y)
        estimated_location    = landmarks[:, :, :2]                             # batch_size x 68 x 2

        # Landmark ground truth location
        ground_truth_location = ground_truth[:, :, :2]                          # batch_size x 68 x 2

        # If covariance is supplied
        if is_covariance:
            Sigma             = landmarks[:, :, 2:6]                            # batch_size x 68 x 4
            Sigma             = Sigma.contiguous().view(batch_size, num_points, 2, 2)

        else:
            # Vectorized lower triangular Cholesky factor of cov matrix. Sigma = LL^T
            L_vect                = landmarks[:,:,2:5]

            L_mat = get_zero_variable((batch_size, num_points, 2 ,2), landmarks)
        
            # Reshape L_0, L_1, L_2 into 2x2 Lower triangular matrix:
            #            [ L_0   0  ]
            #            [ L_1  L_2 ]
            L_mat[:, :, 0, 0] = L_vect[:, :, 0]                                 
            L_mat[:, :, 1, 0] = L_vect[:, :, 1]                                 
            L_mat[:, :, 1, 1] = L_vect[:, :, 2]                                 

            # Covariance matrix Sigma = LL^T
            Sigma = torch.matmul(L_mat, L_mat.transpose(2, 3))                  

        # Determinant of Sigma
        det_Sigma = Sigma[:, :, 0, 0] * Sigma[:, :, 1, 1] - Sigma[:, :, 0, 1] * Sigma[:, :, 1, 0] + constants.EPSILON # batch_size x 68

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
        Sigma_inverse = Sigma_inv / det_Sigma.unsqueeze(-1).unsqueeze(-1)       # batch_size x 68 x 2 x 2
        
        if self.laplacian:
            if self.form == "asymmetric":
                # Bivariate Laplacian Distribution is given by
                # https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution#Asymmetric_multivariate_Laplace_distribution
                # v = (2-k)/2 = 0
                #                       ___-1                                                       0.5
                #  P(x/N) =  2 exp[ x^T \    \mu ]        [                                        ]
                #                 [     /__      ]        [            ___-1             ___-1     ]
                #            _______________________  K_0 [ [2 + \mu^T \    \mu ]  [ x^T \    x ]  ]
                #                    ___  0.5             [ [          /__      ]  [     /__    ]  ]
                #             2 \pi |\   |                [                                        ]
                #                   |/__ |        
                # 
                # Using an approximation of Bessel function of second kind
                # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5306441/
                #                       ___-1                                                               0.5
                #  P(x/N) =  2 exp[ x^T \    \mu ]                [                                        ]
                #                 [     /__      ]                [            ___-1             ___-1     ]
                #            ______________________sqrt(\pi) exp -[ [2 + \mu^T \    \mu ]  [ x^T \    x ]  ]
                #                    ___  0.5                     [ [          /__      ]  [     /__    ]  ]
                #             2 \pi |\   |                        [                                        ]
                #                   |/__ |            ______________________________________________________        0.5 
                #                                             [                                            0.5     ]
                #                                     sqrt(2) [  [                                        ]        ]
                #                                             [  [            ___-1             ___-1     ]        ]
                #                                             [  [ [2 + \mu^T \    \mu ]  [ x^T \    x ]  ]   +  a ]
                #                                             [  [ [          /__      ]  [     /__    ]  ]        ]
                #                                             [  [                                        ]        ]
                #
                #
                # Taking log, we get
                #                                                      ___               ___-1         [                                        ]0.5            [ [                                        ]0.5    ]
                # log P(x/N) = log(2) -  0.5*log(0.5*\pi)  - 0.5*log( |\   | )    +  x^T \    \mu  -   [            ___-1             ___-1     ]    - 0.5 * ln [ [            ___-1             ___-1     ]       ]
                #                                                     |/__ |             /__           [ [2 + \mu^T \    \mu ]  [ x^T \    x ]  ]               [ [ [2 + \mu^T \    \mu ]  [ x^T \    x ]  ]   + a ]
                #                                                                                      [ [          /__      ]  [     /__    ]  ]               [ [ [          /__      ]  [     /__    ]  ]       ]
                #                                                                                      [                                        ]               [ [                                        ]       ]
                #
                #            = log(2) -  0.5*log(0.5*\pi)  + log_prob_1 + term_2_1 + term_2_2 + term_2_3
                #            = log(2) -  0.5*log(0.5*\pi)  + log_prob_1 + log_prob_2
                # In asymmetric this is the form.

                term_2_1          = self.get_mahalanobis_distance(ground_truth_location, Sigma_inverse, estimated_location)    # batch_size x 68
                
                term_2_mu_product = 2 + self.get_mahalanobis_distance(estimated_location, Sigma_inverse, estimated_location)   # batch_size x 68
                term_2_x_product  = self.get_mahalanobis_distance(ground_truth_location, Sigma_inverse, ground_truth_location) # batch_size x 68            
                term_2_temp       = torch.sqrt (term_2_mu_product* term_2_x_product)
            
                term_2_2          = - term_2_temp            
                term_2_3          = -0.5 * torch.log(term_2_temp + self.a)                
            
            elif self.form == "symmetric":
                # In case of symmetric, set \mu to zero and replace x by x - \mu
                #                                                      ___              [                                ]0.5            [ [                                    ]0.5    ]
                # log P(x/N) = log(2) -  0.5*log(0.5*\pi)  - 0.5*log( |\   | )      -   [              ___-1             ]    - 0.5 * ln [ [                   ___-1            ]       ]
                #                                                     |/__ |            [ [ (x- \mu)^T \    (x - \mu) ]  ]               [ [  2   [ (x- \mu)^T \    (x- \mu) ]  ]   + a ]
                #                                                                       [ [            /__            ]  ]               [ [      [            /__           ]  ]       ]
                #                                                                       [                                ]               [ [                                    ]       ]
                #            = log(2) -  0.5*log(0.5*\pi)  + log_prob_1 + 0 + term_2_2 + term_2_3
                #            = log(2) -  0.5*log(0.5*\pi)  + log_prob_1 + log_prob_2
                
                term_2_1          = 0.   # batch_size x 68
                
                term_2_mu_product = 2    # batch_size x 68
                term_2_x_product  = self.get_mahalanobis_distance(ground_truth_location-estimated_location, Sigma_inverse, ground_truth_location-estimated_location) # batch_size x 68            
                term_2_temp       = torch.sqrt (term_2_mu_product * term_2_x_product)
                
                term_2_2          = - term_2_temp            
                term_2_3          = -0.5 * torch.log(term_2_temp + self.a)
            else:
                # In case of simplified, we do not have any inequality but
                #                                         ___              [                                 ]0.5  
                # log P(x/N) = -log(2*\pi/3)  - 0.5*log( |\   | )      -   [               ___-1             ]    
                #                                        |/__ |            [ 3[ (x- \mu)^T \    (x - \mu) ]  ]    
                #                                                          [  [            /__            ]  ]    
                #                                                          [                                 ]    
                #            = -log(2*\pi)  + log_prob_1 + 0 + term_2_2 + log(3)
                #            = -log(2*\pi)  + log_prob_1 + 0 + term_2_2 + term_2_3
                #            = -log(2*\pi)  + log_prob_1 + log_prob_2
                term_2_x_product  = self.get_mahalanobis_distance(ground_truth_location-estimated_location, Sigma_inverse, ground_truth_location-estimated_location) # batch_size x 68 
                term_2_1          = 0.   # batch_size x 68
                term_2_2          = -torch.sqrt (term_2_x_product*3.0)  
                term_2_3          = torch.log  (3.0*Variable(torch.ones(1,).cuda()))
         
            # Determinant of the covariance matrix
            log_prob_1 = -0.5 * torch.log(det_Sigma)                                # batch_size x 68
            log_prob_2 = term_2_1 + term_2_2 + term_2_3
            
        else:
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

            # Determinant of the covariance matrix
            log_prob_1 = -0.5 * torch.log(det_Sigma)                                # batch_size x 68

            # Mahalanobis distance calculation
            log_prob_2 = -0.5 * self.get_mahalanobis_distance(ground_truth_location-estimated_location, Sigma_inverse, ground_truth_location-estimated_location) # batch_size x 68

        # return loss_term1, loss_term2 separately
        return log_prob_1, log_prob_2

    def forward_one_stage(self, landmarks, ground_truth, is_covariance=False):
        """
            :param landmarks:     batch_size x 68 x 5 (5 entries - 2 for spatial mean and 3 for entries of Cholesky decomposition of covariance matrix)
                                  batch_size x 68 x 6 (6 entries - 2 for spatial mean and 4 for the covariance matrix itself) if is_covariance is True
            :param ground_truth:  batch_size x 68 x 3 (2 for coordinates and 1 for landmark being occluded or not)
            :param is_covariance: a flag which says whether we are passing Cholesky coefficients or the covariance itself.

            :return: the losses for this stage all scalars
        """
        estimated_location      = landmarks[:,:,:2]
        ground_truth_location   = ground_truth[:,:,:2]

        # Get positive log likelihood
        loss_term1, loss_term2  = self.log_probabilities(landmarks, ground_truth, is_covariance=is_covariance)
            
        # self occluded points are also used for loss calculation
        # Remember to multiply by -1 so that positive log likelihood gets converted to negative log likelihood
        loss_term1 = -torch.mean(loss_term1, dim = 1).mean()
        loss_term2 = -self.lambda_ * torch.mean(loss_term2, dim = 1).mean()

        loss_Gaussian = loss_term1 + loss_term2

        return loss_term1, loss_term2, loss_Gaussian

    def forward(self, list_of_landmarks, ground_truth, is_covariance=False):
        """
            Calculates the Gaussian Log Likelihood Loss on the list of estimates
            of the landmarks and their ground truth. The list corresponds to
            different stages of the network.
            
            :param list_of_landmarks:  batch_size x 68 x 5
                                       batch_size x 68 x 6 (6 entries - 2 for spatial mean and 4 for the covariance matrix itself) if is_covariance is True
            :param ground_truth:       batch_size x 68 x 3
            :param is_covariance: a flag which says whether we are passing Cholesky coefficients or the covariance itself.

            :return loss:              Gaussian loss summed over all stage
            :       loss_stages:       Gaussian loss individually per stage
            :       loss_term1 :       Term1 in Gaussian loss summed over all stages
            :       loss_term2:        Term2 in Gaussian loss summed over all stages
            :       loss_term1_stages: Term1 list individually per stage
            :       loss_term2_stages: Term2 list individually per stage 
        """
        loss_term1_stages = []
        loss_term2_stages = []
        loss_stages       = []

        loss       = 0.                            # loss_term1 + loss_term2
        loss_term1 = 0.
        loss_term2 = 0.

        # Checking for the variables final dimensions to be 6
        if is_covariance:
            for i in range(len(list_of_landmarks)):
                assert list_of_landmarks[i].shape[2] == 6

        # For each stage
        for stage,landmarks in enumerate(list_of_landmarks):
            loss_1, loss_2, loss_Gaussian = self.forward_one_stage(landmarks, ground_truth, is_covariance=is_covariance)
            loss_stages.append(loss_Gaussian)      # total loss of all stages
            loss_term1_stages.append(loss_1)       # loss_term1 of all stages
            loss_term2_stages.append(loss_2)       # loss_term2 of all stages

            loss = loss + loss_Gaussian            # the sum of of total loss of all stages
            loss_term1 = loss_term1 + loss_1       # the sum of loss_term1 of all stages
            loss_term2 = loss_term2 + loss_2       # the sum of loss_term2 of all stages

        # Gaussian_loss, Gaussian Loss in stages, Loss_term1, Loss_term2, Loss_term1 in stages, Loss_term in stages
        return loss, loss_stages, loss_term1, loss_term2, loss_term1_stages, loss_term2_stages
