

"""
    Spatial Mean and Covariance Calculation from the heatmaps

    Version 6 Abhinav Kumar 2019-07-02 Improved heatmaps calculation by taking softmax for mean and mirror images for covariances
    Version 5 Abhinav Kumar 2019-06-09 Normalized heatmaps also returned
    Version 4 Abhinav Kumar 2019-06-06 Heatmaps as weights applied at one place instead of sqrt at two places
    Version 3 Abhinav Kumar 2019-06-05 Covariance calculation added
    Version 2 Abhinav Kumar 2019-06-03 post_processing new function and variables automated based on type
    Version 1 Abhinav Kumar 2019-06-01
"""

import torch
from torch.autograd import Variable

import sys
sys.path.insert(0, './pylib')
from CommonOperations import *
import pylib.Constants as constants
import torch.nn.functional as F

def softplus(input, beta = 1.0):
    """
        Returns softplus of the input
        Reference https://pytorch.org/docs/stable/nn.html#torch.nn.Softplus
    """
    output = torch.log(1 + torch.exp(beta*input)) / beta

    return output

def relu(input):
    """
        Returns relu of the input
    """
    zeros = get_zero_variable_like(input)      

    return torch.max(input, zeros)

def post_process_input(input, postprocess= None):
    if postprocess=="abs":
        output = torch.abs(input)
    elif postprocess=="softplus":
        output = softplus(input)
    elif postprocess=="relu":
        output = relu(input)
    else:
        output = input

    return output


def post_process_and_normalize(heatmaps, use_softmax, tau, postprocess):
    """
        Post process and then normalize to sum to 1
        Input : heatmaps     = batch_size x 68 x 64 x 64 Variable
                use_softmax  = Boolean
                tau          = scaling parameter of the softmax [float > 0]
                postprocess  = string

        Output: htp          = batch_size x 68 x 64 x 64 Variable
    """    

    batch_size = heatmaps.shape[0]
    num_points = heatmaps.shape[1]
    height     = heatmaps.shape[2]
    width      = heatmaps.shape[3]

    # Apply post processing
    htp  = post_process_input(heatmaps, postprocess)

    if use_softmax:
        # Small tau can blow up the numerical values. Use numerically stable
        # softmax by first subtracting the max values from the individual heatmap
        # Use https://stackoverflow.com/a/49212689
        m   = htp.view(batch_size * num_points, height * width)
        m,_ = torch.max(m, 1)                                                                # batch_size
        m   = m.view(batch_size, num_points)                                                 # batch_size x 68
        m   = expand_two_dimensions_at_end(m, height, width)                                 # batch_size x 68 x 64 x 64
        htp = htp - m        
        htp = torch.exp(htp/tau) 

    # Add a small EPSILON for case sum_2 entries are  all zero
    sum2 = get_channel_sum(htp) + constants.EPSILON
    # Get the normalized heatmaps
    htp  = htp/(sum2.view(htp.size(0),htp.size(1),1,1) )

    return htp

def get_channel_sum(input):
    """
        Generates the sum of each channel of the input
        input  = batch_size x 68 x 64 x 64
        output = batch_size x 68
    """
    temp   = torch.sum(input, dim=3)
    output = torch.sum(temp , dim=2)
    
    return output

def get_spatial_mean_along_axis(xv, htp, sum_htp):
    """
        Gets spatial mean along one of the axis.
        Input : htp          = batch_size x 68 x 64 x 64
        Output: means        = batch_size x 68
    """
    batch_size = htp.shape[0]
    num_points = htp.shape[1]    
    height     = htp.shape[2]
    width      = htp.shape[3]

    # x coord * heatmap
    x_times_htp = xv.expand(batch_size,num_points,-1,-1)*htp

    # sume of x coord times heatmap
    s_x = get_channel_sum(x_times_htp)

    # x predicted pts
    # Add a small nudge when sum_htp is all zero
    x_pts = s_x/(sum_htp + constants.EPSILON)

    return x_pts

def get_spatial_mean(htp):
    """
        Gets the spatial mean of each of the heatmap from the batch of
        normalized heatmaps.
        Input : htp          = batch_size x 68 x 64 x 64
        Output: means        = batch_size x 68 x 2

        Convention:
        |----> X (0th coordinate)
        |
        |
        V Y (1st coordinate)
    """
    batch_size = htp.shape[0]
    num_points = htp.shape[1]    
    height     = htp.shape[2]
    width      = htp.shape[3]

    # htp is the normalized heatmap
    sum_htp = get_channel_sum(htp)                                                             # batch_size x 68

    xv, yv = generate_grid(height, width)
    xv = Variable(xv)
    yv = Variable(yv)

    if htp.is_cuda:
        xv = xv.cuda()
        yv = yv.cuda()

    x_pts = get_spatial_mean_along_axis(xv, htp, sum_htp)
    y_pts = get_spatial_mean_along_axis(yv, htp, sum_htp)

    means = torch.cat((x_pts.view(batch_size,num_points,1), y_pts.view(batch_size,num_points,1)), 2)

    return means

def get_covariance_matrix(htp, means):
    """
        Covariance calculation from the normalized heatmaps
        Reference https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
        The unbiased estimate is given by
        Unbiased covariance =
              ___
              \
              /__ w_i (x_i - \mu_i)^T (x_i - \mu_i)
               
          ___________________________________________

                        V_1 - (V_2/V_1)

                    ___                 ___
                    \                   \  
        where V_1 = /__ w_i   and V_2 = /__ w_i^2


        Input:
            htp =        batch_size x 68 x 64 x 64
            means =      batch_size x 68 x 2

        Output:
            covariance = batch_size x 68 x 2  x 2
    """
    batch_size = htp.shape[0]
    num_points = htp.shape[1]
    height     = htp.shape[2]
    width      = htp.shape[3]

    xv, yv = generate_grid(height, width)
    xv = Variable(xv)
    yv = Variable(yv)

    if htp.is_cuda:
        xv = xv.cuda()
        yv = yv.cuda()

    xmean         = means[:,:,0]
    xv_minus_mean = xv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(xmean, height, width)  # batch_size x 68 x 64 x 64
    ymean         = means[:,:,1]  
    yv_minus_mean = yv.expand(batch_size, num_points, -1, -1) - expand_two_dimensions_at_end(ymean, height, width)  # batch_size x 68 x 64 x 64

    # These are the unweighted versions
    wt_xv_minus_mean = xv_minus_mean
    wt_yv_minus_mean = yv_minus_mean 

    wt_xv_minus_mean = wt_xv_minus_mean.view(batch_size*num_points, height*width)            # batch_size*68 x 4096
    wt_xv_minus_mean = wt_xv_minus_mean.view(batch_size*num_points, 1, height*width)         # batch_size*68 x 1    x 4096
    wt_yv_minus_mean = wt_yv_minus_mean.view(batch_size*num_points, height*width)            # batch_size*68 x 4096
    wt_yv_minus_mean = wt_yv_minus_mean.view(batch_size*num_points, 1, height*width)         # batch_size*68 x 1    x 4096
    vec_concat       = torch.cat((wt_xv_minus_mean, wt_yv_minus_mean), 1)                    # batch_size*68 x 2    x 4096

    htp_vec = htp.view(batch_size*num_points, 1, height*width)
    htp_vec = htp_vec.expand(-1, 2, -1)

    # Torch batch matrix multiplication
    # https://pytorch.org/docs/stable/torch.html#torch.bmm
    # Also use the heatmap as the weights at one place now
    covariance       = torch.bmm(htp_vec*vec_concat, vec_concat.transpose(1,2))              # batch_size*68 x 2    x 2
    covariance       = covariance.view(batch_size, num_points, constants.num_dim_image, constants.num_dim_image) # batch_size    x 68   x 2   x 2

    V_1              = get_channel_sum(htp) + constants.EPSILON                              # batch_size x 68
    V_2              = get_channel_sum(torch.pow(htp, 2))                                    # batch_size x 68
    denominator      = V_1 - (V_2/V_1)

    covariance       = covariance / expand_two_dimensions_at_end(denominator, constants.num_dim_image, constants.num_dim_image)

    return (covariance)

def get_spatial_mean_and_covariance(heatmaps, use_softmax= False, tau= 0.02, postprocess= None):
    """
        Gets the spatial mean and covariance from the batch of heatmaps.
        post_process describes how the heatmap is processed. If
        use_softmax flag is on, tau is used otherwise tau is not used.

        Input : heatmaps     = batch_size x 68 x 64 x 64 Variable
                use_softmax  = Boolean
                tau          = scaling parameter of the softmax [float > 0]
                postprocess  = string

        Output: means        = batch_size x 68 x 2      Variable
                covariance   = batch_size x 68 x 2  x 2 Variable
    """
    # Post process and normalize
    htp = post_process_and_normalize(heatmaps, use_softmax= use_softmax, tau= tau, postprocess= postprocess)

    # Get means and covariance
    means       = get_spatial_mean     (htp)
    covariance  = get_covariance_matrix(htp, means)

    return means, covariance, htp

def get_spatial_mean_and_covariance_improved(heatmaps, use_softmax= False, tau= 0.02, postprocess= None, special_mean= False, tau_mean= 0.00003):
    """
        This function first improves the means and covariance calculation by 
        taking softmax for the mean calculation if special_mean variable is set 
        to True. This is especially needed when 
        the covariance of the heatmap is big and at the edges. The covariance is
        obtained by appending the 180 degree reflection of the heatmap to the
        original heatmap and then calculating the covariance.

        One tau is the normal tau for postprocessing and other tau_mean is used 
        for mean calculation.

        NOTE= This method works when half of the heatmap is visible and if the
        mean is inside the heatmap.

        Input : heatmaps     = batch_size x 68 x 64 x 64 Variable
                use_softmax  = Boolean
                tau          = scaling parameter of the softmax [float > 0]
                postprocess  = string
                special_mean = Boolean
                tau_mean     = scaling parameter of the softmax for mean [float > 0]

        Output: means        = batch_size x 68 x 2      Variable
                covariance   = batch_size x 68 x 2  x 2 Variable

        --- NOT used anymore ---
    """

    batch_size = heatmaps.shape[0]
    num_points = heatmaps.shape[1]
    height     = heatmaps.shape[2]
    width      = heatmaps.shape[3]

    if special_mean:
        # Get mean from the differentiable tau version of the heatmaps.
        # It is always calculated from the use_softmax version with tau_mean being 
        # fixed.
        # tau= 0.0000003 value works for covariance matrix of 1000 * identity
        heatmaps_mean  = post_process_and_normalize(heatmaps, use_softmax= True, tau= tau_mean, postprocess= postprocess)
        means_cal      = get_spatial_mean     (heatmaps_mean)
    else:
        heatmaps_mean  = post_process_and_normalize(heatmaps, use_softmax= use_softmax, tau= tau, postprocess= postprocess)
        means_cal      = get_spatial_mean     (heatmaps_mean)

    """
    # Indexes will be integers only
    means_int = torch.ceil(means_cal.clone()).type(torch.LongTensor)

    # Get a new variable which is thrice times in each axis of the heatmaps
    heatmaps_big = Variable(torch.zeros((batch_size, num_points, 3*height, 3*width)).float())
    if heatmaps.is_cuda:
        heatmaps_big = heatmaps_big.cuda()

    # DO NOT use .data anywehere otherwise it would disconnect the variable from 
    # the graph and also does not copy as expected. Using .data in normal 
    # variables makes means and covariance wrong
    heatmaps_big[:, :, height:2*height, width:2*width] = heatmaps

    # TODO: Check if OK to do data on indexing variables
    # Convention of axis for means is
    #    |----> X (0th coordinate)
    #    |
    #    |
    #    V Y (1st coordinate)
    means_col = means_int[:, :, 0].data
    means_row = means_int[:, :, 1].data

    # TODO: Make this vectorised
    for i in range(batch_size):
        for j in range(num_points):
            means_r = means_row[i ,j]
            means_c = means_col[i, j]

            # Now add reflections with means as the centers
            if means_r > 0 and means_c < width:        
                quad1 = heatmaps[i, j, 0: means_r, means_c: width]
                heatmaps_big[i, j, height+means_r: height + 2*means_r, 2*means_c    : width+means_c]   = flip180_tensor(quad1)

            if means_r > 0 and means_c > 0:
                quad2 = heatmaps[i, j, 0: means_r, 0: means_c]
                heatmaps_big[i, j, height+means_r: height + 2*means_r, width+means_c: width+2*means_c] = flip180_tensor(quad2)

            if means_r < height and means_c > 0:
                quad3 = heatmaps[i, j, means_r: height, 0: means_c]
                heatmaps_big[i, j, 2*means_r     : height+means_r    , width+means_c: width+2*means_c] = flip180_tensor(quad3)

            if means_r < height and means_c < width:
                quad4 = heatmaps[i, j, means_r: height, means_c:width]
                heatmaps_big[i, j, 2*means_r     : height+means_r    , 2*means_c    : width+means_c]   = flip180_tensor(quad4)

    # Calculate covariance matrix from the means_cal values rather than its own 
    # values.
    heatmaps_covar = post_process_and_normalize(heatmaps_big, use_softmax= use_softmax, tau= tau, postprocess= postprocess)

    # Assumed here that the width and height are same and therefore only one 
    # of them needs to be added.
    # Addition required since means_cal is calculated over original heatmap 
    # while the covar is calculated from the big heatmap
    covar_cal      = get_covariance_matrix(heatmaps_covar, means_cal+ width)

    return means_cal, covar_cal, heatmaps_covar[:, :, height: 2*height, width:2*width]
    """
    scale    = 1
    #heatmaps = F.upsample(heatmaps, scale_factor= scale, mode='bilinear')

    heatmaps_covar = post_process_and_normalize(heatmaps, use_softmax= use_softmax, tau= tau, postprocess= postprocess)
    covar_cal      = get_covariance_matrix(heatmaps_covar, means_cal*scale)

    return means_cal, covar_cal/(scale**2), heatmaps_covar
