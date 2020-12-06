
import os, sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

from loss.gaussian_loss import *
import torch.nn as nn 

def correct_laplacian(input, form):
    correction_for_simplified = 1.0986 # Simplified Laplacian has an extra term as torch.log(3) = 1.0986
    return input

def correct_l2loss(input):
    """
        Our Laplacian likelihood is kind of square root Gaussian loss.
        Specifically, our each entry of 2D Laplacian = sqrt(Mahalanobis distance)
        2D Gaussian = Mahalanobis distance/2
        
        Hence, output of each entry of 2D Laplacian = sqrt(2 * each entry of output of 2D Gaussian)
        
    """  
    # We have to do elementwise processing first and then report the mean
    return torch.mean(torch.sqrt(2.0 * input))

def scale_covariance(input, form, is_covariance):
    if form == "simplified":
        if is_covariance:
            # Multiplying here by 3 since we are using covariance
            return input*3.0
        else:
            # Multiplying here by sqrt(3) since we are using Cholesky
            return input*1.732
    else:
        return input
                
form= "simplified"
loss_fn  = FaceAlignLoss(laplacian= True, form= form)
mae_loss = nn.MSELoss(reduce= False) # We have to do elementwise processing first

b = torch.from_numpy(np.array([[[12,13], [10,11],  [21,22]],[ [30,31], [41,42], [51,52.]]]))
mask = torch.from_numpy(np.array([[1, 0, 1],[0,1,1.]]))
covar = torch.from_numpy(np.array([[[1,0,0,1], [1,0,0,1],  [1,0,0,1]],[ [1,0,0,1], [1,0,0,1], [1,0,0,1.]]]))
covar = scale_covariance(covar, form, is_covariance= True)

b = Variable(b.float().cuda())
mask = Variable(mask.float().cuda())
covar = Variable(covar.float().cuda())
gt = b.clone() + 10.

print(b.data)
#print(mask.data)
#print(gt.data)
#print(covar.data)

print("================== Without Masking and with covariance =============")
pre_pts = torch.cat((b, covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt, is_covariance= True)
print(correct_laplacian(loss_gau.data, form))

loss_mae = correct_l2loss(mae_loss(b, gt))
print(loss_mae.data)


print("================== With Masking and with covariance =============")
print((b*mask.unsqueeze(-1)).data)
pre_pts = torch.cat((b*mask.unsqueeze(-1), covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt*mask.unsqueeze(-1), is_covariance= True)
print(correct_laplacian(loss_gau.data, form))

loss_mae = correct_l2loss(mae_loss(b*mask.unsqueeze(-1), gt*mask.unsqueeze(-1)))
print(loss_mae.data)

# ==============================================================================
# Non-zero values and return correctly
# ==============================================================================
# These are the lower triangular matrices of the covariance matrix
covar = torch.from_numpy(np.array([[[1,0,1], [1,0,1],  [1,0,1]],[ [1,0,1], [1,0,1], [1,0,1.]]]))
covar = scale_covariance(covar, form, is_covariance= False)

covar = Variable(covar.float().cuda())
print("================== Without Masking with Cholesky=============")
pre_pts = torch.cat((b, covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt, is_covariance= False)
print(correct_laplacian(loss_gau.data, form))

loss_mae = correct_l2loss(mae_loss(b, gt))
print(loss_mae.data)

# ==============================================================================
# Input zero and output should be zero
# ==============================================================================
b = torch.from_numpy(np.array([[[0,0], [0,0],  [0,0]],[ [0,0], [0,0], [0,0]]]))
mask = torch.from_numpy(np.array([[1, 0, 1], [0,1,1.]]))

# These are the lower triangular matrices of the covariance matrix
covar = torch.from_numpy(np.array([[[1,0,1], [1,0,1],  [1,0,1]],[ [1,0,1], [1,0,1], [1,0,1.]]]))
covar = scale_covariance(covar, form, is_covariance= False)

b = Variable(b.float().cuda())
mask = Variable(mask.float().cuda())
covar = Variable(covar.float().cuda())
gt = b.clone() 

print("================== Without Masking with Cholesky =============")
pre_pts = torch.cat((b, covar),2)

loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt, is_covariance= False)
print(correct_laplacian(loss_gau.data, form))

loss_mae = correct_l2loss(mae_loss(b, gt))
print(loss_mae.data)
