
import os, sys
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

from loss.gaussian_loss import *
import torch.nn as nn 

b = torch.from_numpy(np.array([[[12,13], [10,11],  [21,22]],[ [30,31], [41,42], [51,52.]]]))
mask = torch.from_numpy(np.array([[1, 0, 1],[0,1,1.]]))
covar = torch.from_numpy(np.array([[[1,0,0,1], [1,0,0,1],  [1,0,0,1]],[ [1,0,0,1], [1,0,0,1], [1,0,0,1.]]]))

b = Variable(b.float())
mask = Variable(mask.float())
covar = Variable(covar.float())
gt = b.clone() + 10.


loss_fn  = FaceAlignLoss()
mse_loss = nn.MSELoss()

print(b.data)
#print(mask.data)
#print(gt.data)
#print(covar.data)

print("================== Without Masking =============")
pre_pts = torch.cat((b, covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt, is_covariance= True)
print(loss_gau.data)

loss_mse = mse_loss(b, gt)
print(loss_mse.data)


print("================== With Masking =============")
print((b*mask.unsqueeze(-1)).data)
pre_pts = torch.cat((b*mask.unsqueeze(-1), covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt*mask.unsqueeze(-1), is_covariance= True)
print(loss_gau.data)

loss_mse = mse_loss(b*mask.unsqueeze(-1), gt*mask.unsqueeze(-1))
print(loss_mse.data)


# ==============================================================================
# Non-zero values and return correctly
# ==============================================================================
# These are the lower triangular matrices of the covariance matrix
covar = torch.from_numpy(np.array([[[1,0,1], [1,0,1],  [1,0,1]],[ [1,0,1], [1,0,1], [1,0,1.]]]))
covar = Variable(covar.float())
print("================== Without Masking with Cholesky=============")
pre_pts = torch.cat((b, covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt, is_covariance= False)
print(loss_gau.data)

loss_mse = mse_loss(b, gt)
print(loss_mse.data)

# ==============================================================================
# Non-zero values and return correctly
# ==============================================================================
b = torch.from_numpy(np.array([[[0,0], [0,0],  [0,0]],[ [0,0], [0,0], [0,0]]]))
b = Variable(b.float())
gt = b.clone()

# These are the lower triangular matrices of the covariance matrix
covar = torch.from_numpy(np.array([[[1,0,1], [1,0,1],  [1,0,1]],[ [1,0,1], [1,0,1], [1,0,1.]]]))
covar = Variable(covar.float())
print("================== Without Masking with Cholesky, zero values=============")
pre_pts = torch.cat((b, covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages= loss_fn([pre_pts], gt, is_covariance= False)
print(loss_gau.data)

loss_mse = mse_loss(b, gt)
print(loss_mse.data)
