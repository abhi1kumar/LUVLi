

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import sys
sys.path.insert(0, './pylib')
from HeatmapStats import get_spatial_mean_and_covariance

from gaussian_loss import *

#optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr, alpha=0.99,
#                                   eps=1e-8, momentum=0, weight_decay=0)

loss = 0
mse_loss         = nn.MSELoss()
loss_fn          = FaceAlignLoss()

pts_var = torch.from_numpy(1.5*np.ones((1,1,2))).float()
pts_var = Variable(pts_var)

a = torch.from_numpy(np.ones((1,1,5,5))).float()
input = Variable(a)

pred_pts_new, _ = get_spatial_mean_and_covariance(input)
#covar = covar.view(covar.shape[0], covar.shape[1], 4)

b = torch.zeros((1,1,4))
b[0,0,0] = 1
b[0,0,3] = 1
covar = Variable(b)

# Concat the calculations for each image and each landmark.
print(pred_pts_new)
print(covar)
pre_pts = torch.cat((pred_pts_new, covar),2)
loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages = loss_fn([pre_pts], pts_var, is_covariance=True)

loss        += loss_gau

# gradient and do SGD step
#optimizer.zero_grad()
loss.backward()
#optimizer.step()
