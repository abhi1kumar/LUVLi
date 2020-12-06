


import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import sys
sys.path.append(os.getcwd())

from GaussianRegularizationLoss import *
from pylib.CommonOperations import *
from pylib.HeatmapStats import *

def flip180(input):

    return np.flipud(np.fliplr(input))

def get_means_covar(heatmaps_new, is_var= False):
    if not is_var:
        heatmaps_new        = Variable(torch.from_numpy(heatmaps_new)).float()
    sum_heatmaps        = expand_two_dimensions_at_end(torch.sum(torch.sum(heatmaps_new, dim= 3), dim= 2), heatmaps_new.shape[2], heatmaps_new.shape[1])
    normalized_heatmaps = heatmaps_new/ sum_heatmaps
    means_cal, covar_cal, _ = get_spatial_mean_and_covariance(normalized_heatmaps, use_softmax=False, tau=0, postprocess=None)
    print(means_cal.data.numpy()-width)
    print(covar_cal.data.numpy())

# Get a 2D numpy first 
delta = 1
xmin  = 0
xmax  = 64
x = np.arange(xmin, xmax, delta)
y = x
X, Y = np.meshgrid(x, y)


num_input = 1
num_heatmaps_per_input= 1
width = xmax
height= xmax

heatmaps = np.zeros((num_input, num_heatmaps_per_input, width, height))
means    = np.zeros((num_input, num_heatmaps_per_input, 2))
covar    = np.zeros((num_input, num_heatmaps_per_input, 2, 2))

#means_fixed = np.array([12.5, 12.5])
#covariance = np.array([[58.2933, -4.6795],[-4.6795, 18.9155]])

means_fixed = np.array([62.5, 12.5])
covariance  = np.array([[5, 0],[0, 10]])
means_fixed = np.array([53.9460, 21.9780])
covariance = np.array([[1016.8052, -943.6020], [-943.6019, 985.1257]])

#means_fixed = np.array([43.9560, 11.9880])
#covariance  = np.array([[1077.9333, -1000.3964], [-1000.3964, 1006.2441]])
print("-------------------------------------------------")
print("Ground truth mean and covariances")
print("-------------------------------------------------")
print(means_fixed)
print(covariance)

for i in range(num_input):
    for j in range(num_heatmaps_per_input):

        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
        Z = multivariate_normal.pdf(np.concatenate((X.flatten()[:, np.newaxis], Y.flatten()[:, np.newaxis]), axis=1), mean= means_fixed, cov=covariance).reshape(X.shape[0], X.shape[1])        
        image_min = 0
        image_max = np.max(Z)
        #print("\nImage max")
        #print(image_max)

        heatmaps[i,j] = Z
        means[i, j]   = means_fixed
        covar[i, j]   = covariance

print("\nPrior to normalization")
print(np.sum(heatmaps))

sum_heatmaps        = np.sum(np.sum(heatmaps, 3), 2)
normalized_heatmaps = heatmaps/ sum_heatmaps

print("\nAfter normalization")
print(np.sum(normalized_heatmaps))

print("-------------------------------------------------")
print("Incorrect calculation")
print("-------------------------------------------------")
get_means_covar(heatmaps)
print("-------------------------------------------------")
print("Improved calculation")
print("-------------------------------------------------")
means_int = np.ceil(means).astype(int)
heatmaps_new = np.zeros((num_input, num_heatmaps_per_input, 3*width, 3*height))
print(heatmaps_new.shape)

"""
# Writing in numpy

for i in range(num_input):
    for j in range(num_heatmaps_per_input):
        heatmaps_new[i, j, width:2*width, height:2*height] = heatmaps[i,j]

        # Now add reflections about means
        means_r = means_int[i, j, 0]
        means_c = means_int[i, j, 1]

        quad1 = heatmaps[i, j, 0: means_r, means_c: width]
        quad2 = heatmaps[i, j, 0: means_r, 0: means_c]
        quad3 = heatmaps[i, j, means_r: height, 0: means_c]
        quad4 = heatmaps[i, j, means_r: height, means_c:width]

        heatmaps_new[i, j, 2*means_r     : height+means_r    , 2*means_c    : width+means_c]   = flip180(quad4)
        heatmaps_new[i, j, 2*means_r     : height+means_r    , width+means_c: width+2*means_c] = flip180(quad3)
        heatmaps_new[i, j, height+means_r: height + 2*means_r, width+means_c: width+2*means_c] = flip180(quad2)
        heatmaps_new[i, j, height+means_r: height + 2*means_r, 2*means_c    : width+means_c]   = flip180(quad1)


get_means_covar(heatmaps_new, is_var= False)

t = normalized_heatmaps[0, 0]
vmax = np.max(t)
vmin = np.min(t)    
plt.subplot(121)
plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)       
t = heatmaps_new[0,0]     
plt.subplot(122)
vmax = np.max(t)
vmin = np.min(t)
plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
plt.show()

"""
heatmaps     = Variable(torch.from_numpy(heatmaps)).float()
heatmaps_new = Variable(torch.from_numpy(heatmaps_new)).float()

means_cal, _, _ = get_spatial_mean_and_covariance(heatmaps, use_softmax=True, tau=0.000003, postprocess=None)
print(means_cal)

means_int = torch.ceil(means_cal.clone()).type(torch.LongTensor)
# DO NOT use .data anywehere otherwise it would disconnect the variable from 
# the graph and also doesnot copy as expected. So, means and covariance would
# be wrong
heatmaps_new[:, :, width:2*width, height:2*height] = heatmaps
means_row = means_int[:, :, 1].data
means_col = means_int[:, :, 0].data


for i in range(num_input):
    for j in range(num_heatmaps_per_input):
        means_r = means_row[i ,j]
        means_c = means_col[i, j]

        # Now add reflections about means of the different quadrants
        quad1 = heatmaps[i, j, 0: means_r, means_c: width]
        quad2 = heatmaps[i, j, 0: means_r, 0: means_c]
        quad3 = heatmaps[i, j, means_r: height, 0: means_c]
        quad4 = heatmaps[i, j, means_r: height, means_c:width]

        heatmaps_new[i, j, 2*means_r     : height+means_r    , 2*means_c    : width+means_c]   = flip180_tensor(quad4)
        heatmaps_new[i, j, 2*means_r     : height+means_r    , width+means_c: width+2*means_c] = flip180_tensor(quad3)
        heatmaps_new[i, j, height+means_r: height + 2*means_r, width+means_c: width+2*means_c] = flip180_tensor(quad2)
        heatmaps_new[i, j, height+means_r: height + 2*means_r, 2*means_c    : width+means_c]   = flip180_tensor(quad1)


get_means_covar(heatmaps_new, is_var= True)

t = normalized_heatmaps[0, 0]
vmax = np.max(t)
vmin = np.min(t)    
plt.subplot(131)
plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
plt.plot(means_int[0,0,0].data.numpy(), means_int[0,0,1].data.numpy(), marker= 'x', markersize= 8, color='green')
     
t = heatmaps_new[0,0].clone().data.numpy()
plt.subplot(132)
vmax = np.max(t)
vmin = np.min(t)
plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
plt.plot(means_int[0,0,0].data.numpy()+ height, means_int[0,0,1].data.numpy()+ width, marker= 'x', markersize= 8, color='green')

plt.subplot(133)
t = heatmaps_new[0,0, height:2*height, width:2*width].clone().data.numpy()
vmax = np.max(t)
vmin = np.min(t)
plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
plt.plot(means_int[0,0,0].data.numpy(), means_int[0,0,1].data.numpy(), marker= 'x', markersize= 8, color='green')
plt.show()
"""
means, covar, htp = get_spatial_mean_and_covariance_improved(heatmaps, use_softmax= True, tau= 1, postprocess= None)
print(means)
print(covar)

t = normalized_heatmaps[0, 0]
vmax = np.max(t)
vmin = np.min(t)    
plt.subplot(121)
plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)       
t = htp[0,0].clone().data.numpy()
plt.subplot(122)
vmax = np.max(t)
vmin = np.min(t)
plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
plt.show()
"""
