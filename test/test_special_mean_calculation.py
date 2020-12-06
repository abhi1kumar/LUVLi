

"""
    Checking the mean for floating point number calculation to check which
    one is best.

    
"""

import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import sys
sys.path.append(os.getcwd())
from pylib.HeatmapStats import *

def get_error(means_estimated, means_ground):
    error = 0.0    
    for i in range(num_input):
        for j in range(num_heatmaps_per_input):
            error += np.linalg.norm(means_ground[i, j] - means_estimated[i, j])

    return error/ (num_input*num_heatmaps_per_input)

def test_estimate_function(means_fixed, covariance, measure= "kld"):
    heatmaps = np.zeros((num_input, num_heatmaps_per_input, width, height))
    means    = np.zeros((num_input, num_heatmaps_per_input, 2))
    covar    = np.zeros((num_input, num_heatmaps_per_input, 2, 2))

    # Get heatmaps first
    for i in range(num_input):
        print(i)
        for j in range(num_heatmaps_per_input):
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
            Z = multivariate_normal.pdf(np.concatenate((X.flatten()[:, np.newaxis], Y.flatten()[:, np.newaxis]), axis=1), mean= means_fixed[i, j], cov= covariance[i, j]).reshape(X.shape[0], X.shape[1]) 

            heatmaps[i,j] = Z/np.sum(Z)
            means[i, j]   = means_fixed[i, j]
            covar[i, j]   = covariance[i, j]


    means_cal_1, covar_cal_1, _ = get_spatial_mean_and_covariance_improved(Variable(torch.from_numpy(heatmaps).float()), use_softmax=False, tau=0.0003, postprocess=None, special_mean= True)
    means_cal_1 = means_cal_1.data.numpy()
    print("Error due to special mean= {}".format(get_error(means_cal_1, means)))

    means_cal_2, covar_cal_2, _ = get_spatial_mean_and_covariance_improved(Variable(torch.from_numpy(heatmaps).float()), use_softmax=False, tau=0.0003, postprocess=None, special_mean= False)
    means_cal_2 = means_cal_2.data.numpy()
    print("Error when used normally = {}".format(get_error(means_cal_2, means)))

# Get a 2D numpy first 
delta = 1
xmin  = 0
xmax  = 64
x = np.arange(xmin, xmax, delta)
y = x
X, Y = np.meshgrid(x, y)

max_iter = 10
num_input = max_iter
num_heatmaps_per_input = max_iter
width = x.shape[0]
height= x.shape[0]

heatmaps    = np.zeros((num_input, num_heatmaps_per_input, width, height))
means_fixed = np.zeros((num_input, num_heatmaps_per_input, 2))
covariance  = np.zeros((num_input, num_heatmaps_per_input, 2, 2))

for i in range(num_input):
    delta_x = float(i)/max_iter
    for j in range(num_heatmaps_per_input):
        delta_y = float(j)/max_iter
        means_fixed[i, j] = np.array([32 + delta_x, 32 + delta_y])
        covariance[i, j]  = np.array([[5, 0],[0, 5]])

test_estimate_function(means_fixed, covariance)

# ==============================================================================
# Conclusion- do not use special mean
# ==============================================================================
