

"""
    Sample Run
    python test/test_GaussianRegularizationLoss.py

    Version 1 Abhinav Kumar
"""

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

def test_function(means_fixed, covariance, measure= "kld"):
    heatmaps = np.zeros((num_input, num_heatmaps_per_input, width, height))
    means    = np.zeros((num_input, num_heatmaps_per_input, 2))
    covar    = np.zeros((num_input, num_heatmaps_per_input, 2, 2))

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

    print("\n\nSum prior to normalization")
    print(np.sum(heatmaps))

    # Pass Variables
    heatmaps = Variable(torch.from_numpy(heatmaps).float())
    covar    = Variable(torch.from_numpy(covar).float())
    means    = Variable(torch.from_numpy(means).float())
    covar    = covar.view(covar.shape[0], covar.shape[1], 4)

    sum_heatmaps = expand_two_dimensions_at_end(torch.sum(torch.sum(heatmaps, dim= 3), dim= 2), height, width)
    normalized_heatmaps = heatmaps/ sum_heatmaps

    print("Sum after normalization")
    print(torch.sum(normalized_heatmaps).data.numpy()[0])

    # Get older heatmap statistics
    means_cal_1, covar_cal, _ = get_spatial_mean_and_covariance(normalized_heatmaps, use_softmax=False, tau=0.0003, postprocess=None)
    print("-------------------------------------------------")
    print("Ground truth mean and covariances")
    print("-------------------------------------------------")
    print(means_fixed)
    print(covariance)
    print("-------------------------------------------------")
    print("Older heatmap statistics")
    print("-------------------------------------------------")
    print(means_cal_1.data.numpy())
    print(covar_cal.data.numpy())


    # Get improved heatmap statistics
    print("-------------------------------------------------")
    print("Improved heatmap statistics")
    print("-------------------------------------------------")
    means_cal, covar_cal, _ = get_spatial_mean_and_covariance_improved(normalized_heatmaps, use_softmax=False, tau=0.0003, postprocess=None)
    print(means_cal.data.numpy())
    print(covar_cal.data.numpy())

    """
    t = normalized_heatmaps[0,0].clone().data.numpy()
    vmax = np.max(t)
    print(vmax)
    vmin = np.min(t)
    plt.subplot(122)
    plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
    t = normalized_heatmaps[0, 0].clone().data.numpy()
    vmax = np.max(t)
    vmin = np.min(t)
    plt.subplot(121)
    plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
    plt.show()
    """

    gauss_regln_loss = GaussianRegularizationLoss(measure= measure)
    loss_gauss_regln = gauss_regln_loss(normalized_heatmaps, means_cal, covar_cal, display= True, means_other= means_cal_1)
    print("{} distance between original and the fitted gaussian (should be very small)= {}".format(measure, loss_gauss_regln.data.numpy()[0]))


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

"""
means_fixed = np.array([2.5, 32.5])
covariance  = np.array([[58.2933, -4.6795],[-4.6795, 18.9155]])
test_function(means_fixed, covariance)

means_fixed = np.array([2.5, 32.5])
covariance  = np.array([[1000, -500],[500, 1000]])
test_function(means_fixed, covariance)

max_l = 5
for i in range(max_l):
    means_fixed = np.array([2.9970e+00 + i * int(64/max_l), 37.962 + i * int(32/max_l)])
    covariance  = np.array([[2.6545e+02, 4.9269e+01],  [4.9269e+01,6.7774e+01]])
    test_function(means_fixed, covariance)

max_l = 5
for i in range(max_l):
    means_fixed = np.array([2.9970e+00 + i * int(64/max_l), 37.962 + i * int(32/max_l)])
    covariance  = np.array([[25.45, 4.9269],  [4.9269,6.77]])
    test_function(means_fixed, covariance)
"""
measures = ["kld", "l1", "l2"]

for i in range(len(measures)):
    means_fixed = np.array([32.5, 32.5])
    covariance  = np.array([[2.60, 0],[0, 0.09]])
    test_function(means_fixed, covariance, measure= measures[i])
