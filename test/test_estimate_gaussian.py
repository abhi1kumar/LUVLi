

"""
    Checking the fitting formulation of the 2D Gaussian.

    
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

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
#np.set_printoptions(threshold=sys.maxsize)

def estimate_param(prob, means):
    # Use true means for shifting the grid
    x = np.arange(xmin, xmax, delta) - means[0]
    y = np.arange(xmin, xmax, delta) - means[1]
        
    X, Y = np.meshgrid(x, y)

    # reshape X and Y
    num_pts = x.shape[0]
    X = X.reshape((num_pts**2, ))
    Y = Y.reshape((num_pts**2, ))
    Z = prob.reshape((num_pts**2, ))

    data = np.zeros((num_pts**2, 3))
    data[:, 0] = X**2
    data[:, 1] = Y**2
    data[:, 2] = 2*X*Y

    coeff = np.linalg.lstsq(data, -2 * np.log(Z + 10e-800), rcond=None)[0]

    alpha = coeff[0]
    beta  = coeff[1]
    gamma = coeff[2]

    print(coeff)

    covar = np.zeros((2,2))
    denom = alpha*beta - (gamma**2)
    covar[0, 0] = beta/ denom
    covar[0, 1] = - gamma/ denom
    covar[1, 0] = - gamma/ denom
    covar[1, 1] = (1/beta) + ((gamma**2)/(beta * denom))

    print(covar)    

def test_estimate_function(means_fixed, covariance, measure= "kld"):
    heatmaps = np.zeros((num_input, num_heatmaps_per_input, width, height))
    means    = np.zeros((num_input, num_heatmaps_per_input, 2))
    covar    = np.zeros((num_input, num_heatmaps_per_input, 2, 2))

    # Get heatmaps first
    for i in range(num_input):
        for j in range(num_heatmaps_per_input):
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
            Z = multivariate_normal.pdf(np.concatenate((X.flatten()[:, np.newaxis], Y.flatten()[:, np.newaxis]), axis=1), mean= means_fixed, cov=covariance).reshape(X.shape[0], X.shape[1])   
            image_min = 0
            image_max = np.max(Z)
            #print("\nImage max")
            #print(image_max)

            heatmaps[i,j] = Z/np.sum(Z)
            means[i, j]   = means_fixed
            covar[i, j]   = covariance
            
            #print("-------------------------------------------------")
            #print("Ground truth mean and covariances")
            #print("-------------------------------------------------")
            #print(means_fixed)
            #print(covariance)

            #print("-------------------------------------------------")
            #print("Improved statistics")
            #print("-------------------------------------------------")
            means_cal, covar_cal, _ = get_spatial_mean_and_covariance(Variable(torch.from_numpy(heatmaps).float()), postprocess=None)
            #print(means_cal.data.numpy()*delta)
            #print(covar_cal.data.numpy()*(delta**2))
            """
            print("-------------------------------------------------")
            print("Estimated statistics")
            print("-------------------------------------------------")
            estimate_param(heatmaps[i, j], means_cal[i, j].data.numpy())
           
            t = heatmaps[i, j]
            vmax = np.max(t)
            vmin = np.min(t)
            plt.imshow(t, cmap= 'jet', vmin= vmin, vmax= vmax)
            plt.show()
            """
            return covar_cal[i, j, 0, 0], means_cal[i, j]

# Get a 2D numpy first 
delta = 1
xmin  = 0
xmax  = 64
x = np.arange(xmin, xmax, delta)
y = x
X, Y = np.meshgrid(x, y)

num_input = 1
num_heatmaps_per_input= 1
width = x.shape[0]
height= x.shape[0]

"""
means_fixed = np.array([32.1, 33.1])
covariance  = np.array([[0.09, 0],[0, 0.01]])
test_estimate_function(means_fixed, covariance)

max_iter = 1
for j in range(max_iter):
    means_fixed = np.array([32.25 + float(j)/max_iter, 32])
    covariance  = np.array([[0.98, -0.35],[-0.35, 1.32]])
    test_estimate_function(means_fixed, covariance)
"""

delta_1 = 0.001
x = np.arange(delta_1, 0.3, delta_1)
max_iter = x.shape[0]
input = np.zeros((max_iter, ))
output = np.zeros((max_iter, ))

max_x = 20
for i in range(max_x):
    print(i)
    means_fixed = np.array([32., 33.0]) + float(i)/max_x
    for j in range(max_iter):
        alpha = x[j]
        covariance  = np.array([[alpha, 0],[0, alpha]])
        input[j] = alpha
        output[j],_ = test_estimate_function(means_fixed, covariance)    
    plt.plot(input, output)

plt.plot(input, input, 'r')
plt.grid()
plt.show()


"""
covar = np.array([[0.1, 0], [0, 0.1]])
means_fixed = np.array([32.35, 33.45])
for j in range(max_iter):
    alpha = x[j]
    
    #print(covariance)
    input[j] = alpha
    output[j], _ = test_estimate_function(means_fixed, covariance)

plt.plot(input, output, input, input)
plt.grid()
"""
plt.show()
