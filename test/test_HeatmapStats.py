
"""
    Tests the heatmaps statistics. The groundtruth is generated  using the numpy
    average and covariance term.

    Sample run:
    python test/test_covariance.py

    Verison 2 Abhinav Kumar 2019-06-06 Groundtruth using numpy. Can test random heatmaps now
    Version 1 Abhinav Kumar 2019-06-05
"""


import sys
import numpy as np
import torch
from torch.autograd import Variable

import sys
sys.path.insert(0, './pylib')
from HeatmapStats import get_spatial_mean_and_covariance

EPS = 1e-6

def get_diff(x,y, prefix):
    """
        Calculates the difference of two entities where y is taken as the
        ground truth. Raises an error if it finds significant deviation
        between the predicted and the ground truth
    """
    diff = np.linalg.norm(x.flatten()-y.flatten(),1)
    diff/= np.linalg.norm(y.flatten(),1)

    if diff <= EPS:
        print(prefix + " passed")
    else:
        print(prefix + " failed !!!")
        print("diff= {}, EPS= {}".format(diff, EPS))
        sys.exit()

def generate_grid_np(h, w):
    """
        Generates an equally spaced grid with coordinates as integers with the
        size same as the input heatmap in numpy
    """
    x = np.linspace(0, w - 1, num = w)
    y = np.linspace(0, h - 1, num = h)
    yv = np.tile(y[:, np.newaxis], (1, w))
    xv = np.tile(x  , (h, 1))

    return xv, yv

def generate_grid(h, w):
    """
        Generates an equally spaced grid with coordinates as integers with the
        size same as the input heatmap
    """
    x = torch.linspace(0, w - 1, steps =w)
    y = torch.linspace(0, h - 1, steps = h)
    yv = y.view(-1, 1).repeat(1, w)
    xv = x.repeat(h, 1)

    return xv, yv

def test_cov_calculation(x):
    batch_size   = x.shape[0]
    num_points   = x.shape[1]
    h            = x.shape[2] 
    w            = x.shape[3]

    xvar         = Variable(x)
    means,cov,_  = get_spatial_mean_and_covariance(xvar, use_softmax=False, postprocess="")
    means        = means.data.numpy()
    cov          = cov.data.numpy()

    for i in range(batch_size):
        for j in range(num_points):
            means_p = means[i,j]
            cov_p   = cov  [i,j]

            # Ground truth from numpy
            xv, yv = generate_grid_np(h,w)
            xv  = xv.flatten()[:,np.newaxis].T
            yv  = yv.flatten()[:,np.newaxis].T
            arr = np.concatenate((xv, yv), axis=0)
            wt  = x[i,j].numpy().flatten()

            means_gt = np.average(arr, weights=wt, axis=1)
            cov_gt   = np.cov    (arr, aweights=wt)

            print("")
            #print("Input Variable")
            #print(xvar.data[i,j])
            print("Means and covariance")
            #print(means_p)
            print(cov_p)
            print("Numpy means and covariance")
            #print(means_gt)
            print(cov_gt)

            get_diff(means_p, means_gt, "Mean")
            get_diff(cov_p  , cov_gt  , "Covariance")

"""
x          = torch.ones((1,1,4,4))
test_cov_calculation(x)

x[0,0]     = torch.from_numpy(np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]))
test_cov_calculation(x)

x = torch.ones((1, 1, 3, 3))
test_cov_calculation(x)

x[0, 0]   = torch.from_numpy(np.array([[1,0,1],[1,0,1],[1,0,1]]))
test_cov_calculation(x)
"""
torch.manual_seed(1739)
# Generate some heatmaps
x = torch.ones((2,3,64, 64)).uniform_(0, 1)
test_cov_calculation(x)
