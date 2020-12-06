

"""
    Sample Run
    python plot/show_multi_pie_landmark_uncertainties.py -i run_493_evaluate
    python plot/show_multi_pie_landmark_uncertainties.py -i run_109_evaluate --laplacian

    Plots landmark uncertainities on the normalized faces of multi PIE dataset.

    Version 2 2019-11-08 Abhinav Kumar (Support for Laplacian added)
    Version 1 2019-06-24 Abhinav Kumar 
"""

import os, sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import math

import plotting_params as params
from matplotlib import pyplot as plt
from matplotlib.mlab import bivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.image as mpimg

from pylib.Cholesky import *
from CommonPlottingOperations import *

DPI          = 150
ms           = 5
xmin         = -.05
xmax         = 2
h            = 16
std_dev      = 2
num_examples = 5
# Break points are points where the landmark points are not joined to the next 
# landmark point
break_points = np.array([17, 22, 27, 31, 36, 42, 48, 60, 68]) - 1 # -1 since data is 1 indexed and python is 0 indexed.

#===============================================================================
# Argument Parsing
#===============================================================================
ap     = argparse.ArgumentParser()
ap.add_argument('-i', '--input'    , default= 'run_493_evaluate', help= 'input folder relative path')
ap.add_argument(      '--laplacian', action = 'store_true'      , help= 'use laplacian likelihood instead of Gaussian while training')
args   = ap.parse_args()

folder = os.path.join("abhinav_model_dir", os.path.join(args.input, "multi_pie"))
print("Folder= {}".format(folder))

ground, means, L_vect= load_all(folder_path= folder, load_heatmaps= False, load_cholesky= True, load_vis= False, load_images= False)

print(means.shape)
print(L_vect.shape)
num_points = means.shape[1]
covar, _ = cholesky_to_covar(L_vect, args.laplacian)

# Get the average of the means and covariances
print("Normalizing by inter-ocular distances as in \n\t300 Faces In-The-Wild Challenge:  Database and Results,\n\tSagonas et al, Image and Vision Computing, 2016")
d = np.linalg.norm(ground[:, 36] - ground[:, 45], axis = 1)
mean_of_means, means_norm = normalize_input(means, d)
mean_of_covariance, covar_norm = normalize_input(covar, d*d, use_method= "log-exponential")

print(mean_of_means.shape)
print(mean_of_covariance.shape)
print("Normalization done")


# Change of coordinates from 
#   -----> X    ^ Y
#   |           |
#   |        to |
#   |           |----> X
#   vX
# Convention of x and y for MultiPIE is opposite to matplotlib plot
mean_of_means[:,1] = -mean_of_means[:,1]
mean_of_covariance[:, 0, 1] = - mean_of_covariance[:, 0, 1]
mean_of_covariance[:, 1, 0] = - mean_of_covariance[:, 1, 0]

means_norm[:, :, 1]    = -means_norm[:, :, 1]
covar_norm[:, :, 0, 1] = -covar_norm[:, :, 0, 1]
covar_norm[:, :, 1, 0] = -covar_norm[:, :, 1, 0]

#===============================================================================
# Start plotting 300W journal paper uncertainty ellipses and our uncertainty
# ellipses
#===============================================================================
# Using values from a colormap
# Reference https://stackoverflow.com/a/8931396
jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin= 1e-6, vmax= 1e-3)
scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=jet)

print("Plotting 300W journal paper uncertainty ellipses and our uncertainty ellipses")
fig = plt.figure(figsize=(h, 8), dpi= DPI)

ax = plt.subplot(121)
img = mpimg.imread('images/multi_pie_paper.png')
thresh = 0.1
#gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
#ax.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

# Convert all other colors except black and white to dodgeblue
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        pixel = img[i,j]
        if pixel[0] < thresh and pixel[1] < thresh and pixel[2] < thresh:
            img[i, j] = np.zeros(img[i,j].shape)
        elif pixel[0] > 1-thresh and pixel[1] > 1-thresh and pixel[2] > 1-thresh:
            pass
        else:
            img[i, j] = params.dodge_blue
ax.imshow(img)
plt.axis("off")

ax  = fig.add_subplot(122)

for i in range(num_points):
    mx  = mean_of_means[i,0]
    my  = mean_of_means[i,1]
    cov = mean_of_covariance[i]
    plot_mean_and_ellipse(cov, mx, my, n= 2, nstd= std_dev, ax= ax, color1= params.covar_color, color2= params.covar_color, linewidth= params.lw, markersize= ms)#, scalar_map= scalar_map)

    if i not in break_points:
        plt.plot(mean_of_means[i:i+2, 0], mean_of_means[i:i+2, 1], 'k--')
    else:
        if i == 41:
           plt.plot(mean_of_means[[36,i], 0], mean_of_means[[36,i], 1], 'k--')
        elif i == 47:
           plt.plot(mean_of_means[[42,i], 0], mean_of_means[[42,i], 1], 'k--')
        elif i == 59:
           plt.plot(mean_of_means[[48,i], 0], mean_of_means[[48,i], 1], 'k--') 

plt.axis("off")
plt.xlim([xmin , xmax])
plt.ylim([-xmax, -xmin])
plt.tight_layout()
path = "images/multi_pie_uncertainties.png"
savefig(plt, path)
plt.close()

#===============================================================================
# Plot mean uncertainty ellipses with the samples of uncertainty ellipses of
# the examples.
#===============================================================================
print("Plotting mean uncertainty ellipses with the samples of uncertainty ellipses of the examples")
example_indices = np.linspace(0, means.shape[0]-1, num_examples).astype(int)
colors = cmx.rainbow(np.linspace(0, 1, num_examples))

fig = plt.figure(figsize=(h/2, 8), dpi= DPI)
ax  = fig.add_subplot(111)
for i in range(36):
    mx  = mean_of_means[i,0]
    my  = mean_of_means[i,1]
    cov = mean_of_covariance[i]
    # Plot mean ellipses first
    plot_mean_and_ellipse    (cov, mx, my, n= 2, nstd= std_dev, ax= ax, color1= params.covar_color, color2= params.covar_color, linewidth= params.lw, markersize= 5)#, scalar_map= scalar_map)

    # Plot sample ellipses next
    for j in range(num_examples):
        mx  = means_norm[example_indices[j], i, 0]
        my  = means_norm[example_indices[j], i, 1]
        cov = covar_norm[example_indices[j], i]
        plot_mean_and_ellipse(cov, mx, my, n= 2, nstd= std_dev, ax= ax, color1= colors[j]         , color2= colors[j]         , linewidth= 0.25     , markersize= 1)

    if i not in break_points:
        plt.plot(mean_of_means[i:i+2, 0], mean_of_means[i:i+2, 1], 'k--')
    else:
        if i == 41:
           plt.plot(mean_of_means[[36,i], 0], mean_of_means[[36,i], 1], 'k--')
        elif i == 47:
           plt.plot(mean_of_means[[42,i], 0], mean_of_means[[42,i], 1], 'k--')
        elif i == 59:
           plt.plot(mean_of_means[[48,i], 0], mean_of_means[[48,i], 1], 'k--') 

plt.axis("off")
plt.xlim([xmin , xmax])
plt.ylim([-xmax, -xmin])
plt.tight_layout()
path = "images/multi_pie_mean_and_sample_uncertainties_ellipses.png"
savefig(plt, path)
plt.close()
