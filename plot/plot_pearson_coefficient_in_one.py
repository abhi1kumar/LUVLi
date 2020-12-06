

"""
    Sample Run
    python plot/plot_pearson_coefficient_in_one.py -i run_50_evaluate run_51_evaluate --lisha
    python plot/plot_pearson_coefficient_in_one.py -i run_940_evaluate --lisha

    Plots Pearson coefficient in the form of scatter plot. 
    The first one is assumed to be Laplacian and therefore while converting from
    cholesky to covar, an option is passed as true.

    Version 2 2019-11-06 Abhinav Kumar (Support for multiple models, root as 2,4)
    Version 1 2019-07-29 Abhinav Kumar 
"""

import os, sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import math
import matplotlib
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
from CommonPlottingOperations import *
from pylib.Cholesky import *


means_rel            = "means.npy"
cholesky_rel         = "cholesky.npy"
ground_rel           = "ground_truth.npy"
nme_file             = "nme_new_box_per_image_per_landmark.npy"

color_pool = np.array([[1,0.45,0.45], [0.12, 0.56, 1.0], [1.0, 0.62, 0.0]])
label_pool = ["KDN [Chen 2018]", "LUVLi (Laplacian)", "LUVLi (Gaussian)"]
marker_pool= ["x", "o", "3"]

figsize= (9.6,6)
dpi   = 200
alpha = 1
size  = 20
fs    = 18
matplotlib.rcParams.update({'font.size': fs})
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

def plot_scatter_image_and_landmark(xdata_list, ydata_list, color_list, label_list, marker_pool_list):
    image_plot_x_list = []
    image_plot_y_list = []

    lmark_plot_x_list = []
    lmark_plot_y_list = []

    for i in range(len(xdata_list)):
        image_plot_x_list.append(np.mean(xdata_list[i], 1))
        image_plot_y_list.append(np.mean(ydata_list[i], 1))

        lmark_plot_x_list.append(xdata_list[i].flatten())
        lmark_plot_y_list.append(ydata_list[i].flatten())

    if args.root == 4:
        image_plot_ylabel = r'Mean $|\Sigma_{box}|^{1/4}$ for each image'
        lmark_plot_ylabel = r'$|\Sigma_{box}|^{1/4}$ for each landmark'
        plot_ylim   = np.array([0.0, 0.05])
    else:
        image_plot_ylabel = r'Mean $|\Sigma_{box}|^{1/2}$ for each image'
        lmark_plot_ylabel = r'$|\Sigma_{box}|^{1/2}$ for each landmark'
        plot_ylim   = np.array([0.0, 0.05])**2

    image_plot_xlabel = r'$NME_{box}$ for each image'
    lmark_plot_xlabel = r'Normalized Error for each landmark'

    # First get wrt image
    plot_single_scatter_with_corr_for_multi_model(image_plot_x_list, image_plot_y_list, color_list, label_list, marker_pool_list, 
        xlim= [0.0, 0.22], ylim= plot_ylim, path= "images/scatter_sigma_vs_nme_image.png", 
        xlabel= image_plot_xlabel, ylabel= image_plot_ylabel, frac= 1, 
        figsize= figsize, dpi= dpi, size= size, alpha= alpha)
    
    plot_single_scatter_with_corr_for_multi_model(lmark_plot_x_list, lmark_plot_y_list, color_list, label_list, marker_pool_list, 
        xlim= [0.0, 0.30], ylim= plot_ylim, path= "images/scatter_sigma_vs_nme_lmark.png", 
        xlabel= lmark_plot_xlabel, ylabel= lmark_plot_ylabel, frac= 0.025, 
        figsize= figsize, dpi= dpi, size= size, alpha= alpha)

#===============================================================================
# Argument Parsing
#===============================================================================
ap     = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id', default= 'run_940_evaluate', help= 'input folder relative path', nargs='*')
ap.add_argument('-r', '--root'  , default= 4                 , help= 'which root to take - 2/4. default: 4')
ap.add_argument(      '--lisha' , action='store_true'        , help= 'use lisha neurips folder in plotting')  
args   = ap.parse_args()

# These lists keep the data and nme
nme_norm_list       = []
det_sigma_norm_list = []
color_list          = []
label_list          = []
marker_pool_list    = []

# Check if lisha's values are to be plotted
if args.lisha:
    #===============================================================================
    #====== Lisha entries ======
    #===============================================================================
    nips_folder  = 'plot_uncertainty_NeurlPS'
    lisha_folder = os.path.join(nips_folder, 'result_new')
    print("Folder= {}".format(lisha_folder))

    ground = np.load(os.path.join(lisha_folder, 'twtest64_gt.npy'))
    means = np.load(os.path.join(lisha_folder, 'kdng.npy'))
    # Lisha gave us the uncertainty which is normalized square root of determinant of 
    # covariance matrix 
    det = np.load(os.path.join(lisha_folder, 'uncertainty_kdng.npy'))
    det = det.reshape((600, 68))

    d = compute_scale(ground)

    nme_norm = get_err(means, ground)
    nme_norm_lisha = nme_norm.reshape((600, 68))

    # Get the average of the means and covariances
    _, means_norm = normalize_input(means, d)

    if args.root == 4:
        # Lisha gave us the uncertainty which is normalized square root of determinant of 
        # covariance matrix and therefore we are not taking fourth square root
        # We only take the second square root and normalize once
        det_sigma_norm_lisha = np.sqrt(det)/d[:, np.newaxis]
    else:
        det_sigma_norm_lisha = det

    nme_norm_list.append(nme_norm_lisha)
    det_sigma_norm_list.append(det_sigma_norm_lisha)
    color_list.append(color_pool[0])
    label_list.append(label_pool[0])
    marker_pool_list.append(marker_pool[0])


# Add our folders into the list as well
for i in range(len(args.exp_id)):
    #===============================================================================
    #==== Our folder entries =============
    #===============================================================================
    folder = os.path.join("abhinav_model_dir", os.path.join(args.exp_id[i], "300W_test"))
    print("Folder= {}".format(folder))

    means            = np.load(os.path.join(folder, means_rel))
    L_vect           = np.load(os.path.join(folder, cholesky_rel))
    ground           = np.load(os.path.join(folder, ground_rel))
    nme_norm         = np.load(os.path.join(folder, nme_file))

    d = compute_scale(ground)
    
    if args.exp_id[i] == "run_940_evaluate" or i > 0:
        covar, det_sigma = cholesky_to_covar(L_vect)
    else:
        covar, det_sigma = cholesky_to_covar(L_vect, True)

    # Get the normalized means and covariances
    _, means_norm = normalize_input(means, d)
    _, covar_norm = normalize_input(covar, d*d)

    if args.root == 4:
        det_sigma_norm = np.sqrt(np.sqrt(covar_norm[:,:,0,0]*covar_norm[:,:,1,1] - covar_norm[:,:,0,1]*covar_norm[:,:,1,0]))
    else:
        det_sigma_norm = np.sqrt(covar_norm[:,:,0,0]*covar_norm[:,:,1,1] - covar_norm[:,:,0,1]*covar_norm[:,:,1,0])

    nme_norm_list.append(nme_norm)
    det_sigma_norm_list.append(det_sigma_norm)
    color_list.append(color_pool[i+1])

    # Special case for run_940_evaluate
    if args.exp_id[i] == "run_940_evaluate":
        label_list.append("UGLLI (Ours)")
    else:
        label_list.append(label_pool[i+1])
    
    marker_pool_list.append(marker_pool[i+1])

print("\nTotal number of models to plot = {}".format(len(nme_norm_list)))
print(label_list)
print("")

# Now plot
plot_scatter_image_and_landmark(nme_norm_list, det_sigma_norm_list, color_list, label_list, marker_pool_list)
