

"""
	Sample Run
    python plot/plot_pearson_on_trace_eccentricities.py -i run_940_evaluate

	Plots the scatter plot by replacing mean and covariance with a normal variable
	sampled from the mean and covariance.

	Version 1 2019-10-03 Abhinav Kumar
"""


import os, sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import math
import matplotlib
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CommonPlottingOperations import *
from pylib.Cholesky import *

means_rel            = "means.npy"
cholesky_rel         = "cholesky.npy"
width_height_box_rel = "width_height_box.npy"
ground_rel           = "ground_truth.npy"
nme_file             = "nme_new_box_per_image_per_landmark.npy"

dodge_blue = np.array([30, 144, 255])/255.
dpi   = 200
alpha = 1
size  = 10
fs    = 18
matplotlib.rcParams.update({'font.size': fs})
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
power = 4

color1 = (1,0.45,0.45)
color2 = dodge_blue
#===============================================================================
# Argument Parsing
#===============================================================================
ap     = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id', default= 'run_940_evaluate', help= 'input folder relative path')
args   = ap.parse_args()

#===============================================================================
#==== Our folder entries =============
#===============================================================================
folder = os.path.join("abhinav_model_dir", os.path.join(args.exp_id, "300W_test"))
print("Folder= {}".format(folder))

means            = np.load(os.path.join(folder, means_rel))
L_vect           = np.load(os.path.join(folder, cholesky_rel))
width_height_box = np.load(os.path.join(folder, width_height_box_rel))
ground           = np.load(os.path.join(folder, ground_rel))
nme_norm         = np.load(os.path.join(folder, nme_file))

print(width_height_box.shape)
d = np.sqrt(width_height_box[:,0] * width_height_box[:,1]) 
#d = np.linalg.norm(ground[:, 36] - ground[:, 45], axis = 1)
covar, det_sigma = cholesky_to_covar(L_vect)


nme_norm = get_err(means, ground)
nme_norm = nme_norm.reshape((600, 68))
print("NME = {:.2f}".format(100.0*np.mean(nme_norm)))

# Get the average of the means and covariances
_, means_norm = normalize_input(means, d)
_, covar_norm = normalize_input(covar, d*d)
#covar_norm = covar

det_sigma_norm = np.sqrt(np.sqrt(covar_norm[:,:,0,0]*covar_norm[:,:,1,1] - covar_norm[:,:,0,1]*covar_norm[:,:,1,0]))

nme_image    = np.mean(nme_norm, 1)
det_sigma_norm_image = np.mean(det_sigma_norm, 1)

nme_landmark     = nme_norm.flatten()
det_sigma_norm_landmark = det_sigma_norm.flatten()


num_points = nme_landmark.shape[0]
index = np.random.choice(range(num_points), int(0.025 * num_points), replace=False)


def plot_scatter_with_corr(x, y, figsize= (9.6, 6), xlabel= None, ylabel= None, xlim= None, ylim= None, path= ""):
	pearson_image = pearsonr(x, y)
	print("Pearson = {}".format(pearson_image[0]))

	# Scatter plot of all points with respect to their errors
	plt.figure(figsize= figsize, dpi= dpi)
	ax = plt.gca()
	p2 = plt.scatter(x, y, c= dodge_blue, edgecolors= 'none', s= size, alpha= alpha)

	plt.grid(True)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	if xlim is not None:
		plt.xlim(xlim)
	if ylim is not None:
		plt.ylim(ylim)

	#plt.legend((p1, p2), ('KDN-Gaussian (Chen et al), Correlation= ' + str(round(pearson_image_lisha[0], 2)), 'UGLLI (Ours), Correlation= ' + str(round(pearson_image[0], 2))), scatterpoints= 1, loc='upper right', fontsize= fs)
	plt.legend((p2,), ('UGLLI (Ours), Correlation= ' + str(round(pearson_image[0], 2)),), scatterpoints= 1, loc='upper right', fontsize= fs)

	print("Saving to {}".format(path))
	plt.savefig(path)
	print("")
	plt.close()

#
trace_norm = np.trace(covar_norm, axis1=2, axis2=3) 

figsize = (12, 6)
path = "images/scatter_trace_vs_nme_image.png"
plot_scatter_with_corr(np.mean(nme_norm, 1), np.mean(trace_norm, axis=1), figsize= figsize, xlabel= r"$NME_{box}$ for each image"        , ylabel= r"Mean Normalized Trace for each image", xlim= None, ylim= None, path= path)
path = "images/scatter_trace_vs_nme_landmark.png"
plot_scatter_with_corr(nme_norm.flatten() , trace_norm.flatten()        , figsize= figsize, xlabel= r"Normalized Error for each landmark", ylabel= r"Normalized Trace for each landmark"  , xlim= None, ylim= None, path= path)

# Calculate the eigen values
eig_norm_cov = np.zeros((nme_norm.shape[0], nme_norm.shape[1], 2))
for i in range(nme_norm.shape[0]):
	for j in range(nme_norm.shape[1]):
		w, v = np.linalg.eig(covar_norm[i,j])
		eig_norm_cov[i,j] = np.sort(w) 

"""
# Condition number is larger eigen value by smaller eigen value
# Eccentricity = sqrt(1- smaller eigen value/ larger eigen value)
condition_number_norm   = np.multiply(eig_norm_cov[:,:,1], 1.0/(eig_norm_cov[:,:,0] + 0.0001))
figsize = (12, 6)
path = "images/scatter_kappa_vs_nme_image.png"
plot_scatter_with_corr(np.mean(nme_norm, 1), np.mean(condition_number_norm, axis=1), figsize= figsize, xlabel= r"$NME_{box}$ for each image"        , ylabel= r"Mean Normalized $\kappa$ for each image", xlim= None, ylim= None, path= path)
path = "images/scatter_kappa_vs_nme_landmark.png"
plot_scatter_with_corr(nme_norm.flatten() , condition_number_norm.flatten()        , figsize= figsize, xlabel= r"Normalized Error for each landmark", ylabel= r"Normalized $\kappa$ for each landmark"  , xlim= None, ylim= None, path= path)


ecc_norm = (1 - np.multiply(eig_norm_cov[:,:,0], 1.0/eig_norm_cov[:,:,1]))
figsize = (12, 6)
path = "images/scatter_ecc_vs_nme_image.png"
plot_scatter_with_corr(np.mean(nme_norm, 1), np.mean(ecc_norm, axis=1), figsize= figsize, xlabel= r"$NME_{box}$ for each image"        , ylabel= r"Mean Normalized Ecc for each image", xlim= None, ylim= None, path= path)
path = "images/scatter_ecc_vs_nme_landmark.png"
plot_scatter_with_corr(nme_norm.flatten() , ecc_norm.flatten()        , figsize= figsize, xlabel= r"Normalized Error for each landmark", ylabel= r"Normalized Ecc for each landmark"  , xlim= None, ylim= None, path= path)
"""

path = "images/3d_scatter_eigen_vs_nme_landmark.png"

fig = plt.figure(figsize= figsize, dpi= dpi)
ax = Axes3D(fig)
ax.scatter(nme_norm.flatten(), eig_norm_cov[:,:,1].flatten(), eig_norm_cov[:,:,0].flatten(), c= dodge_blue, edgecolors= dodge_blue, s= size, alpha= alpha)
ax.set_xlabel("\n" + r"Normalized Error" +"\n" + r"for each landmark")
ax.set_ylabel("\n\n" + r"Highest eigen value")
ax.set_zlabel("\n\n" + r"Lowest eigen value")
print("Saving to {}".format(path))
plt.savefig(path)
plt.close()

path = "images/3d_scatter_eigen_vs_nme_image.png"

fig = plt.figure(figsize= figsize, dpi= dpi)
ax = Axes3D(fig)
ax.scatter(np.mean(nme_norm, axis= 1), np.mean(eig_norm_cov[:,:,1], axis= 1), np.mean(eig_norm_cov[:,:,0], axis= 1), c= dodge_blue, edgecolors= dodge_blue, s= size, alpha= alpha)
ax.set_xlabel("\n" +r"$NME_{box}$ for each image" )
ax.set_ylabel("\n\n" + r"Highest eigen value")
ax.set_zlabel("\n\n" + r"Lowest eigen value")
print("Saving to {}".format(path))
plt.savefig(path)
plt.close()