

"""
	Sample Run
	python plot/plot_residual_covariance_vs_predicted_covariance.py -i run_62_evaluate

	Plots the scatter plot of the predicted covariance and the residual covariance.

	Version 1 2019-10-03 Abhinav Kumar
"""


import os, sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import math
import matplotlib
from scipy.stats import pearsonr

import plotting_params as params
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from CommonPlottingOperations import *
from pylib.Cholesky import *

means_rel            = "means.npy"
cholesky_rel         = "cholesky.npy"
width_height_box_rel = "width_height_box.npy"
ground_rel           = "ground_truth.npy"
nme_file             = "nme_new_box_per_image_per_landmark.npy"

BINS_FINAL_NUM = 50
SCALER         = 10000
FRAC_TO_KEEP   = 0.9

figsize = (8, 6)
msize   = 150
fs      = 32
matplotlib.rcParams.update({'font.size': fs})

def get_bins(x, num_bins):
	# sort the data
	x_min = np.min(x)
	x     = np.sort(x)

	pts_per_bin = int(np.ceil(x.shape[0]/(num_bins)))
	print("Num bins= {}, pts per bin  = {}".format(num_bins, pts_per_bin))
	# bins contain a lower bound. so 1 extra element
	bins  = np.zeros((num_bins+1, ))

	bins[0] = x_min
	for i in range(1,bins.shape[0]):
		if i*pts_per_bin < x.shape[0]:
			end_ind = i*pts_per_bin
		else:
			end_ind = x.shape[0]-1
		bins[i] = x[end_ind]

	return bins 

def throw_samples(x,y, frac_to_keep= FRAC_TO_KEEP):
	"""
		Throws the outliers based on sorting

		frac = 0.75 suggests keep the first three quarters of the bins and throw
		away the remaining bins. Keep the elements belonging to the valid bins
	"""
	print("Using {:.2f}% of the initial bins".format(frac_to_keep*100.0))
	samples_to_keep = int(x.shape[0] * frac_to_keep)

	# Sort the x array and get the indices 
	sorted_indices = np.abs(x).argsort()
	# Keep the indices which are required	
	keep_index = sorted_indices[0:samples_to_keep]
	
	# Use the same index for x and y
	x = x[keep_index]
	y = y[keep_index]

	return x, y

def plot_scatter_with_corr(x, y, figsize= params.size,  xlabel= None, ylabel= None, xlim= None, ylim= None, path= ""):
	"""
		Plots scatter plots
	"""
	cumulative = CUMULATIVE
	bins_final_num = BINS_FINAL_NUM

	# Throw the outliers	
	x,y = throw_samples(x,y)

	# Do the second binning based on sqrt after removing the outliers
	#_, bins = np.histogram(x, bins= 'sqrt')
	# Get the max bin and min bin and get logarithm bins 
	# (more on the smaller values and less on the larger)
	#bin_min = np.min(bins)
	#bin_max = np.max(bins)
	#bins_final = np.sort(np.logspace(np.log10(bin_min),np.log10(bin_max), bins_final_num))
	# Do the final logarithm binning on the data
	#print("Final number of bins in log scale = {}".format(bins_final.shape[0]-1))

	# Get bins on the data
	bins = get_bins(x, bins_final_num)
	input, _ = np.histogram(x, bins= bins)
	yplot = np.zeros(input.shape)
	xplot = np.zeros(input.shape)

	for j in range(0, bins.shape[0]-1):
		if cumulative:
			index = (x < bins[j+1])
		else:
			index = np.logical_and((x >= bins[j]), (x < bins[j+1]))
		xplot[j] = np.mean(x[index])
		yplot[j] = np.mean(y[index])
		
	fig = plt.figure(figsize= figsize, dpi= params.DPI)
	yplot_max = np.max(yplot)
	yplot_min = np.min(yplot)
	# Set the transparency in decreasing order for each bin
	# Smallest one should have alpha= 1, largest one should have alpha= 0.2
	#alpha_slope = float(0.8)/(yplot_max - yplot_min)
	#alpha = [-n*alpha_slope + 1 for n in yplot]
	rgba_colors = np.zeros((yplot.shape[0],4))
	rgba_colors[:,0:3] = params.covar_color_array
	rgba_colors[:,3]   = params.alpha

	# Set the marker size in increasing order for each bin
	# Smallest one should have small circle while larger one should have big circle
	#ms_slope = float(1400)/(yplot_max - yplot_min)
	#s = [n*ms_slope + 20 for n in yplot]
	
	# Plot the scatter plot now
	#st_line_x = np.insert(xplot,0,0.00001)
	#st_line_x = st_line_x[st_line_x <=  0.000195]
	#plt.plot(st_line_x, 10*st_line_x, 'k--', lw= 1.5, zorder= -1)
	pearson = pearsonr(xplot, yplot)[0]
	label= 'Correlation=' + str(round(pearson, 2))
	plt.grid(True)
	plt.scatter(xplot, yplot, color= rgba_colors, s= msize, edgecolor= 'black', label= label, zorder= 1)
	
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	if xlim is not None:
		plt.xlim(xlim)
	if ylim is not None:
		plt.ylim(ylim)

	# Redefine the xticklabels and yticklabels
	fig.canvas.draw()
	ax = plt.gca()

	if xlim is not None:
		if xlim[0] >= 0:
			plt.xticks(np.arange(0, xlim[1] + 1e-6, 0.0001))
		else:
			plt.xticks(np.arange(xlim[0] + 0.0001, xlim[1] - 0.0001 + 1e-6, 0.0002))

	if ylim is not None:
		if ylim[0] >= 0:
			plt.yticks(np.arange(0, ylim[1] + 1e-6, 0.0001))
		else:
			plt.yticks(np.arange(ylim[0] + 0.0001, ylim[1] - 0.0001 + 1e-6, 0.0002))

	xlabels = get_scaled_ticks(ax.get_xticks(), SCALER)
	ax.set_xticklabels(xlabels)
	ylabels = get_scaled_ticks(ax.get_yticks(), SCALER)
	ax.set_yticklabels(ylabels)	
	
	if xlim is not None:
		if xlim[0] >= 0:
			plt.gcf().subplots_adjust(bottom=0.12,left=0.13)
		else:
			plt.gcf().subplots_adjust(bottom=0.12,left=0.15)	
	
	print("Pearson = {:.4f}".format(pearson))
        # Suppress the marker small so that the marker is not considered as outlier
        # Reference
        # https://stackoverflow.com/a/38454818
        plt.legend(loc= "upper left", markerscale= 0, handletextpad= 0, handlelength= 0)
        savefig(plt, path, tight_flag= True)
	#plt.show()		
	plt.close()

def plot_image_and_landmark(data1, data2, initial_path, label1, function):
	figsize = (8, 6)
	#print("\nPlotting per image")	
	#path = os.path.join("images", initial_path + "_image.png")
	#function(np.mean(data1, 1), np.mean(data2, 1), figsize= figsize, xlabel= r"Predicted " + label1 + r" for each image $(10^{-5})$", ylabel= r"Residual " + label1 + r" for each image $(10^{-4})$", xlim= xlim, ylim= ylim, path= path)
	print("\nPlotting per landmark")	
	path = os.path.join("images", initial_path + "_lmark.png")
	function(data1.flatten(), data2.flatten()    , figsize= figsize, xlabel= "Predicted " + label1 + r" $(\times 10^{-4})$", ylabel= "Residual " + label1 + r" $(\times 10^{-4})$", xlim= xlim, ylim= ylim, path= path)

def get_ylim(xlim):
	if xlim is None:
		ylim = None
	else:
		ylim = xlim
	return ylim

def get_scaled_ticks(ticklabels, scale):
	labels = []
	for i, item in enumerate(ticklabels):
		num = float(item)#guess(item.get_text()))
		labels.append(str(scale*num))
	return labels

#===============================================================================
# Argument Parsing
#===============================================================================
ap     = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id'    , default= 'run_940_evaluate', help= 'input folder relative path')
ap.add_argument(      '--laplacian' , action='store_true',         help="use laplacian ")
ap.add_argument(      '--cumulative', action='store_true',         help="use cumulative")
args   = ap.parse_args()

#===============================================================================
#==== Our folder entries =============
#===============================================================================
folder = os.path.join("abhinav_model_dir", os.path.join(args.exp_id, "300W_test"))
print("Folder= {}".format(folder))

CUMULATIVE = args.cumulative

nme_norm         = np.load(os.path.join(folder, nme_file))
ground, means, L_vect, vis_gt, _ = load_all(folder, load_heatmaps= False, load_cholesky= True, load_vis= True, load_images= False)

# Convert multi-class vis to binary class
vis_gt[vis_gt >= 1] = 1

d = compute_scale(ground)
covar, _ = cholesky_to_covar(L_vect, args.laplacian)

# Sanity checks
print("\nRunning some sanity checks...")
nme_norm_2 = get_err(means, ground)
nme_norm_2 = nme_norm_2.reshape((means.shape[0], means.shape[1]))
# Normalize the mean
_, means_norm = normalize_input(means, d)
nme_norm = np.linalg.norm(means-ground, axis=2)/ d[:, np.newaxis]
print("NME from Lisha's function  = {:.2f}".format(100.0*np.mean(nme_norm_2)))
print("NME from our implementation= {:.2f}".format(100.0*np.mean(nme_norm)))

# Calculate residual covariances
residual = (means-ground)
covar_residual = np.zeros((residual.shape[0], residual.shape[1], 2, 2))
for i in range(residual.shape[0]):
	for j in range(residual.shape[1]):
		temp = residual[i, j]
		temp = temp[:, np.newaxis]
		covar_residual[i, j] = np.matmul(temp, np.transpose(temp))

# Normalize the predicted covariance and residual covariance
print("Normalizing the covariances...")
_, covar_norm = normalize_input(covar, d*d) 
_, covar_residual_norm = normalize_input(covar_residual, d*d)

# Drop covariances where vis is zero
zero_indices = np.where(vis_gt.flatten() == 0)[0]
covar_norm   = covar_norm.reshape((-1,2,2))
covar_residual_norm = covar_residual_norm.reshape((-1,2,2))

#After deletion they may not be in exact multiples of 68 landmarks so making them as 
# with one landmark
covar_norm   = np.delete(covar_norm, zero_indices, 0)
covar_norm   = covar_norm.reshape((-1,1,2,2))

covar_residual_norm = np.delete(covar_residual_norm, zero_indices, 0)
covar_residual_norm = covar_residual_norm.reshape((-1,1,2,2))

# Plot scatter plots
if args.cumulative:
	print("Doing in a cumulative fashion")
	xlim = np.array([0.0000, 0.0002])
else:
	print("Doing individual binwise")
	xlim = np.array([0.0000, 0.0004])
ylim = get_ylim(xlim)

# Along xx
plot_image_and_landmark(covar_norm[:,:,0,0], covar_residual_norm[:,:,0,0], initial_path= "resid_vs_pred_sigma_xx", label1= r"$\Sigma_{xx}$", function= plot_scatter_with_corr)
# Along yy
plot_image_and_landmark(covar_norm[:,:,1,1], covar_residual_norm[:,:,1,1], initial_path= "resid_vs_pred_sigma_yy", label1= r"$\Sigma_{yy}$", function= plot_scatter_with_corr)

# Along xy
xlim = np.array([-0.0005, 0.0005])
ylim = get_ylim(xlim)
plot_image_and_landmark(covar_norm[:,:,0,1], covar_residual_norm[:,:,0,1], initial_path= "resid_vs_pred_sigma_xy", label1= r"$\Sigma_{xy}$", function= plot_scatter_with_corr)

