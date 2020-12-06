

"""
    Sample Run
    python plot/plot_histogram_smallest_eigen_value.py -i run_109_evaluate --laplacian
    
    Plots histogram of smallest eigen value of the covariance matrix of each 
    landmark
"""
import os,sys
sys.path.append(os.getcwd())
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from CommonPlottingOperations import *
from pylib.Cholesky import *
import plotting_params as params

fs    = 30
lw    = 3
matplotlib.rcParams.update({'font.size': fs})

#===============================================================================
# Argument Parsing
#===============================================================================
ap     = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id', default= 'run_940_evaluate', help= 'input folder relative path')
ap.add_argument(      '--laplacian', action = 'store_true'   , help= 'use laplacian likelihood instead of Gaussian')
args   = ap.parse_args()

#===============================================================================
#==== Our folder entries =============
#===============================================================================
folder = os.path.join("abhinav_model_dir", os.path.join(args.exp_id, "300W_test"))
print("Folder= {}".format(folder))

# Cholesky is saved in 256x256 dimensions
cholesky = np.load(os.path.join(folder, "cholesky.npy"))/4.0
print(cholesky.shape)
covar, _ = cholesky_to_covar(cholesky, laplacian= args.laplacian)
print(covar.shape)

# Calculate the eigen values
eig_norm_cov = np.zeros((covar.shape[0], covar.shape[1], 2))
for i in range(covar.shape[0]):
	for j in range(covar.shape[1]):
		w, v = np.linalg.eig(covar[i,j])
		eig_norm_cov[i,j] = np.sort(w) 
		
eig_cov_minor = np.sqrt(eig_norm_cov[:, :, 0]).flatten()
if np.max(eig_cov_minor) < 4:
    xmax = 2
    size = (8,6)
else:
    xmax = np.max(eig_cov_minor)
    size = (9.6,6)

plt.figure(figsize= size, dpi = params.DPI)
n, bins, patches = plt.hist(eig_cov_minor, facecolor= params.color2, bins= 'fd', density= True)
plt.xlabel('Pixels')
plt.ylabel('PDF of sqrt of\nsmallest eigenvalue')
plt.xlim([0., xmax])
plt.xticks(np.arange(0, xmax+1, step=1))
plt.grid(True)
path = "images/histogram_covar_smallest_eigen_value.png"
# Make small room for the labels so that they are not cutoff
plt.gcf().subplots_adjust(bottom=0.15, left= 0.14)
savefig(plt, path, tight_flag= True)
plt.show()
