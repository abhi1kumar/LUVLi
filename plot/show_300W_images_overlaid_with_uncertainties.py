

"""
    Sample Run:
    python plot/show_300W_images_overlaid_with_uncertainties.py --exp_id abhinav_model_dir/run_109_eval/ --laplacian

    Plots the images along with our uncertainties on the images of the Lisha's
    paper.

    Version 1 2019-07-xx Abhinav Kumar
"""

import os, sys
sys.path.insert(0, os.getcwd())
import json
import argparse

from CommonPlottingOperations import *
from pylib.Cholesky import *
import plotting_params as params

input_image_folder = "bigdata1/zt53/data/face/300W/"
output_folder      = "qualitative/300W_test/"
list_300W_images   = "plot/lists/300W_Indoor_Outdoor_Images_for_plotting.txt"

# Which points to show for which images.
# Lisha shows two sets of points in her NeurIPS 2018 paper. 
# These two lists correspond to two rows. 
# We use minus 1 since indexing in Python starts from zero.
lisha_pts_list = np.array([[1,5,9,13,17], [31,46,37,49,55]]).astype(int) - 1

def get_full_folder_path(subfolder):
    return os.path.join(os.path.join(params.IMAGE_DIR, output_folder), subfolder)

#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id'                           , help = "path of the directory", default= "abhinav_model_dir/run_940_evaluate/")
ap.add_argument(      '--laplacian', action = 'store_true' , help= 'use laplacian likelihood instead of Gaussian')
args = ap.parse_args()
model_folder_path = os.path.join(args.exp_id, "300W_test")

# Load image paths to and JSON
with open(list_300W_images) as f:
    rel_paths = [line.rstrip() for line in f.readlines()]

# Use older json for run_940 UGGLI model
if "run_940" in model_folder_path:
    json_to_use = 'dataset/all_300Wtest_val_old.json'
else:
    json_to_use = 'dataset/all_300Wtest_val.json'

with open(json_to_use) as json_file:
    data = json.load(json_file)

# Get the indices at which the paths mentioned are stored in the JSON
num_images = len(rel_paths)
indices = - np.ones((num_images, )).astype(int)

for i in range(num_images):
    if rel_paths[i]:
        key   = os.path.join(input_image_folder, rel_paths[i])
        index = search_json(data, key)
        indices[i] = index
    else:
        print("Empty string")

# Load the things
images, ground_truth, means, cholesky = load_all(model_folder_path)
covar, det_covar = cholesky_to_covar(cholesky)

#===============================================================================
# # Plot following images
#===============================================================================
#print("\nPlotting selected images ...")

subfolder    = "original"
full_folder = get_full_folder_path(subfolder)
makedir(full_folder)

selected_indices = np.array([0])

pts_list = np.array([np.arange(68)])
pts_list_row_for_images = np.zeros((num_images,)).astype(int)


plot_multiple_images(rel_paths, images, pts_list_row_for_images, None, means, covar, ground_truth, indices, selected_indices, save_folder= full_folder, prefix= "")

#===============================================================================
# # Plot groundtruth of all landmarks
#===============================================================================
print("\nPlotting only ground truth without means and covar of all landmarks on selected images...")

subfolder   = "ground_truth"
full_folder = get_full_folder_path(subfolder)
makedir(full_folder)

selected_indices = np.array([0])

pts_list = np.array([np.arange(68)])
pts_list_row_for_images = np.zeros((num_images,)).astype(int)

plot_multiple_images(rel_paths, images, pts_list_row_for_images, pts_list, None, None, ground_truth, indices, selected_indices, save_folder= full_folder, prefix= "")

#===============================================================================
# # Plot means of all landmarks
#===============================================================================
print("\nPlotting means without covar and ground truth of all landmarks on selected images...")

subfolder   = "means"
full_folder = get_full_folder_path(subfolder)
makedir(full_folder)

selected_indices = np.array([0])

pts_list = np.array([np.arange(68)])
pts_list_row_for_images = np.zeros((num_images,)).astype(int)

plot_multiple_images(rel_paths, images, pts_list_row_for_images, pts_list, means, None, None, indices, selected_indices, save_folder= full_folder, prefix= "")


#===============================================================================
# # Plot means and groundtruth points of all landmarks
#===============================================================================
print("\nPlotting means (without covar) and ground truth of all landmarks on selected images...")

subfolder   = "means_and_ground_truth"
full_folder = get_full_folder_path(subfolder)
makedir(full_folder)

selected_indices = np.array([0])

pts_list = np.array([np.arange(68)])
pts_list_row_for_images = np.zeros((num_images,)).astype(int)

plot_multiple_images(rel_paths, images, pts_list_row_for_images, pts_list, means, None, ground_truth, indices, selected_indices, save_folder= full_folder, prefix= "")

#===============================================================================
# # Plot both means and covariances without groundtruth of all landmarks
#===============================================================================
print("\nPlotting both means and covariances without ground truth of all landmarks on selected images ...")

subfolder   = "means_and_covar"
full_folder = get_full_folder_path(subfolder)
makedir(full_folder)

selected_indices = np.array([0])

pts_list = np.array([np.arange(68)])
pts_list_row_for_images = np.zeros((num_images,)).astype(int)

plot_multiple_images(rel_paths, images, pts_list_row_for_images, pts_list, means, covar, None, indices, selected_indices, save_folder= full_folder, prefix= "")

#===============================================================================
# # Plot all predictions with ground truth of all landmarks
#===============================================================================
print("\nPlotting all predictions with ground truth of all landmarks now on selected images ...")

subfolder   = "all_with_all_landmarks"
full_folder = get_full_folder_path(subfolder)
makedir(full_folder)

selected_indices = np.array([0, 1, 5, 6, 7, 8, 10])

pts_list = np.array([np.arange(68)])
pts_list_row_for_images = np.zeros((num_images,)).astype(int)

plot_multiple_images(rel_paths, images, pts_list_row_for_images, pts_list, means, covar, ground_truth, indices, selected_indices, save_folder= full_folder, prefix= "")

#===============================================================================
# Plot all predictions with groudtruth of Lisha's selected landmarks.
#===============================================================================
print("\nPlotting predictions with ground truth on Lisha's selected landmarks on all images similar to Lisha Chen et al, NeurIPS Workshops 2018 ...")

subfolder   = "all_with_lisha_landmarks"
full_folder = get_full_folder_path(subfolder)
makedir(full_folder)

selected_indices = np.arange(num_images)

pts_list_row_for_images     = np.zeros((num_images, )).astype(int)
pts_list_row_for_images[7]  = 1
pts_list_row_for_images[8]  = 1
pts_list_row_for_images[9]  = 1
pts_list_row_for_images[18] = 1
pts_list_row_for_images[19] = 1

plot_multiple_images(rel_paths, images, pts_list_row_for_images, lisha_pts_list, means, covar, ground_truth, indices, selected_indices, save_folder= full_folder, prefix= "")
