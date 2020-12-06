

"""
    Sample Run:
    python plot/show_aflw_ours_overlaid_with_uncertainties.npy --exp_id abhinav_model_dir/run_5001_eval/

    Plots the images along with our uncertainties on the images of our dataset

    Version 1 2019-11-11 Abhinav Kumar
"""

import os, sys
sys.path.insert(0, os.getcwd())
import json
import argparse

from CommonPlottingOperations import *
from pylib.Cholesky import *

folder = "bigdata1/zt53/data/face/aflw_ours_organized/"
IMAGE_DIR = "images"
DPI   = 200
alpha = 0.9
ms    = 2
lw    = 1.5
dodge_blue = np.array([30, 144, 255])/255.

#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument    ('-i', '--exp_id' , help = "path of the directory", default= "abhinav_model_dir/run_5001_evaluate/")
args = ap.parse_args()
model_folder_path = os.path.join(args.exp_id, "aflw_ours_profile")

print("Loading JSON data...")
with open('dataset/aflw_ours_profile_val.json') as json_file:
    json_data = json.load(json_file)
print("Done")
print(len(json_data))

num_images = 10#len(data)
indices    = - np.ones((num_images, )).astype(int)
rel_paths  = []

for i in range(num_images):
        indices[i] = i
        temp = json_data[i]
        rel_paths.append(temp['img_paths'])

# Load the things
images, ground_truth, means, cholesky, vis_gt, vis_estimated = load_all(model_folder_path, load_vis= True)
covar, det_covar = cholesky_to_covar(cholesky)

#===============================================================================
# Plot all landmark points of following images
#===============================================================================
print("\nPlotting all landmarks now on selected images ...")
selected_indices = indices

pts_list = np.array([np.arange(68)])
pts_list_row_for_images = np.zeros((num_images,)).astype(int)

plot_multiple_images(rel_paths, images, pts_list_row_for_images, pts_list, means, covar, ground_truth, indices, selected_indices, prefix= "aflw_ours/all_landmarks/all_", vis_gt= vis_gt, vis_estimated= vis_estimated)
