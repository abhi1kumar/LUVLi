

"""
    Sample Run:
    python plot/show_dataset_images_overlaid_with_uncertainties.py --exp_id abhinav_model_dir/run_507_evaluate  -s 3 --laplacian
    python plot/show_dataset_images_overlaid_with_uncertainties.py --exp_id abhinav_model_dir/run_1005_evaluate -s 4 --laplacian
    python plot/show_dataset_images_overlaid_with_uncertainties.py --exp_id abhinav_model_dir/run_5004_evaluate -s 5 --laplacian

    Selects some random NUM_IMAGES from the JSON and then plots the means and covar along with the ground truth of all landmarks on
    the chosen dataset.

    Version 1 2019-11-22 Abhinav Kumar
"""

import os, sys
sys.path.insert(0, os.getcwd())
import json
import argparse

from CommonPlottingOperations import *
from pylib.Cholesky import *

IMAGE_DIR  = "images/qualitative/"
NUM_IMAGES = 50
SEED_VAL   = 1729 # Fix the seed so that everytime we see the same images being plotted

#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id' , help = "path of the directory", default= "abhinav_model_dir/run_940_evaluate/")
ap.add_argument(      '--laplacian', action = 'store_true'   , help= 'use laplacian likelihood instead of Gaussian')
ap.add_argument('-s', '--split', default= 5, type= int, help= "split_to_use")
args = ap.parse_args()

model_folder_path = os.path.join(args.exp_id, "300W_test")


#===============================================================================
# Get what all jsons we have to run depending on the split
#===============================================================================
if args.split == 2:
    print("Using models trained on 300-W Split " + str(args.split))
    keyword   = ["300W_test" ]
    jsons     = ["dataset/all_300Wtest_val.json"]
    model_folder_path = os.path.join(args.exp_id, "300W_test")
    class_num = 68
elif args.split == 3:
    print("Using models trained on original AFLW-19 " + str(args.split))
    keyword   = ["aflw_full" ]
    jsons     = ["dataset/aflw_test_all.json"]
    model_folder_path = os.path.join(args.exp_id, "aflw_full")
    class_num = 19
elif args.split == 4:
    print("Using models trained on WFLW-98 " + str(args.split))
    keyword   = ["wflw_full"]
    jsons     = ["dataset/wflw_test.json"]
    model_folder_path = os.path.join(args.exp_id, "wflw_full")
    class_num = 98
elif args.split == 5:
    print("Using models trained on AFLW-68 " + str(args.split))
    keyword   = ["aflw_ours"]
    jsons     = ["dataset/aflw_ours_all_val.json"]
    model_folder_path = os.path.join(args.exp_id, "aflw_ours_all")
    class_num = 68
else:
    print("Some unknown split. Aborting!!!")
    sys.exit(0)


keyword = keyword[0]
json_to_use = jsons[0]
IMAGE_DIR = os.path.join(IMAGE_DIR, keyword)

with open(json_to_use) as json_file:
    data_json = json.load(json_file)
num_images_json = len(data_json)

rel_paths  = []
np.random.seed(SEED_VAL)
indices = np.random.choice(num_images_json, NUM_IMAGES, replace=False)
print("Number of images = {}".format(NUM_IMAGES))

for i in range(NUM_IMAGES):
    rel_paths.append(os.path.basename(data_json[indices[i]]['img_paths']))
#print(rel_paths)
# Load the things
images, ground_truth, means, cholesky, vis_gt, vis_estimated = load_all(model_folder_path, load_vis= True)
covar, det_covar = cholesky_to_covar(cholesky, args.laplacian)

#===============================================================================
# # Plot all landmark points of following images
#===============================================================================
print("\nPlotting all landmarks now on selected images ...")
selected_indices = np.arange(NUM_IMAGES)

pts_list = np.array([np.arange(class_num)])
pts_list_row_for_images = np.zeros((NUM_IMAGES,)).astype(int)

plot_multiple_images(rel_paths, images, pts_list_row= pts_list_row_for_images, pts_list= pts_list, means= means, covar= covar, ground_truth= ground_truth, indices= indices, selected_indices= selected_indices, save_folder= IMAGE_DIR, prefix= "", vis_gt= vis_gt, vis_estimated= vis_estimated, use_different_color_for_ext_occ= False)
