

"""
    Sample Run:
    python plot/show_300W_images_overlaid_with_uncertainties.py --exp_id abhinav_model_dir/run_940_eval/

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
from utils import imutils

folder = "bigdata1/zt53/data/face/aflw_ours_organized/"
IMAGE_DIR = "images/qualitative"
DPI   = 200
alpha = 0.9
ms    = 1
lw    = 1.5

# Load image paths to and JSON
with open('plot/AFLW_ours_images_for_plotting.txt') as f:
    rel_paths = [line.rstrip() for line in f.readlines()]

with open('dataset/aflw_ours_all_train.json') as json_file:
    data_train = json.load(json_file)
with open('dataset/aflw_ours_all_val.json') as json_file:
	data_val   = json.load(json_file)

# Combine the two jsons
data_all = []
for i in range(len(data_train)):
    data_all.append(data_train[i])
for i in range(len(data_val)):
    data_all.append(data_val[i])

print("JSON operations done")
num_images = len(rel_paths)
indices_in_json = - np.ones((num_images, )).astype(int)

for i in range(num_images):
    if rel_paths[i]:
        key   = os.path.join(folder, rel_paths[i])
        index = search_json(data_all, key)
        indices_in_json[i] = index
    else:
        print("Empty string")

#===============================================================================
# # Plot all groundtruth
#===============================================================================
print("\nPlotting all ground truth now on selected images ...")
full_folder = os.path.join(IMAGE_DIR, "aflw_ours/original_res")
makedir(full_folder)

for i in range(num_images):
    index     = indices_in_json[i]
    img_data  = data_all[index]

    fig= plt.figure(dpi= DPI)
    img_path = img_data['img_paths']
    img = imutils.load_image(img_path)
    pts = np.array(img_data['pts'])

    # Assume all points are visible for a dataset. This is a multiclass
    # visibility
    vis = np.ones(pts.shape[0])
    # The pts which are labelled -1 in both x and y are not visible points
    self_occluded_landmark     = (pts[:,0] == -1) & (pts[:,1] == -1)
    external_occluded_landmark = (pts[:,0] <  -1) & (pts[:,1] < -1)

    vis[self_occluded_landmark]     = 0
    vis[external_occluded_landmark] = 2

    pts= np.abs(pts)

    # Get visible points which have 1 in the visibility
    landmark_id_list_1 = np.where(vis == 1)[0]
    pts_1 = pts[landmark_id_list_1]

    # Get externally occluded points which have zero in the visibility
    landmark_id_list_2 = np.where(vis == 2)[0]
    pts_2 = pts[landmark_id_list_2]

    plt.imshow(swap_channels(img))
    plt_pts(plt, pts_2, color= "red"      , text= None, shift= 2, text_color= "magenta", ms= 2, ignore_text= True)
    plt_pts(plt, pts_1, color= "limegreen", text= None, shift= 2, text_color= "blue", ms= 2, ignore_text= True)
    plt.xticks([])
    plt.yticks([])    
    #plt.show()
    path = os.path.join(full_folder, os.path.basename(img_path))
    savefig(plt, path, tight_flag= True, newline= False)
    plt.close()
