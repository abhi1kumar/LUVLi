
"""
    Visualizes the heatmap at intermediate hourglasses of the CU-Net in terms of 
    examples as well as channels.
    The examples at the intermediate hourglasses are visualized by taking the 
    max at every pixel. Outputs two images from each of the mat files.
 
    Sample Run:
    python visualize_heatmap.py

    Version 3 Abhinav Kumar 2019-07-05 Code refactored
    Version 2 Abhinav Kumar 2019-06-07 Covariance visualizer added
    Version 1 Abhinav Kumar 2019-05-28
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys

sys.path.insert(0, os.getcwd())
from pylib.HeatmapStats import get_spatial_mean_and_covariance

import torch #Needed since we are using spatial_mean function and that operates on Variables
from torch.autograd import Variable 

IMAGE_DIR = "images"
scale     = 3 # scale to display eigen vectors
images_file = "images.npy"
heatmaps_file = "heatmaps.npy"
ground_truth_file       = "ground_truth.npy"
DPI = 200
alpha = 0.5
ms = 2

def swap_channels(input):
    return np.transpose(input, (1, 2, 0))

def plot_and_save_heatmaps(images, heatmaps, prefix, ground_truth= None, means_provided= None, postprocess= None, use_softmax= False, tau=0.02, num_hg= 8, image_min= 0., image_max= 0.1077, res= 256, downsample= 1, use_per_image_min_max= True, show_covariance= False):
    row_end = int(res/ downsample)
    # ==========================================================================
    # Selecting the indices
    # ==========================================================================
    if (heatmaps.shape[1] > 13):
        example_indices  = np.array([1, 4, 7, 10, 13, 16, 19, 22, 23])
    else:
        example_indices  = np.arange(heatmaps.shape[1]) #np.array([2, 3, 5, 6, 10, 11])

    if num_hg == 4:
        landmark_indices = np.array([0, 32, 56, 64])
        hg_indices = np.array([0, 1, 2, 3]) + 4
    elif num_hg == 1:
        landmark_indices = np.array([31, 46, 37, 49, 55])-1 # Same as NeurIPS paper.       1, 5, 9, 13, 17, 31, 46, 37, 49, 55
        hg_indices = [7]
    else:
        landmark_indices = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64])
        hg_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    # ==========================================================================
    # Heatmaps postprocessing
    # ==========================================================================
    if postprocess == "abs":
        heatmaps = np.absolute(heatmaps)
    elif postprocess == "relu":
        heatmaps[heatmaps<0] = 0
    else:
        postprocess="zero"

    if use_softmax:
        softmax = "_smax_" + str(tau)
    else:
        softmax = ""

    print("Range of data = ({}, {})".format(np.min(heatmaps), np.max(heatmaps) ))

    if use_per_image_min_max:
        print("Using image specific min and max")
    else:
        print("Using fixed min = {} and max = {} for every image".format(image_min, image_max))

    means     = Variable(torch.zeros((8, heatmaps.shape[1], heatmaps.shape[2], 2)))
    cov       = Variable(torch.zeros((8, heatmaps.shape[1], heatmaps.shape[2], 2, 2)))
    temp_data = Variable(torch.from_numpy(heatmaps)).float()
    for i in range(num_hg):
        # We do not use any postprocess since it has been already done
        means[hg_indices[i]],cov[hg_indices[i]],_ = get_spatial_mean_and_covariance(temp_data[i], postprocess="", use_softmax=use_softmax, tau=tau)
    # Convert Variables back to numpy
    means = means.data.numpy()
    cov   = cov.data.numpy()


    # ==========================================================================
    # Plot wrt channels
    # ==========================================================================
    num_samples = landmark_indices.shape[0]
    plt.figure(figsize= (2*landmark_indices.shape[0], 2*num_hg), dpi= DPI)
    example_chosen = int(np.median(example_indices))

    for i in range(num_hg):
        data = heatmaps[hg_indices[i], example_chosen]

        # display channels
        for j in range(num_samples):
            index = i* num_samples + j + 1
            image = data[landmark_indices[j],:, :]

            if use_per_image_min_max:
                image_max = np.max(image)
                image_min = np.min(image)

            plt.subplot(num_hg, num_samples, index)
            plt.title("HG " +  str(hg_indices[i]+1) + " Ch " +  str(landmark_indices[j]+1))
            #plt.imshow(swap_channels(images[example_chosen]), vmin=0, vmax=1)
            plt.imshow(image, cmap='jet', vmin= image_min, vmax= image_max, alpha= alpha)
            #plt.axis('off')
            mean_x = means[hg_indices[i], example_chosen, landmark_indices[j], 0]
            mean_y = means[hg_indices[i], example_chosen, landmark_indices[j], 1]
            #plt.plot(mean_x, mean_y, 'bo', markersize= ms)
            if ground_truth is not None:
                gt_x = ground_truth[example_chosen, landmark_indices[j], 0]
                gt_y = ground_truth[example_chosen, landmark_indices[j], 1]
                plt.plot(gt_x, gt_y, 'go', markersize= ms)
            if means_provided is not None:
                means_cal_x = means_provided[example_chosen, landmark_indices[j], 0]
                means_cal_y = means_provided[example_chosen, landmark_indices[j], 1]
                plt.plot(means_cal_x, means_cal_y, 'rx', markersize= ms)

            if show_covariance:
                #Get the EVD of covariance matrix
                cov_example = cov[hg_indices[i], example_chosen, landmark_indices[j]]
                eVa, eVe = np.linalg.eig(cov_example)
                for e, v in zip(eVa, eVe.T):
                    if (e > 0):
                        plt.plot([mean_x, mean_x + scale*np.sqrt(e)*v[0]], [mean_y, mean_y + scale*np.sqrt(e)*v[1]], 'k-', lw=2)    

    plt.tight_layout()
    image_file_path = os.path.join(IMAGE_DIR, prefix + '_channels_pp_' + postprocess + softmax + '.png')
    print("Saving {}".format(image_file_path))    
    plt.savefig(image_file_path)
    plt.close()


    # ==========================================================================
    # Plot wrt examples
    # ==========================================================================
    num_samples      = example_indices.shape[0]

    plt.figure(figsize= (2*num_samples, 2*num_hg), dpi= DPI)
    for i in range(num_hg):
        data = heatmaps[hg_indices[i]]
        maxd = np.max(data[:, landmark_indices], axis=1)

        for j in range(num_samples):
            index = i* num_samples + j + 1
            image = maxd[example_indices[j]][:,:]

            if use_per_image_min_max:
                image_max = np.max(image)
                image_min = np.min(image)

            plt.subplot(num_hg, num_samples, index)
            plt.title("HG " +  str(hg_indices[i]+1) + " Eg " +  str(example_indices[j]))
            plt.imshow(swap_channels(images[example_indices[j]]), vmin=0, vmax=1)
            plt.imshow(image, cmap='jet', vmin= image_min, vmax= image_max, alpha= alpha)
            #plt.axis('off')

            for k in range(landmark_indices.shape[0]):
                mean_x = means[hg_indices[i], example_indices[j], landmark_indices[k], 0]
                mean_y = means[hg_indices[i], example_indices[j], landmark_indices[k], 1]
                #plt.plot(mean_x, mean_y,'bo', markersize= ms)
                if ground_truth is not None:
                    gt_x = ground_truth[example_indices[j], landmark_indices[k], 0]
                    gt_y = ground_truth[example_indices[j], landmark_indices[k], 1]
                    plt.plot(gt_x, gt_y, 'go', markersize= ms)
                if means_provided is not None:
                    means_cal_x = means_provided[example_indices[j], landmark_indices[k], 0]
                    means_cal_y = means_provided[example_indices[j], landmark_indices[k], 1]
                    plt.plot(means_cal_x, means_cal_y, 'rx', markersize= ms)

    plt.tight_layout()
    image_file_path = os.path.join(IMAGE_DIR,  prefix + '_examples_pp_' + postprocess + softmax + '.png')
    print("Saving {}".format(image_file_path))    
    plt.savefig(image_file_path)
    plt.close()

def load(numpy_file_path, prefix):
    print("Loading {} numpy file= {}".format(prefix, numpy_file_path))
    data = np.load(numpy_file_path)
    print("Done")
    print(data.shape)
    print("")
    return data

def load_all(folder_path):
    images       = load(os.path.join(folder_path, images_file)  , "images")
    heatmaps     = load(os.path.join(folder_path, heatmaps_file), "heatmaps")
    ground_truth = load(os.path.join(folder_path, ground_truth_file), "ground_truth")
    means        = load(os.path.join(folder_path, "means.npy"),  "means")  
    return images, heatmaps, ground_truth, means

# ==============================================================================
# Main function
# ==============================================================================
"""
smax_flag  = [True, False]
for i in range(len(smax_flag)):
    filename1 = "abhinav_model_dir/save_heatmaps/heatmap_2019_05_29.npy"

    prefix1   = "blah"
    #plot_and_save_heatmaps(filename1, prefix=prefix1, postprocess="abs" , use_softmax=smax_flag[i])
    #plot_and_save_heatmaps(filename1, prefix=prefix1, postprocess="relu", use_softmax=smax_flag[i])
    #plot_and_save_heatmaps(filename1, prefix=prefix1, postprocess=""    , use_softmax=smax_flag[i])
    #plot_and_save_heatmaps(filename1, prefix=prefix1, postprocess=""    , use_softmax=smax_flag[i], tau=0.1)
    #plot_and_save_heatmaps(filename1, prefix=prefix1, postprocess="relu", use_softmax=smax_flag[i], tau=0.1)

    #filename1 = "abhinav_model_dir/save_heatmaps/heatmap_CUNet.npy"
    #prefix1   = "CUNet"
    #save_heatmaps(filename1, prefix=prefix1, postprocess="abs" , use_softmax=smax_flag[i])
    #save_heatmaps(filename1, prefix=prefix1, postprocess="relu", use_softmax=smax_flag[i])
    #save_heatmaps(filename1, prefix=prefix1, postprocess=""    , use_softmax=smax_flag[i])
"""
#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument    ('-f', '--folder_path' , help = "path of the directory", default="abhinav_model_dir/run_504_ibug_selected/")
ap.add_argument    ('-d', '--downsample'  , help = "downsample", type= int, default= 4)
ap.add_argument    ('-n', '--num_hg'      , help = "number of hourglasses", type= int, default= 1)
args = ap.parse_args()

folder_path = args.folder_path

if (folder_path[-1] == "/"):
    prefix   = os.path.dirname(folder_path).split("/")[-1]
else:
    prefix   = os.path.basename(folder_path)
print(prefix + "\n")
images, heatmaps, ground_truth, means = load_all(folder_path)

plot_and_save_heatmaps(images, heatmaps, prefix= prefix, ground_truth= ground_truth, means_provided= means, postprocess="relu", num_hg= args.num_hg, downsample= args.downsample)
