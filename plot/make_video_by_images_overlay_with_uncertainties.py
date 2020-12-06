

"""
    Sample Run:
    python plot/make_video_by_images_overlay_with_uncertainties.py --exp_id abhinav_model_dir/run_5004_evaluate/demo/ --val_json dataset/demo_val.json
    python plot/make_video_by_images_overlay_with_uncertainties.py --exp_id abhinav_model_dir/run_940_evaluate_vw/
    python plot/make_video_by_images_overlay_with_uncertainties.py --exp_id abhinav_model_dir/run_940_evaluate_vw/ --kalman

    Makes videos from the json file and saved images by overlaying predictions-
    means and covariances on them. Also does not those landmarks whose
    estimated visibilities are less than threshold controlled by --threshold
    option

    Scale for the 001 video should be 1.1 and center should be 805, 210
    
    Version 2 2019-09-05 Abhinav Kumar Kalman Filter Added
    Version 1 2019-07-xx Abhinav Kumar
"""

import os, sys
sys.path.insert(0, os.getcwd())
import json
import argparse
import cv2

from CommonPlottingOperations import *
from pylib.Cholesky import *
from pylib.KalmanFilter import *
from splits_prep.common_functions import *

NUM_LMARKS      = 68
VIDEO_DIR       = "videos"
ORIG_FRAME_RATE = 25

def plot_multiple_images(rel_paths, pts_list_row, pts_list, means, covar, ground_truth, indices, selected_indices, prefix= "", save_folder= "videos/frames",  vis_gt= None, vis_estimated= None, kalman= False, fps= ORIG_FRAME_RATE):
    """
        rel_path     = path of the variable
        pts_list_row = row for pts_list which is to be plotted
        selected_indices = indices of the image in the JSON which is to be saved

    """
    num_variable_in_state = 6
    if kalman:
        if num_variable_in_state == 6:
            delta_t = 1#/float(fps)
            half_delta_t_square   = 0.5*delta_t**2
            half_delta_t_cube     = 0.5*delta_t**3
            one_four_delta_t_four = 0.25*delta_t**4

            # Reference: 
            # Automatic Facial Landmark Tracking in Video Sequences using Kalman Filter Assisted Active Shape Models, ECCV 2010
            # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.651.2739&rep=rep1&type=pdf
            sigma_v_square = 0.1

            # Get the transition matrix A
            A = np.eye(6)
            A[0,2] = delta_t
            A[0,4] = half_delta_t_square
            A[1,3] = delta_t
            A[1,5] = half_delta_t_square
            A[2,4] = delta_t
            A[3,5] = delta_t

            # Get the state noise matrix Q now
            Q = np.zeros((6, 6))
            Q[0,0] = one_four_delta_t_four
            Q[0,2] = half_delta_t_cube
            Q[0,4] = half_delta_t_square
            
            Q[1,1] = one_four_delta_t_four 
            Q[1,3] = half_delta_t_cube
            Q[1,5] = half_delta_t_square
            
            Q[2,0] = half_delta_t_cube
            Q[2,2] = half_delta_t_square
            Q[2,4] = delta_t

            Q[3,1] = half_delta_t_cube
            Q[3,3] = half_delta_t_square
            Q[3,5] = delta_t

            Q[4,0] = half_delta_t_square
            Q[4,2] = delta_t
            Q[4,4] = 1

            Q[5,1] = half_delta_t_square
            Q[5,3] = delta_t
            Q[5,5] = 1

            Q = Q* sigma_v_square

            # Get the Process matrix H
            H = np.zeros((2,6))
            H[0,0] = 1
            H[1,1] = 1


        elif num_variable_in_state == 2:
            # Get the transition matrix A
            A = np.eye(2)
            # Get the state noise matrix Q now
            sigma_v_square = 1e-3
            Q = np.eye(2)*sigma_v_square

            H = np.eye(2)

        else:
            print("Not implemented")

        total_points = means.shape[1]
        s_new = np.zeros((total_points, num_variable_in_state))
        P_new = np.eye(num_variable_in_state)        
        P_new = P_new[np.newaxis,:]
        P_new = np.repeat(P_new, total_points, axis= 0)

        print(A) 
        print(Q)
        print(H)
        print(P_new.shape)
        print("Running Kalman filter")

    num_images = selected_indices.shape[0]
    for i in range(num_images):
        # Start plotting
        fig= plt.figure(dpi= DPI)
        ax = fig.add_subplot(111)

        if kalman:
            if i == 0:
                for j in range(total_points):
                    s_new[j,:2] = means[selected_indices[i], j]

            if i > 0:
                # Update means and covariances
                for j in range(17):
                    s_new[j], P_new[j] = kalman_filter(A, Q, H, R= covar[selected_indices[i], j], z= means[selected_indices[i], j], s= s_new[j], P= P_new[j])
                    means[selected_indices[i], j] = s_new[j, :2]

        if pts_list is not None:
            pts_list_1 = pts_list[pts_list_row[selected_indices[i]]]
        
        plot_image(ax, images, pts_list_1= pts_list_1, means= means, covar= covar, ground_truth= ground_truth, i= selected_indices[i], indices= indices, vis_gt= vis_gt, vis_estimated= vis_estimated, use_different_color_for_ext_occ= False, use_threshold_on_vis= True, threshold_on_vis= args.threshold)

        save_path = os.path.join(save_folder, prefix + os.path.basename(rel_paths[selected_indices[i]]))
        if i %100 == 0 or i== num_images-1:
            print(i)
            show_message= True
        else:
            show_message= False

        # No margin around the border
        # Reference https://stackoverflow.com/a/50512838
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        savefig(plt, save_path, show_message= show_message, tight_flag= True, newline= False)
        plt.close()

#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id'                     , help= "path of the directory"             , default= "abhinav_model_dir/run_940_300vw/")
ap.add_argument('-s', '--save'                       , help= "path of the saved image directory" , default= "videos/frames")
ap.add_argument('-k', '--kalman', action='store_true', help= "apply Kalman Filtering"            , default= False)
ap.add_argument('-j', '--val_json'                   , help= "path of the val json"              , default= "dataset/300vw_001_val.json")
ap.add_argument('-t', '--threshold' , type= float    , help= "threshold to not show landmarks"   , default= 0.65)
ap.add_argument(      '--video_name'                 , help= "name of the video"                 , default= "demo.avi")
ap.add_argument(      '--frame_rate', type= int      , help= "frame rate of the output video"    , default= 12)
args = ap.parse_args()
model_folder_path = args.exp_id

if not os.path.isdir(args.save):
    print("Making directory " + args.save)
    os.makedirs(args.save)

with open(args.val_json) as json_file:
    data = json.load(json_file)

num_images = len(data)
rel_paths = []
for i in range(num_images):
    rel_paths.append(data[i]["img_paths"])
indices = np.arange(num_images)

# Load the predictions
# We do not have ground_truth visibility and most of the time it is dummy one.
images, ground_truth, means, cholesky, _, vis_estimated = load_all(model_folder_path, load_vis= True)
covar, det_covar = cholesky_to_covar(cholesky)

#===============================================================================
# # Plot all landmark points of following images
#===============================================================================
print("\nPlotting all landmarks now on selected images ...")
selected_indices = np.arange(num_images)

# The pts_list contains the list of the indices of the points which will be plotted.
pts_list = np.array([np.arange(NUM_LMARKS)])

# pts_list_row refers to the row in the pts_list variable for a particular image. 
# There is only one entry in pts_list which contains all the points.
# So, the following code is equivalent to plot all the points for the variable.
pts_list_row_for_images = np.zeros((num_images,)).astype(int)

plot_multiple_images(rel_paths, pts_list_row_for_images, pts_list, means, covar, None, indices, selected_indices, prefix= "", save_folder= args.save, vis_gt= None, vis_estimated= vis_estimated, kalman= args.kalman)

#===============================================================================
# Now make the video by loading the saved images
#===============================================================================
img_array       = []
saved_files     = sorted(grab_files(folder_full_path= args.save, EXTENSIONS= [".png",".jpg"]))
video_save_path = os.path.join(VIDEO_DIR, args.video_name)

# Reference
# https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
for filename in saved_files:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'DIVX'), args.frame_rate, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("\nSaved output video at {} fps to {}".format(args.frame_rate, video_save_path))
