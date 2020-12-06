

"""
    Sample Run:
    python splits_prep/organize_AFLW_ours.py

    Moves landmark points and the images of AFLW dataset into a common 
    directory so that get_jsons_from_config.py script could be used.

    Version 1 2019-07-18 Abhinav Kumar
"""

import os
import glob
import numpy as np
from common_functions import grab_files, copy_files

from random import shuffle

IMAGE_EXTENSIONS= [".jpg"]
LANDMARK_GROUND_TRUTH_EXTENSIONS = [".pts"]

input_folder    = "bigdata1/zt53/data/face/aflw_ours_abhinav_split"
output_folder   = "bigdata1/zt53/data/face/aflw_ours_organized"

sub_directories   = ["frontal", "left", "lefthalf", "right", "righthalf"]
# Each of the above subdirectories contain the following two folders
sub_directories_2 = ["trainset", "testset"]

images_folder   = "faces"
landmarks_folder= "labels"

if os.path.exists(output_folder):
    print("Output directory exists")
else:
    print("Creating directory {}".format(output_folder))
    os.makedirs(output_folder)

# Go to each of the sub-directories
for i in range(len(sub_directories)):

    for j in range(len(sub_directories_2)):
         # Put elements from train and test in the different directory
        temp = [input_folder, sub_directories[i], sub_directories_2[j], images_folder]
        input_sub_directory_full_path = os.path.join(*temp)
        # Grab all the image files
        image_files_grabbed = grab_files(input_sub_directory_full_path, IMAGE_EXTENSIONS)
        num_images = len(image_files_grabbed)
        
        # Do the same for landmarks
        temp = [input_folder, sub_directories[i], sub_directories_2[j], landmarks_folder]
        input_landmarks_full_path = os.path.join(*temp)
        # Grab all the landmarks
        landmark_files_grabbed = grab_files(input_landmarks_full_path, LANDMARK_GROUND_TRUTH_EXTENSIONS)
        num_landmarks = len(landmark_files_grabbed)

        print("\n{:10s}".format(sub_directories[i]))
        print("Image_directory=     {:75s} #Images=    {}".format(input_sub_directory_full_path, num_images))
        print("Landmarks_directory= {:75s} #Landmarks= {}".format(input_landmarks_full_path, num_landmarks))

        temp = [output_folder, sub_directories[i], sub_directories_2[j]]
        output_sub_directory_full_path = os.path.join(*temp)
        if not os.path.exists(output_sub_directory_full_path):
            print("Creating subdirectory {}".format(output_sub_directory_full_path))
            os.makedirs(output_sub_directory_full_path)

        # Copy all image files and landmark files to the same directory as required by get_jsons_from_config.py script
        copy_files(image_files_grabbed   , output_sub_directory_full_path)
        copy_files(landmark_files_grabbed, output_sub_directory_full_path)
