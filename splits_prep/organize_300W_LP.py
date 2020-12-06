

"""
    Sample Run:
    python splits_prep/organize_300W_LP.py

    Moves landmark points and the images of 300W_LP dataset into a common 
    directory so that get_jsons_from_config.py script could be used.

    Version 1 2019-06-13 Abhinav Kumar
"""

import os
import glob
import numpy as np
from common_functions import grab_files, copy_files

IMAGE_EXTENSIONS= [".png", ".jpg"]
LANDMARK_GROUND_TRUTH_EXTENSIONS = [".pts", ".mat"]

input_folder    = "./bigdata1/zt53/data/face/300W_LP"
output_folder   = "./bigdata1/zt53/data/face/300W_LP_organized"

sub_directories = ["AFW", "HELEN", "IBUG", "LFPW"]
landmarks_folder= "landmarks"

if os.path.exists(output_folder):
    print("Output directory exists")
else:
    print("Creating directory {}".format(output_folder))
    os.makedirs(output_folder)

# Go to each of the sub-directories
for i in range(len(sub_directories)):
    input_sub_directory_full_path  = os.path.join(input_folder, sub_directories[i])
    # Grab all the image files
    image_files_grabbed = grab_files(input_sub_directory_full_path, IMAGE_EXTENSIONS)
    num_images = len(image_files_grabbed)
    
    # Do the same for landmarks
    input_landmarks_full_path = os.path.join(input_folder, landmarks_folder, sub_directories[i])
    # Grab all the landmarks
    landmark_files_grabbed = grab_files(input_landmarks_full_path, LANDMARK_GROUND_TRUTH_EXTENSIONS)
    num_landmarks = len(landmark_files_grabbed)

    print("\n{:10s}".format(sub_directories[i]))
    print("Image_directory= {:54s} #Images= {}".format(input_sub_directory_full_path, num_images))
    print("Landmarks_directory= {:50s} #Landmarks= {}".format(input_landmarks_full_path, num_landmarks))

    output_sub_directory_full_path = os.path.join(output_folder, sub_directories[i])
    if not os.path.exists(output_sub_directory_full_path):
        print("Creating subdirectory {}".format(output_sub_directory_full_path))
        os.makedirs(output_sub_directory_full_path)

    # Copy all image files
    copy_files(image_files_grabbed   , output_sub_directory_full_path)
    # Copy all landmark files
    copy_files(landmark_files_grabbed, output_sub_directory_full_path)
