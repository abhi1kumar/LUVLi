

"""
    Sample Run:
    python splits_prep/split_aflw_ours.py

    Makes a new split of our AFLW dataset

    Version 1 2019-08-16 Abhinav Kumar
"""

import os
import glob
import numpy as np
from common_functions import grab_files, copy_files

from random import shuffle

def create_mapping_images_landmarks(image_files_grabbed, landmark_files_grabbed):
    num_images = len(image_files_grabbed)
    num_landmarks = len(landmark_files_grabbed)
    assert num_images == num_landmarks

    mapped_landmark_files = []
    for i in range(num_images):
        key = os.path.basename(image_files_grabbed[i])[:-4]

        for j in range(num_landmarks):
            value = os.path.basename(landmark_files_grabbed[j])[:-4]

            if key == value:
                mapped_landmark_files.append(landmark_files_grabbed[j])
                break

    assert num_images == len(mapped_landmark_files)

    return mapped_landmark_files


IMAGE_EXTENSIONS= [".jpg"]
LANDMARK_GROUND_TRUTH_EXTENSIONS = [".pts"]

input_folder    = "bigdata1/zt53/data/face/AFLW_sorted_wmou_171117"
output_folder   = "bigdata1/zt53/data/face/aflw_ours_abhinav_split"

split_ratio = 0.8

sub_directories = ["Frontal", "Left", "LeftHalf", "Right", "RightHalf"]
# Each of the above subdirectories contain the following two folders
sub_directories_2 = ["trainset", "testset"]

images_folder   = "Faces"
landmarks_folder= "Labels"

if os.path.exists(output_folder):
    print("Output directory exists")
else:
    print("Creating directory {}".format(output_folder))
    os.makedirs(output_folder)

# Go to each of the sub-directories
for i in range(len(sub_directories)):

    image_files_grabbed = []
    landmark_files_grabbed = []

    # Put elements from train and test in the same repository
    for j in range(len(sub_directories_2)):
        temp = [input_folder, sub_directories[i], sub_directories_2[j], images_folder]
        input_sub_directory_full_path = os.path.join(*temp)
        # Grab all the image files
        image_files_grabbed += grab_files(input_sub_directory_full_path, IMAGE_EXTENSIONS)
        num_images = len(image_files_grabbed)
        
        # Do the same for landmarks
        temp = [input_folder, sub_directories[i], sub_directories_2[j], landmarks_folder]
        input_landmarks_full_path = os.path.join(*temp)
        # Grab all the landmarks
        landmark_files_grabbed += grab_files(input_landmarks_full_path, LANDMARK_GROUND_TRUTH_EXTENSIONS)
        num_landmarks = len(landmark_files_grabbed)

    print("\n{:10s}".format(sub_directories[i]))
    print("Image_directory=     {:75s} #Images=    {}".format(input_sub_directory_full_path, num_images))
    print("Landmarks_directory= {:75s} #Landmarks= {}".format(input_landmarks_full_path, num_landmarks))

    # Shuffle the images
    shuffle(image_files_grabbed)

    # Create the corresponding mapping of image files.
    mapped_landmark_files = create_mapping_images_landmarks(image_files_grabbed, landmark_files_grabbed)

    train_end_index = int(split_ratio * len(image_files_grabbed))
    print("\nTrain= {} Test= {} Total= {}\n".format(train_end_index, len(image_files_grabbed) - train_end_index, len(image_files_grabbed)))

    for j in range(len(sub_directories_2)):
        if j == 0:
            start_index = 0
            end_index   = train_end_index
        else:
            start_index = train_end_index
            end_index   = len(image_files_grabbed)

        temp = [output_folder, sub_directories[i].lower(), sub_directories_2[j].lower(), images_folder.lower()]
        image_output_sub_directory_full_path = os.path.join(*temp)
        if not os.path.exists(image_output_sub_directory_full_path):
            print("Creating subdirectory {}".format(image_output_sub_directory_full_path))
            os.makedirs(image_output_sub_directory_full_path)

        temp = [output_folder, sub_directories[i].lower(), sub_directories_2[j].lower(), landmarks_folder.lower()]
        landmark_output_sub_directory_full_path = os.path.join(*temp)
        if not os.path.exists(landmark_output_sub_directory_full_path):
            print("Creating subdirectory {}".format(landmark_output_sub_directory_full_path))
            os.makedirs(landmark_output_sub_directory_full_path)

        # Copy all image files
        copy_files(image_files_grabbed[start_index: end_index]  , image_output_sub_directory_full_path)
        # Copy all landmark files
        copy_files(mapped_landmark_files[start_index: end_index], landmark_output_sub_directory_full_path)
