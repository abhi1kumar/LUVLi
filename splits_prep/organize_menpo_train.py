

"""
    Sample Run:
    python splits_prep/organize_menpo_train.py

    Copies landmark points and the images of menpo dataset into a common 
    directory so that get_jsons_from_config.py script could be used. Before 
    copying check if the landmark corresponding to the image has 68 landmark
    points otherwise do not copy.

    Version 1 2019-06-20 Abhinav Kumar
"""

import os
import numpy as np
from common_functions import grab_files, copy_files

LANDMARK_DELIMITER              = " "
LANDMARK_HEADERS                = 3
LANDMARK_FOOTERS                = 1
IMAGE_EXTENSIONS= [".png", ".jpg"]
LANDMARK_GROUND_TRUTH_EXTENSIONS = [".pts"]

input_folder    = "./bigdata1/zt53/data/face/menpo/trainset"
output_folder   = "./bigdata1/zt53/data/face/menpo_organized/trainset"

if os.path.exists(output_folder):
    print("Output directory exists")
else:
    print("Creating directory {}".format(output_folder))
    os.makedirs(output_folder)


input_sub_directory_full_path  = input_folder
# Grab all the image files
image_files_grabbed = grab_files(input_sub_directory_full_path, IMAGE_EXTENSIONS)
num_images_1 = len(image_files_grabbed)

image_files_68landmarks = []
landmark_files_68landmarks = []

# Read all the landmarks and check if there are 68 landmarks for each of them
# since some images in the menpo also have 39 landmarks which can not be used 
# for testing. Copy only those which have 68 landmarks
for i in range(num_images_1):
    landmark_file_path = image_files_grabbed[i][:-4] + LANDMARK_GROUND_TRUTH_EXTENSIONS[0]
    landmark = np.genfromtxt(landmark_file_path, skip_header=LANDMARK_HEADERS, skip_footer=LANDMARK_FOOTERS, delimiter=LANDMARK_DELIMITER)

    if landmark.shape[0] == 68:
        image_files_68landmarks.append(image_files_grabbed[i])
        landmark_files_68landmarks.append(landmark_file_path)

num_images = len(image_files_68landmarks)
print("Image_directory= {:54s} #Images= {} #Images_with_68_landmarks= {}".format(input_sub_directory_full_path, num_images_1, num_images))

output_sub_directory_full_path = output_folder
if not os.path.exists(output_sub_directory_full_path):
    print("Creating subdirectory {}".format(output_sub_directory_full_path))
    os.makedirs(output_sub_directory_full_path)

# Copy all image files
copy_files(image_files_68landmarks   , output_sub_directory_full_path)
# Copy all landmark files
copy_files(landmark_files_68landmarks, output_sub_directory_full_path)
