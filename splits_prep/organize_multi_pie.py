

"""
    Sample Run:
    python splits_prep/organize_multi_pie.py

    Reads the MultiPIE dataset and then puts the landmarks in the specified
    location

    Version 2 2019-06-24 Abhinav Kumar Some images have less than 68 landmarks. Do not use them
    Version 1 2019-06-24 Abhinav Kumar
"""

import os
import sys
from common_functions import grab_files, copy_files
from scipy.io import loadmat

def convert(i):
    return str(i+1)

IMAGE_EXTENSIONS  = ".png"
LANDMARK_GROUND_TRUTH_EXTENSIONS  = "_lm.mat"


#base_folder       = "/projects/CV2/FacePak2/Multi-PIE_original_data/"
base_folder       = "./bigdata1/zt53/data/face/Multi-PIE_original_data/"
output_folder     = "./bigdata1/zt53/data/face/multi_pie_organized"
camera_id         = "05_1" # frontal
lighting_id       = "05"   # 08 has absolutely no shadow and clearly visible face but doesnot have ground truth. Only 05 has ground truth
recording_id      = "01"

landmark_folder_rel = "MPie_Labels/labels" 
images_folder_rel   = "Multi-PIE/data"
num_sessions        = 4
num_subjects        = 292
num_recordings      = 3
join_char           = "_"
camera_id_no_sep  = camera_id.replace("_", "")

image_files_68landmarks = []
landmark_files_68landmarks = []
cnt = 0

for i in range(num_sessions):
    session_id = convert(i).zfill(2)
    session_path = os.path.join("session" + session_id, "multiview")
 
    for j in range(num_subjects):
        subject_id = convert(j).zfill(3)

        # The first recording contains normal expression. So, using only that
        for k in range(1):#num_recordings):
            recording_id = convert(k).zfill(2)

            image_folder_full   = os.path.join(base_folder, images_folder_rel, session_path, subject_id, recording_id, camera_id)
            image_filename_list = [subject_id, session_id, recording_id, camera_id_no_sep, lighting_id]
            image_filename_base = join_char.join(image_filename_list)
            image_filename      = image_filename_base + IMAGE_EXTENSIONS

            image_filename_full_path = os.path.join(image_folder_full, image_filename)


            landmark_folder_full = os.path.join(base_folder, landmark_folder_rel, camera_id_no_sep)
            landmark_filename     = image_filename_base + LANDMARK_GROUND_TRUTH_EXTENSIONS
            landmark_filename_full_path = os.path.join(landmark_folder_full, landmark_filename)

            # Check if both the files are present otherwise ignore
            if os.path.isfile(image_filename_full_path) and os.path.isfile(landmark_filename_full_path):
                # Read if the landmark has exactly 68 points
                data = loadmat(landmark_filename_full_path)
                
                if (data['pts'].shape[0] == 68):
                    cnt += 1
                    image_files_68landmarks.append(image_filename_full_path)
                    landmark_files_68landmarks.append(landmark_filename_full_path)


print("Image_directory= {:54s} #Images= {}".format(base_folder, cnt))
if not os.path.exists(output_folder):
    print("Creating directory {}".format(output_folder))
    os.makedirs(output_folder)
else:
    print("Directory present {}".format(output_folder))

# Copy all image files
copy_files(image_files_68landmarks   , output_folder)
# Copy all landmark files
copy_files(landmark_files_68landmarks, output_folder)
