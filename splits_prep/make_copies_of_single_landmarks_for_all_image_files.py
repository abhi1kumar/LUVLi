

"""
    python splits_prep/make_copies_of_single_landmarks_for_all_image_files.py --image_folder bigdata1/zt53/data/face/demo --reference_file bigdata1/zt53/data/face/demo/sample.pts

    Takes a landmark file and creates a landmark file for each image by copying
    the reference file in the directory with the same name as image.

    This makes the image folder usable by get_jsons_from_config.py file

"""
import os
from common_functions import *


#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image_folder'  , help= "folder containing extracted images from video", default= "bigdata1/zt53/data/face/demo")
ap.add_argument('-r', '--reference_file', help= "file which should be copied "                 , default= "bigdata1/zt53/data/face/demo/sample.pts")
args = ap.parse_args()

input_folder   = args.input_folder
reference_file = args.reference_file

image_files    = grab_files(folder_full_path= input_folder, EXTENSIONS= [".png",".jpg"])
num_images     = len(image_files)

for i in range(num_images):
    landmark_file = image_files[i][:-4] + ".pts"
    
    # Check if landmark file is there
    if os.path.exists(landmark_file):
        pass
    else:
        # make a copy of reference file
        command = "cp " + reference_file + " " + landmark_file
        execute(command)
