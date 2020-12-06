

"""
    Sample Run:
    python splits_prep/get_jsons_from_config.py -i splits_prep/config_dummy.txt
	python splits_prep/get_jsons_from_config.py -i splits_prep/config_trump_test.txt --noise 0.00

    Reads a config file to create two jsons - one for train and one for test

    Version 4 Abhinav Kumar 2019-06-25 Support for Multi_PIE dataset mat files added
    Version 3 Abhinav Kumar 2019-06-14 Support for ground truth bounding boxes from landmark points
    Version 2 Abhinav Kumar 2019-06-13 Support for reading ground truth from mat files of 300W_LP added
    Version 1 Abhinav Kumar 2019-05-23
"""

import argparse
import os
import sys
import glob

# Reference https://stackoverflow.com/a/1875584
if sys.version_info < (2, 6):
    import ConfigParser as config_parser
else:
    import configparser as config_parser

import numpy as np
from scipy.io import loadmat
import json

sys.path.insert(0, './src')
from image_bounding_boxes import get_image_bounding_boxes_annotation_from_mat, get_image_bounding_boxes_from_landmarks


LANDMARK_DELIMITER              = " "
LANDMARK_HEADERS                = 3
LANDMARK_FOOTERS                = 1
LANDMARK_GROUND_TRUTH_EXTENSIONS = [".pts", ".mat"]
IMAGE_EXTENSIONS                = [".png", ".jpg"]
TRAIN_JSON                      = "train.json"
VAL_JSON                        = "val.json"


def convert_string_to_list(str):

    return [x.strip() for x in str.split(',')]

def read_landmark_ground_truth(path):
    """
        Reads the landmark groundtruth from a file specified
        Typically it is returned as list of num_points x 2
        NOTE - Numpy array is not JSON serializable
    """
    _, extension = os.path.splitext(path)

    if extension == ".pts":
        landmark = np.genfromtxt(path, skip_header=LANDMARK_HEADERS, skip_footer=LANDMARK_FOOTERS, delimiter=LANDMARK_DELIMITER)
    elif extension == ".mat":
        x = loadmat(path)
        if 'pts_2d' in x:
            # 'pts_2d' is in 300W_LP dataset
            landmark =  x['pts_2d'] # List
        elif 'pts' in x:
            # 'pts'    is in Multi_PIE dataset
            landmark =  x['pts']
        else:
            print("Unknown key to read in Mat file. Not implemented") 
        landmark = np.array(landmark)
    else:
        print("Unknown extension. Not implemented")

    return landmark.tolist()

def check_names_folders_annotations(names, folders, annotations):
    assert len(names) == len(folders)
    #assert len(folders) == len(annotations)

def get_data_for_json(names, folders, annotations, isValidation=False):
    """
        Gets the data to be written in the json format
        data is list of dictionaries.
    """ 
    check_names_folders_annotations(names, folders, annotations)
    num_datasets = len(names)

    if isValidation:
        print("\nValidation data prepartion started...")
    else:
        print("\nTraining data preparation started...")

    data = []

    # Iterate over all datasets
    for i in range(num_datasets):
        dataset_name = names[i]

        if not dataset_name:
            continue

        folder       = folders[i]

        if (annotations is None or i >= len(annotations)):
            annotation   = ""
        else:
            annotation   = annotations[i]

        img_files_grabbed = []
        for j in range(len(IMAGE_EXTENSIONS)):
            key = os.path.join(folder, "*"+IMAGE_EXTENSIONS[j])

            # We use sorted here to sort by name. This is useful when we 
            # images of a video sequence and we want to iteratively update
            # Reference https://stackoverflow.com/a/6774404
            img_files_grabbed.extend(sorted(glob.glob(key)))

        num_images = len(img_files_grabbed)

        if annotation:
            print("Dataset= {:10s}\tPath= {:50s} Annotations= {:60s} #Images= {}".format(dataset_name, folder, annotation, num_images))
        else:
            print("Dataset= {:10s}\tPath= {:50s} Annotations= {:60s} #Images= {}".format(dataset_name, folder, "None! Using Noisy Tight Bounding Boxes", num_images))

        # Iterate over all files
        for j in range(num_images):
            filename = img_files_grabbed[j]

            # Check for all possible landmark ground truth files
            found_flag = False
            for k in range(len(LANDMARK_GROUND_TRUTH_EXTENSIONS)):

                if LANDMARK_GROUND_TRUTH_EXTENSIONS[k] == ".mat":
                    if "300W_LP" in folder:
                        # 300W_LP dataset has landmarks in the format of "_pts.mat"
                        landmark_ground_truth_path = filename[:-4] + "_pts" + LANDMARK_GROUND_TRUTH_EXTENSIONS[k]
                    elif "multi_pie" in folder:
                        # Multi_PIE dataset has landmarks in the format of "_lm.mat"
                        landmark_ground_truth_path = filename[:-4] + "_lm"  + LANDMARK_GROUND_TRUTH_EXTENSIONS[k]
                    else:
                        pass
                else:
                    landmark_ground_truth_path = filename[:-4] +          LANDMARK_GROUND_TRUTH_EXTENSIONS[k]

                # Check if landmark ground truth is there in the same directory
                if os.path.exists(landmark_ground_truth_path):
                    found_flag = True
                    pts = read_landmark_ground_truth(landmark_ground_truth_path)
                    break

            if not found_flag:
                print("{} No landmark groundtruth found. Skipping".format(filename))
                continue

            # Check for bounding box annotations
            if annotation:
                objpos_det, scale_det, objpos_gd, scale_gd, _, _, width_height_det, width_height_gd = get_image_bounding_boxes_annotation_from_mat(filename, annotation)
            else:
                # Get annotations from landmark points adding noise to the boundary
                objpos_det, scale_det, objpos_gd, scale_gd, _, _, width_height_det, width_height_gd = get_image_bounding_boxes_from_landmarks(filename, pts, args.noise)

            # Insert the information of new image in variable data
            data.append({
                "isValidation"       : isValidation,
                "pts_paths"          : landmark_ground_truth_path,
                "objpos_det"         : objpos_det,
                "dataset"            : dataset_name,
                "scale_provided_det" : scale_det,
                "width_height_det"   : width_height_det,
                "objpos_grnd"        : objpos_gd,
                "scale_provided_grnd": scale_gd,
                "width_height_grnd"  : width_height_gd,
                "pts"                : pts,
                "img_paths"          : filename
            })
                        
    return data

def write_data_to_json(data, output_file_path):
    if len(data) == 0:
        return

    print("\nWriting to {}".format(output_file_path))
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile)


#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument    ('-i', '--input'  , help = 'path of the input config file', default='./splits_prep/config.txt')
ap.add_argument    ('-n', '--noise'  , help = 'noise to add to the ground truth bounding box if no bounding box given', type= float, default= 0.05)
args = ap.parse_args()

print("If no bounding box is given we add a noise of +/- {}%".format(args.noise*100))

# Parse the config file to get the directories and bounding box annotations
# https://stackoverflow.com/a/19379306
cp = config_parser.RawConfigParser()
cp.read(args.input)

input_folder_path    = cp.get('config', 'input_folder_path')
annotations_path     = cp.get('config', 'annotations_path')
num_keypoints        = cp.get('config', 'num_keypoints')

train_datasets_names = convert_string_to_list(cp.get('config', 'train_datasets_names'))
train_folders        = convert_string_to_list(cp.get('config', 'train_folders'))
train_annotations    = convert_string_to_list(cp.get('config', 'train_annotations'))

val_datasets_names   = convert_string_to_list(cp.get('config', 'val_datasets_names'))
val_folders          = convert_string_to_list(cp.get('config', 'val_folders'))
val_annotations      = convert_string_to_list(cp.get('config', 'val_annotations'))

output_json_folder   = cp.get('config', 'output_folder')
output_json_prefix   = cp.get('config', 'output_prefix')

print("Adding input folder path to the train and val relative dataset paths...")
for i in range(len(train_folders)):
    train_folders[i] = os.path.join(input_folder_path, train_folders[i])

for i in range(len(val_folders)):
    if val_folders[i]:
        val_folders[i]   = os.path.join(input_folder_path, val_folders[i])

print("Adding annotations path to the train and val relative annotations paths...")
for i in range(len(train_annotations)):
    # if any of the path is empty, that means we do not have ground truth annotations.
    if train_annotations[i]:
        train_annotations[i] = os.path.join(annotations_path, train_annotations[i])

for i in range(len(val_annotations)):
    # if any of the path is empty, that means we do not have ground truth annotations.
    if val_annotations[i]:
       val_annotations[i]   = os.path.join(annotations_path, val_annotations[i])

# Get train and val data
train_data = get_data_for_json(train_datasets_names, train_folders, train_annotations, isValidation=False)
val_data   = get_data_for_json(val_datasets_names  , val_folders  , val_annotations  , isValidation=True)

#ind = 2
#print(train_data[ind]["objpos_grnd"])
#print(train_data[ind]["scale_provided_grnd"])
#print(val_data[ind]["objpos_grnd"])
#print(val_data[ind]["scale_provided_grnd"])

# Write to json
write_data_to_json(train_data, os.path.join(output_json_folder, output_json_prefix + TRAIN_JSON))
write_data_to_json(val_data, os.path.join(output_json_folder, output_json_prefix + VAL_JSON))
