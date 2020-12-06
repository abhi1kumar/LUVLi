
"""
   This function fits the bounding box data to the 4 coordinates in the mat file
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
import json

sys.path.insert(0, './src')
from read_mat_file import get_image_bounding_boxes_annotation

LANDMARK_DELIMITER              = " "
LANDMARK_HEADERS                = 3
LANDMARK_FOOTERS                = 1
LANDMARK_GROUND_TRUTH_EXTENSION = ".pts"
IMAGE_EXTENSIONS                = [".png", ".jpg"]
TRAIN_JSON                      = "train.json"
VAL_JSON                        = "val.json"


def convert_string_to_list(str):
    return [x.strip() for x in str.split(',')]


def read_landmark_ground_truth(path):
    """
       Reads the landmark groundtruth from a file specified
    """
    landmark = np.genfromtxt(path, skip_header=LANDMARK_HEADERS, skip_footer=LANDMARK_FOOTERS, delimiter=LANDMARK_DELIMITER)
    return landmark.tolist()


def check_names_folders_annotations(names, folders, annotations):
    assert len(names) == len(folders)
    assert len(folders) == len(annotations)


def get_data_for_json(names, folders, annotations, isValidation=False):
    """
       Gets the data which is to be written in the json format
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
        folder       = folders[i]
        annotation   = annotations[i]

        img_files_grabbed = []
        for j in range(len(IMAGE_EXTENSIONS)):
            key = os.path.join(folder, "*"+IMAGE_EXTENSIONS[j])
            img_files_grabbed.extend(glob.glob(key))

        num_images = len(img_files_grabbed)
        print("Dataset= {:10s}\tPath= {:50s} Annotations= {:60s} #Images= {}".format(dataset_name, folder, annotation, num_images))

        # Iterate over all files
        for j in range(num_images):
            filename = img_files_grabbed[j]

            landmark_ground_truth_path = filename[:-4] + LANDMARK_GROUND_TRUTH_EXTENSION

            # Check if landmark ground truth is there in the same directory
            if os.path.exists(landmark_ground_truth_path):
                pts = read_landmark_ground_truth(landmark_ground_truth_path)
            else:
                print("{} No landmark groundtruth found. Skipping".format(filename))
                continue

            # Check for bounding box annotations
            objpos_det, scale_det, objpos_gd, scale_gd, bb_det, bb_gd = get_image_bounding_boxes_annotation(filename, annotation)

            # Insert the information of new image in variable data
            data.append({
                "isValidation"       : isValidation,
                "pts_paths"          : landmark_ground_truth_path,
                "objpos_det"         : objpos_det, 
                "dataset"            : dataset_name, 
                "scale_provided_det" : scale_det, 
                "objpos_grnd"        : objpos_gd, 
                "scale_provided_grnd": scale_gd, 
                "pts"                : pts,
                "img_paths"          : filename,
                "bb_det"             : bb_det,
                "bb_gd"              : bb_gd
            })

    return data


def fit_data(X, Y):
    print("Fitting started...")
    from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

    print("Least Square")
    coef = np.linalg.lstsq(X, Y, rcond=None)[0]
    print(coef)
    #print(intercept)

    print("Ridge")
    #ridgecv = RidgeCV(alphas = None, cv = 10)
    #ridgecv.fit(X, Y)

    rr100 = Ridge(alpha=100) 
    rr100.fit(X, Y)
    print(rr100.coef_)
    print(rr100.intercept_)


    print("LASSO")
    lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000)
    lassocv.fit(X, Y)

    rr100 = Lasso(max_iter=10000) #  comparison with alpha value
    rr100.set_params(alpha=lassocv.alpha_)
    rr100.fit(X, Y)
    print(rr100.coef_)
    print(rr100.intercept_)
    print("")


def prepare_and_fit_data(data_to_use):
    num_samples = len(data_to_use)

    print("\nFitting preparation started...")
    # Assuming X and Y are indepenedent and dependent variables respectively
    X = np.zeros((num_samples, 7))
    Y = np.zeros((num_samples,))

    with open('./dataset/face.json') as json_file:
        data = json.load(json_file)

    # data is a list of dictionaries
    num_entrires_in_orig_json = len(data)

    for j in range(num_samples):
        scale   = data_to_use[j]["scale_provided_det"]
        name1   = data_to_use[j]["img_paths"]
        key_ds  = data_to_use[j]["dataset"]
        key     = os.path.basename(name1)
        key_val = data_to_use[j]["isValidation"]

        for i in range(num_entrires_in_orig_json):
            objpos_det = np.array(data[i]["objpos_det"])
            pts        = np.array(data[i]["pts"])
            curr_scale = np.array(data[i]["scale_provided_det"])

            name2      = data[i]["img_paths"]
            curr_name  = os.path.basename(name2)
            curr_ds    = data[i]["dataset"]
            curr_val   = data[i]["isValidation"]
            #print(curr_ds)

            if (key == curr_name and key_ds == curr_ds and key_val == curr_val):
                #print("Computed= {:6f} name1= {:60s} Ground_truth= {:6f} name2= {:60s}".format(scale, name1, curr_scale, name2))
                temp = data_to_use[j]["bb_det"]
                X[j,:4]  = temp
                diffx    = temp[2]-temp[0]
                diffy    = temp[3]-temp[1]
                X[j,4:6] = np.array([diffx, diffy])
                X[j, 6]  = np.max([diffx, diffy])
                Y[j]     = curr_scale
                break

    print("Using all features")
    fit_data(X     , Y)

    print("Using diff features")
    fit_data(X[:,4:6], Y)

    print("Using diff features and max")
    fit_data(X[:,4:7], Y)

    print("Using only max")
    fit_data(X[:,6:7], Y)


#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument    ('-i', '--input'  , help = 'path of the input config file', default='./splits_prep/config.txt')
args = ap.parse_args()

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
    val_folders[i]   = os.path.join(input_folder_path, val_folders[i])


print("Adding annotations path to the train and val relative annotations paths...")
for i in range(len(train_folders)):
    train_annotations[i] = os.path.join(annotations_path, train_annotations[i])

for i in range(len(val_folders)):
    val_annotations[i]   = os.path.join(annotations_path, val_annotations[i])


# Get train data and write to json
train_data = get_data_for_json(train_datasets_names, train_folders, train_annotations, isValidation=False)
val_data   = get_data_for_json(val_datasets_names  , val_folders  , val_annotations  , isValidation=True)

# Fit data
prepare_and_fit_data(train_data)

# Check fit to some of the data points to see if the formulation is indeed 
# correct
# The formula has already been plugged in
with open('./dataset/face.json') as json_file:
    data = json.load(json_file)

# data is a list of dictionaries
num_samples = len(data)

for j in range(10):
    scale   = val_data[j]["scale_provided_det"]
    name1   = val_data[j]["img_paths"]
    key_ds  = val_data[j]["dataset"]
    key     = os.path.basename(name1)
    key_val = val_data[j]["isValidation"]

    for i in range(num_samples):
        objpos_det = np.array(data[i]["objpos_det"])
        pts        = np.array(data[i]["pts"])
        curr_scale = np.array(data[i]["scale_provided_det"])

        name2      = data[i]["img_paths"]
        curr_name  = os.path.basename(name2)
        curr_ds    = data[i]["dataset"]
        curr_val   = data[i]["isValidation"]

        if (key == curr_name and key_ds == curr_ds and key_val == curr_val):
            print("name1= {:60s} name2= {:60s} Ground_truth= {:6f} Computed= {:6f}\tDiff= {:.6f}".format(name1, name2, curr_scale, scale, np.abs(curr_scale-scale)))
            break
