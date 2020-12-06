

"""
    Sample Run:
    python splits_prep/get_cofw_68_val_json.py

    Makes a JSON of cofw_dataset in the format of aflw_ours dataset.
    The externally occluded points are negative values of true locations.

    Only run this script after you have run
    python splits_prep/organize_cofw_68_test.py

    Download the images from
    http://www.vision.caltech.edu/xpburgos/ICCV13/Data/COFW_color.zip

    Download the 68 annotations from
    https://github.com/golnazghiasi/cofw68-benchmark/archive/master.zip

    Version 1 2019-11-09 Abhinav Kumar
"""
import os
import numpy as np
from hdf5storage import loadmat as hd5_loadmat
from scipy.io    import loadmat as sci_loadmat

from common_functions import *
from image_bounding_boxes import *

IMAGE_EXTENSIONS= [".png"]
LANDMARK_GROUND_TRUTH_EXTENSIONS = [".mat"]

images_mat_file    = "./bigdata1/zt53/data/face/COFW_color/COFW_test_color.mat"
annotations_folder = "/home/abhinav/Desktop/cofw68-benchmark/COFW68_Data"
output_folder      = "./bigdata1/zt53/data/face/cofw_68_organized/testset"
annotation = False
data       = []


if annotation:
    # Now load the bbox annotations
    det_box = sci_loadmat(os.path.join(annotations_folder, "cofw68_test_bboxes.mat"))['bboxes']
else:
    print("Using noisy boxes")


# Grab all the image files
print("Grabbing all the image")
image_files_grabbed = sorted(grab_files(output_folder, IMAGE_EXTENSIONS))
print("Done\n")


# Grab all the mat files
print("Grabbing all the 68 point annotations")
mat68_files_grabbed = sorted(grab_files(os.path.join(annotations_folder, "test_annotations"), LANDMARK_GROUND_TRUTH_EXTENSIONS))
print("Done\n")

print("Converting to aflw_ours JSON format - external occlusions as negative of true coordinates")

# Check if the images and the landmarks are exactly same in number
assert len(image_files_grabbed) == len(mat68_files_grabbed)
num_images = len(image_files_grabbed)


# Loop over all images
for i in range(num_images):
    key = str(i+1) 
    key_mat = key + "_points.mat"
    key_img = key + ".png"   
    filename = os.path.join(output_folder, key_img)

    index_mat = -1
    # search for the mat file in the mat68_files_grabbed
    for j in range(num_images):
        if os.path.basename(mat68_files_grabbed[j]) ==  key_mat:
            index_mat = j
            break

    if index_mat == -1:
        print("Not found for " + key_img)
    else:
        # This does not get loaded by hdf5storage loadmat      
        mat = sci_loadmat(mat68_files_grabbed[index_mat])
        pts = mat['Points']
        occlusion = mat['Occ'][0] # occlusion is 1 for external occlusions while it is 0 for normal points

        vis_mul = -1 * occlusion # -1 for external occlusions and 0 for normal points
        vis_mul [vis_mul == 0] = 1 # -1 for external occlusions and 1 for normal points
        vis_mul = vis_mul.astype(float)

        # Multiply pts coordinates by vis_mul
        # The externally occluded points will become negative numbers which is the
        # convention of aflw_ours dataset
        # Normal points will remain as it is
        pts = np.multiply(pts, vis_mul[:,np.newaxis])
        
        # Convert points to list since np array is not JSON serializable
        pts = pts.tolist()

        if annotation:
            pass
        else:
            get_image_bounding_boxes_from_landmarks(filename, pts, noise=0.05)
            # Get annotations from landmark points adding noise to the boundary
            objpos_det, scale_det, objpos_gd, scale_gd, _, _, width_height_det, width_height_gd = get_image_bounding_boxes_from_landmarks(filename, pts)

            # Insert the information of new image in variable data
            # This is going to be a validation set
            data.append({
                "isValidation"       : True,
                "pts_paths"          : "unknown.xyz",
                "objpos_det"         : objpos_det,
                "dataset"            : "cofw_68",
                "scale_provided_det" : scale_det,
                "width_height_det"   : width_height_det,
                "objpos_grnd"        : objpos_gd,
                "scale_provided_grnd": scale_gd,
                "width_height_grnd"  : width_height_gd,
                "pts"                : pts,
                "img_paths"          : filename
            })

# Write to json
write_data_to_json(data, os.path.join("dataset", "cofw_68_val.json"))
