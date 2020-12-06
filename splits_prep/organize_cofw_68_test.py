

"""
    Sample Run:
    python splits_prep/organize_cofw_68_test.py

    Copies the images from the cofw-68 dataset to the desired location to get the images.

    Download the images from
    http://www.vision.caltech.edu/xpburgos/ICCV13/Data/COFW_color.zip

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
output_folder      = "./bigdata1/zt53/data/face/cofw_68_organized/testset"

if os.path.exists(output_folder):
    print("Output directory exists")
else:
    print("Creating directory {}".format(output_folder))
    os.makedirs(output_folder)

# Load the images mat file
print("Loading all test images from COFW-29 mat...")
mat = hd5_loadmat(images_mat_file)
print("Done\n")
images = mat['IsT']
pts_29 = mat['phisT']

num_images = len(mat68_files_grabbed)

# Now first save those images in the output_folder
for i in range(num_images):
    img = images[i][0]

    if len(img.shape) == 2:
        # convert grayscale to color
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.repeat  (img, 3, axis=2)

    filename = os.path.join(output_folder, str(i+1) + ".png")
    # cv2 assumes images are in BGR format but our images are in RGB
    # convert images to BGR so that cv2 saves correctly 
    img = img[:,:,::-1]
    cv2.imwrite(filename, img)

    if i% 50 == 0:
        print("{} images done".format(i))
