

"""
    Helper functions to get image bounding box annotations either by reading
    the mat file or adding noise to the tightest bounding box

    Version 3 Abhinav Kumar 2019-06-23 Noisy bounding boxes scaled on tight bounding boxes instead of original image
    Version 2 Abhinav Kumar 2019-06-14 Support for ground truth bounding boxes from landmark points
    Version 1 Abhinav Kumar 2019-05-27
"""

import os
import numpy as np
from scipy.io import loadmat
from PIL import Image


def get_name(data, index):

    return data["bounding_boxes"][0][index][0][0][0][0]

def get_bounding_box_detected(data, index):

    return data["bounding_boxes"][0][index][0][0][1][0]

def get_bounding_box_ground(data, index):

    return data["bounding_boxes"][0][index][0][0][2][0]

def get_center(bounding_box):
    """
        Gets center of the bounding box from the bounding box coordinates
        bounding_box = (xmin, ymin, xmax, ymax)

        Reference- 
        https://ibug.doc.ic.ac.uk/media/uploads/images/300-w/figure_6.png
        https://ibug.doc.ic.ac.uk/resources/300-W/
    """

    xmin, ymin, xmax, ymax = bounding_box

    return 0.5*(xmin+xmax), 0.5*(ymin+ymax)

def get_scale(bounding_box):
    """
        Gets scale of the bounding box from the bounding box coordinates
        bounding_box = (xmin, ymin, xmax, ymax)

        The formula was found by running the regression on the scale_pred values
        on the bounding_box coordinates. Refer fit_scale_from_bounding_box.log
        for more details.

        Reference- 
        https://ibug.doc.ic.ac.uk/media/uploads/images/300-w/figure_6.png
        https://ibug.doc.ic.ac.uk/resources/300-W/
    """
    xmin, ymin, xmax, ymax = bounding_box
    diffx    = xmax - xmin
    diffy    = ymax - ymin
    scale    = 0.005 * np.max([diffx, diffy])

    return scale

def get_width_height(bounding_box):
    """
        Gets width and height of the bounding box coordinates.
        bounding_box = (xmin, ymin, xmax, ymax)
    """ 
    xmin, ymin, xmax, ymax = bounding_box
    width    = xmax - xmin
    height   = ymax - ymin

    return [width, height]

def get_center_scale_width_height(bounding_box):
    """ 
        Gets the center, scale, width and height of the bounding box
    """
    xcenter, ycenter = get_center(bounding_box)
    scale            = get_scale(bounding_box)
    width_height     = get_width_height(bounding_box)

    objpos           = [xcenter, ycenter]

    return objpos, scale, width_height

def get_image_bounding_boxes_annotation_from_mat(filename, annotations_file):
    """
        Gets center and scales of all the bounding boxes - predicted and ground
        
        Reference-
        https://scipy-cookbook.readthedocs.io/items/Reading_mat_files.html
    """

    x = loadmat(annotations_file)
    num_data_points = len(x["bounding_boxes"][0])
    key = os.path.basename(filename)[:-4] # remove the extension

    xcenter_d = -1
    ycenter_d = -1
    xcenter_g = -1
    ycenter_g = -1
    scale_d   = -1
    scale_g   = -1

    for i in range(num_data_points):
        name = get_name(x, i)
        name = name[:-4] # remove the extension

        if (name == key):
            # Getting bounding box predicted
            bounding_box_p                    = get_bounding_box_detected(x, i)
            objpos_p, scale_p, width_height_p = get_center_scale_width_height(bounding_box_p)

            # Getting bounding box ground
            bounding_box_g                    = get_bounding_box_ground(x, i)
            objpos_g, scale_g, width_height_g = get_center_scale_width_height(bounding_box_g)

            break

    return objpos_p, scale_p, objpos_g, scale_g, bounding_box_p, bounding_box_g, width_height_p, width_height_g

def get_noisy_bounding_box(image, tight_bounding_box, noise):
    """ 
        tight_bounding_box = [xmin, ymin, xmax, ymax]
        image              = PIL image

        Generates the image bounding boxes by adding noise to the extreme points
        similar to the following paper where authors add 10% of noise to the
        ground truth bounding boxes. The 10% refers to the scale, therefore
        sqrt(h*w) of the GT bboxes and the perturbations are drawn from the same
        distribution uniformly. Xmin, Xmax, Ymin and Ymax, were perturbed up to 
        5%, adding up to a total of 10% if they are perturbed in different
        directions.

        Convention of x and y axis
        -----> X
        |
        |
        V Y
        bounding_box = (xmin, ymin, xmax, ymax)

        Reference-
        How far are we from solving the 2D & 3D Face Alignment problem? (and a
        dataset of 230,000 3D facial landmarks) - Bulat and Tzimiropoulos
        ICCV 2017
        http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf
    """
    width, height = image.size

    xmin, ymin, xmax, ymax = tight_bounding_box

    # Get scale of the ground truth bounding box
    width_box  = xmax - xmin
    height_box = ymax - ymin
    scale = np.sqrt(width_box* height_box)
    
    # Add noise to the the ground truth
    # Tim said to add in both directions rather than in one direction. Padding 
    # can take care of the things
    # NME is independent of the constant scaling since d takes care of that.
    xmin_new = xmin + np.random.uniform(-noise * scale, noise * scale)
    ymin_new = ymin + np.random.uniform(-noise * scale, noise * scale)

    xmax_new = xmax + np.random.uniform(-noise * scale, noise * scale)
    ymax_new = ymax + np.random.uniform(-noise * scale, noise * scale)

    if xmin_new < 1:
        xmin_new = 1
    if ymin_new < 1:
        ymin_new = 1
    if xmax_new > width:
        xmax_new = width
    if ymax_new > height:
        ymax_new = height

    bounding_box = [xmin_new, ymin_new, xmax_new, ymax_new]

    return bounding_box

def get_image_bounding_boxes_from_landmarks(filename, pts, noise=0.05):
    """
        Generates the detected and ground truth image bounding boxes by adding
        noise to the tightest bounding box formed by the landmark points.
        5 % noise to both sums to 10% in total.
    """
    image = Image.open(filename)

    # pts is list of num_points x 2
    pts = np.array(pts)

    # An abs is needed since some points in aflw_ours dataset are correct but 
    # are externally occluded and therefore labelled negative
    pts = np.abs(pts)
    
    # There would be -1 as well and if not corrected they would result in minimum
    # being 1. Find a valid coordinate first
    valid_found = False    
    for i in range(pts.shape[0]):
        if pts[i, 0] == 1 and pts[i, 1] == 1:
            pass
        else:
            valid_found = True
            valid = pts[i]
            break

    # Replace the illegal coordinate with valid ones
    if valid_found:
        for i in range(pts.shape[0]):
            if pts[i, 0] == 1 and pts[i, 1] == 1:
                pts[i] = valid
    else:
        print("All coordinates are invalid...")


    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)

    tight_bounding_box = [xmin, ymin, xmax, ymax]

    # Getting bounding box detected
    bounding_box_p                    = get_noisy_bounding_box(image, tight_bounding_box, noise)
    objpos_p, scale_p, width_height_p = get_center_scale_width_height(bounding_box_p)

    # Getting bounding box ground
    bounding_box_g                    = get_noisy_bounding_box(image, tight_bounding_box, noise)
    objpos_g, scale_g, width_height_g = get_center_scale_width_height(bounding_box_g)

    return objpos_p, scale_p, objpos_g, scale_g, bounding_box_p, bounding_box_g, width_height_p, width_height_g
