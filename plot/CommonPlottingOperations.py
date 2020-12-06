

"""
    Plotting library

    Version 2 2019-07-17 Abhinav Kumar More ways to calculate variances added
    Version 1 2019-06-25 Abhinav Kumar
"""

import os
import numpy as np
from scipy.linalg import logm, expm
from numpy.linalg import inv
from scipy.stats import pearsonr

import plotting_params as params
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

images_file       = "images.npy"
heatmaps_file     = "heatmaps.npy"
ground_truth_file = "ground_truth.npy"
means_file        = "means.npy"
cholesky_file     = "cholesky.npy"
vis_gt_file       = "vis_gt.npy"
vis_estimated_file= "vis_estimated.npy"


IMAGE_DIR         = "images"
DPI   = 200
alpha = 0.9
ms    = 2
lw    = 1.5

def get_log(input):
    temp = np.zeros(input.shape)
    ndim = input.ndim
    
    if ndim == 4:
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                temp[i, j]= logm(input[i,j])
    elif ndim == 3:
        for i in range(input.shape[0]):
            temp[i] = logm(input[i])

    return temp

def get_exp(input):
    output = np.zeros(input.shape)
    ndim = input.ndim
    
    if ndim == 4:
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                output[i, j]= expm(input[i,j])
    elif ndim == 3:
        for i in range(input.shape[0]):
            output[i]= expm(input[i])

    return output

def get_jbld(input):
    """
        Calculates the mean of covariances based on the Jensen-Bregman LogDet 
        Divergence. Equation (25) of the paper is implemented
        Jensen-Bregman LogDet Divergence with Application to Efficient 
        Similarity Search for Covariance Matrices - Cherian, Sra, Banerjee and
        Papanikolopoulos, TPAMI 2012
    """
    num_points = input.shape[1]
    output     = np.zeros((num_points, 2, 2))
    num_iterations = 30

    for i in range(num_points):
        data = input[:, i]
        center = np.array([[1, 0],[0, 1.]])
        
        for j in range(num_iterations):
            center_new = inv(np.mean(inv( (data + center)/2.), axis= 0))
            #print(np.linalg.norm(center_new- center, ord='fro'))
            center     = center_new

        output[i] = center

    return output

def plot_mean_and_ellipse( cov, mu_x, mu_y, n, nstd, ax, color1= "red", color2= "blue", markersize= 2, linewidth= 1.5, scalar_map= None, markeredgecolor= None, zorder= 10, marker= 'o', markeredgewidth= 0.5, alpha= 1):
    """
        Plots an ellipse with the covariance matrix. The data is entered in 
        matplotlib coordinate frame.

        ^ Y
        |
        |
        |----> X
    """
    if alpha is None:
        alpha = 1
    if mu_x is not None and mu_y is not None:
        plt.plot(mu_x, mu_y, marker= marker, color= color1, markersize= markersize, markeredgewidth= markeredgewidth, markeredgecolor= markeredgecolor, zorder= zorder, alpha= alpha)
    
    if cov is not None:
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals  = vals[order]
        vecs  = vecs[:, order]

        theta = np.degrees(np.arctan(vecs[1, 0] / vecs[0, 0]))
        # width and height of the ellipse.
        # width and height are twice the size of the eigen-values.
        # References - 
        # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.patches.Ellipse.html 
        # https://matplotlib.org/3.1.0/gallery/units/ellipse_with_units.html#sphx-glr-gallery-units-ellipse-with-units-py 
        #
        # Can be checked by plotting mu_x = 0, mu_y = 0 and cov = [[1,0],[0,1]].
        # The output should be x^2 + y^2 = 1 
        # If we do not multiply by 2, the plot generated is x^2 + y^2 = 0.5
        w, h  = 2 * nstd * np.sqrt(vals)

        area  = np.sqrt(vals[0]*vals[1])
        if scalar_map is not None:
            color_ellipse = scalar_map.to_rgba(area)
        else:
            color_ellipse = color2

        for i in range(1, n):
            ell = Ellipse(xy= (mu_x, mu_y), width= w*i, height= h*i, angle= theta, color= color_ellipse, linewidth= linewidth, alpha= alpha)
            ell.set_facecolor('none')
            ell.set_zorder(zorder-3)
            ax.add_artist(ell)

def normalize_input(input, d, use_method= "euclidean"):
    """
        Normalizes the variable input by the variable d
        input = num_images x 68 x 2 numpy (means) or 
                num_images x 68 x 2 x 2 numpy (covariances)
        d     = num_images numpy

        returns 68 x 2 or 68 x 2 x 2 numpy
    """
    if input.ndim == 3:
        # Mean normalized by d
        temp = np.repeat(d[:, np.newaxis], input.shape[1] * input.shape[2], axis=1)
        d    = temp.reshape((input.shape[0], input.shape[1], input.shape[2]))
    elif input.ndim == 4:
        # Covariances normalized by d as well
        temp = np.repeat(d[:, np.newaxis], input.shape[1] * input.shape[2] * input.shape[3], axis=1)
        d    = temp.reshape((input.shape[0], input.shape[1], input.shape[2], input.shape[3]))
    else:
        pass

    input = input/d
    input_2 = input

    if use_method == "log-exponential":
        print("Taking log mean exponential")
        input = get_log(input)
        output = np.mean(input, axis=0)
        output = get_exp(output)
    elif use_method == "euclidean":
        output = np.mean(input, axis=0)
    elif use_method == "jbld":
        output = get_jbld(input)
        output = np.mean(input, axis=0)
    else:
        print("Some unknown method to use")

    return output, input_2


def load(numpy_file_path, prefix):
    """
        Loads a single numpy file
    """
    print("Loading {} numpy file= {}".format(prefix, numpy_file_path))
    data = np.load(numpy_file_path)
    print("Done")
    print(data.shape)
    print("")

    return data

def load_all(folder_path, load_heatmaps= False, load_cholesky= True, load_vis= False, load_images= True):
    """
        Loads a whole bunch of numpy files depending on what options we want
    """
    output = []

    if load_images:
        images       = load(os.path.join(folder_path, images_file)  , "images")
        output.append(images)

    if load_heatmaps:
        heatmaps     = load(os.path.join(folder_path, heatmaps_file), "heatmaps")
        output.append(heatmaps)

    ground_truth = load(os.path.join(folder_path, ground_truth_file), "ground_truth")
    output.append(ground_truth)

    means        = load(os.path.join(folder_path, means_file),  "means")
    output.append(means)

    if load_cholesky:
        cholesky        = load(os.path.join(folder_path, cholesky_file),  "cholesky")
        output.append(cholesky)
        
    if load_vis:
        vis_gt          = load(os.path.join(folder_path, vis_gt_file)       , "vis_gt")
        vis_estimated   = load(os.path.join(folder_path, vis_estimated_file), "vis_estimated")
        
        output.append(vis_gt)
        output.append(vis_estimated)

    return output

def swap_channels(input):
    return np.transpose(input, (1, 2, 0))

def plt_pts(plt, pts, color= 'limegreen', marker= True, text= None, shift= 1, text_color= "blue", ms= 5, ignore_text= False):
    for i in range(pts.shape[0]):
        x = pts[i,0]
        y = pts[i,1]
        plt.plot(x, y, marker='o', ms= ms, color= color)
        if marker:
            if ignore_text:
                pass
            else:
                if text is not None:
                    plt.text(x+ shift, y+ shift, str(text[i]), color= text_color, fontsize= 9)
                else:
                    plt.text(x+ shift, y+ shift, str(i), color= text_color, fontsize= 9)

def plt_pair_of_pts(ax, pts, gt, index, color= "red", linewidth= 2, zorder= 2):
    """
        Plots pair of two points for an image
    """
    ix1 = index[0]
    ix2 = index[1]
    
    if gt[ix1,0] <0  or gt[ix1,1] < 0 or gt[ix2, 0] < 0 or gt[ix2, 1] < 1:
        return
    else:
        x1 = pts[ix1, 0]
        x2 = pts[ix2, 0]

        y1 = pts[ix1, 1]
        y2 = pts[ix2, 1]
        ax.plot([x1, x2], [y1, y2], color= color,linewidth= linewidth, zorder= zorder) 

def plot_cuc(nme_per_img):# similar to figure 8 in ICCV'17
    order_nme = np.sort(nme_per_img)
    n_img = len(order_nme)
    x = order_nme
    y = np.arange(1, n_img+1)*1.0/n_img

    return x, y

def compute_NME(pts_prd, pts_gt, bbox_d):
    nme = np.linalg.norm(pts_prd - pts_gt)
    nme = np.mean(nme)/bbox_d
    return nme

def compute_bboxd(pts_gt):
    scale = np.sqrt((np.max(pts_gt[:,1])-np.min(pts_gt[:,1]))*(np.max(pts_gt[:,0])-np.min(pts_gt[:,0]))) 
    return scale

def compute_scale(pts_gt):
    scale = np.zeros((pts_gt.shape[0], ))
    for i in range(scale.shape[0]):
        scale[i] = compute_bboxd(pts_gt[i])
    return scale

def get_err(prd, gt):
    err = []
    for batch_idx in range(prd.shape[0]):
        bboxd = compute_bboxd(gt[batch_idx])
        for i in range(68):
            err.append(compute_NME(prd[batch_idx][i], gt[batch_idx][i], bboxd))
    return np.array(err)
    
def search_json(data, key):
    num_images= len(data)

    for i in range(num_images):
        if os.path.basename(data[i]["img_paths"]) == os.path.basename(key):
            return i

    return -1

def plot_image(ax, images, pts_list_1, means, covar, ground_truth, i, indices, vis_gt= None, vis_estimated= None, use_different_color_for_ext_occ= True, use_threshold_on_vis= False, threshold_on_vis= 0.65):
    """
        Plots an image and then also plots the landmarks ground truth, means and
        covar depending on the options given. Can also save a blank image with no
        landmarks.

        use_different_color_for_ext_occlusion = Allows to go for different colors for showing ext occlusions
        use_threshold_on_vis = If this option is true, it does not show points which have predicted visibilities < 0.65
    """
    image = swap_channels(images[indices[i]])
    ax.imshow(image, cmap='jet', vmin= 0, vmax= 1, alpha= params.alpha)

    if pts_list_1 is not None:
        num_points = pts_list_1.shape[0]
        for j in range(num_points):
            if means is not None:
                mx  = means[indices[i], pts_list_1[j], 0]
                my  = means[indices[i], pts_list_1[j], 1]
            else:
                mx  = None
                my  = None
            
            if covar is not None:
                cov = covar[indices[i], pts_list_1[j]]
            else:
                cov = None

            if vis_estimated is None:
                alpha_landmark = 1
            else:
                alpha_landmark = vis_estimated[indices[i], pts_list_1[j]]
                if use_threshold_on_vis:
                    if alpha_landmark < threshold_on_vis:
                        alpha_landmark = 0

            # Plot predictions - mean and covariance
            plot_mean_and_ellipse(cov, mx, my, n= 2, nstd= 1, ax= ax, color1= params.mean_color, color2= params.covar_color, linewidth= params.lw, markersize= params.ms, alpha= alpha_landmark)
            
            if ground_truth is not None:
                gt_x = ground_truth[indices[i], pts_list_1[j], 0]
                gt_y = ground_truth[indices[i], pts_list_1[j], 1]
                
                if vis_gt is None:
                    landmark_vis_gt = 1
                else:
                    landmark_vis_gt = int(vis_gt[indices[i], pts_list_1[j]])
                    
                # Plot according to the type of landmark
                if  landmark_vis_gt == 1:
                    # Normal landmark 
                    ax.plot(gt_x, gt_y, color= params.unoccluded_color, marker= 'o', markersize= params.ms)
                elif landmark_vis_gt == 2:
                    # External Occluded
                    if use_different_color_for_ext_occ:
                        color_name = params.ext_occluded_color
                    else:
                        color_name = params.unoccluded_color
                    ax.plot(gt_x, gt_y, color= color_name             , marker= 'o', markersize= params.ms)
                else:
                    pass
                    # Self Occluded
                    # ax.plot(gt_x, gt_y, color= 'black'  , marker='o', markersize= params.ms)

def plot_multiple_images(rel_paths, images, pts_list_row, pts_list, means, covar, ground_truth, indices, selected_indices, prefix= "processed_", save_folder= IMAGE_DIR, vis_gt= None, vis_estimated= None, use_different_color_for_ext_occ= True):
    """
        indices specifies the index relative to the json
        selected_indices specifies which among the indices are to be plotted
        
        pts_list     = consists of list of points to be plotted
        pts_list_row = consists of mapping to each image
        
        Suppose we want to show different set of points on different faces.
        row 0 --> [0,1,5, 10, 23]  for images 0,3
        row 1 --> [2,20,50, 67,89] for images 1,2 
        
        Then, pts_list_row will be [0,1,1,0] 
    
    """
    
    for i in range(selected_indices.shape[0]):
        if indices[selected_indices[i]] > 0:
            # Start plotting
            fig= plt.figure(dpi= params.DPI)
            ax = fig.add_subplot(111)

            if pts_list is not None:
                pts_list_1 = pts_list[pts_list_row[selected_indices[i]]]
                plot_image(ax, images, pts_list_1, means, covar, ground_truth, selected_indices[i], indices, vis_gt, vis_estimated, use_different_color_for_ext_occ= use_different_color_for_ext_occ)
            else:
                plot_image(ax, images, None      , means, covar, ground_truth, selected_indices[i], indices, vis_gt, vis_estimated, use_different_color_for_ext_occ= use_different_color_for_ext_occ)

            full_folder_path = os.path.join(save_folder, os.path.dirname(prefix))
            makedir(full_folder_path, show_message= False)
            save_path = os.path.join(save_folder, prefix + os.path.basename(rel_paths[selected_indices[i]]))
            # No margin around the border            
            # Reference https://stackoverflow.com/a/50512838
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            savefig(plt, save_path, show_message= True, tight_flag= True, newline= False)
            plt.close()

def plot_single_scatter_with_corr_for_one_model(x, y, marker, marker_color, plt, legend_text= None, frac= 1, size= 2, alpha= 1):
    # Calculate Pearson from all data
    pearson_image = pearsonr(x, y)
    print("{:25s} Pearson = {:.2f}".format(legend_text, pearson_image[0]))

    # Plot only a fraction of points
    num_points = x.shape[0]
    index = np.random.choice(range(num_points), int(frac * num_points), replace=False)
    plt.scatter(x[index], y[index], c= marker_color, marker= marker, edgecolors= 'none', s= size, alpha= alpha, label= legend_text + ', Correlation= ' + str(round(pearson_image[0], 2)))

def plot_single_scatter_with_corr_for_multi_model(xdata_list, ydata_list, color_list, label_list, marker_list, xlim= None, ylim= None, path= "", xlabel= None, ylabel= None, frac= 1, figsize= (9.6,6), dpi= 200, size= 2, alpha= 1):
    plt.figure(figsize= figsize, dpi= dpi)
    ax = plt.gca()

    plt.grid(True)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    for i in range(len(xdata_list)):
        plot_single_scatter_with_corr_for_one_model(xdata_list[i], ydata_list[i], marker_list[i], color_list[i], plt, legend_text= label_list[i], frac= frac, size= size, alpha= alpha)
    plt.legend(loc= 'upper right')

    print("Plotting {}% of data".format(frac*100.))
    savefig(plt, path, tight_flag= True)
    plt.close()

def savefig(plt, path, show_message= True, tight_flag= False, newline= True):
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(path)
    if newline:
        print("")

def makedir(full_folder_path, show_message= True): 
    if not os.path.exists(full_folder_path):
        print("Making directory {}".format(full_folder_path))
        os.makedirs(full_folder_path)
    else:
        if show_message:
            print("{} present".format(full_folder_path))

import unicodedata as UD

DIGIT = {
    'MINUS': u'-',
    'ZERO': u'0',
    'ONE': u'1',
    'TWO': u'2',
    'THREE': u'3',
    'FOUR': u'4',
    'FIVE': u'5',
    'SIX': u'6',
    'SEVEN': u'7',
    'EIGHT': u'8',
    'NINE': u'9',
    'STOP': u'.'
    }

def guess(unistr):
    return ''.join([value for u in unistr
                    for key,value in DIGIT.iteritems()
                    if key in UD.name(u)])
