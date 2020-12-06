

"""
    Sample Run:
    python plot/plot_uncertainties_in_transformed_space.py -i run_940_evaluate/300W_test
    python plot/plot_uncertainties_in_transformed_space.py -i run_50_evaluate/300W_test --laplacian

    Version 1 2019-08-14 Abhinav Kumar

    Visualizes the errors of the test set in transformed space by using uncertainties
    and predictions.
"""

import os, sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.linalg import fractional_matrix_power

from pylib.Cholesky import *

prefix = "test"
dpi = 200
alpha = 0.8
lw = 2
IMAGES_FOLDER = "images/transformed_space"
EXTENSION = ".png"

def sample_from_simplified_laplacian(num_points, laplacian= True):
    """
        The idea is to generate a discrete probability first and assign probabilities
        to each of the entries. Once that is done, we convert the 2D grid to 
        1D grid and sample indices based on the probability. Once we have the index
        we map them back to the original numbers.
        
        This is being done since Simplified Laplacian is not there in numpy
    """
    x = np.arange(-15,15,0.01)
    num = x.shape[0]
    prob = np.zeros((num,num))

    # Generate probability density first
    xvect = np.array(np.meshgrid(x, x)) # 2 x num x num
    dist  = np.power(xvect[0],2) + np.power(xvect[1],2) # num x num 

    if laplacian:
        prob = 1./(2*np.pi/3.0) * np.exp(-np.sqrt(3.0 * dist)) + 1e-16
    else:
        prob = 1./(2*np.pi)     * np.exp(-(dist))              + 1e-16

    prob = prob/np.sum(prob)
    prob = prob.flatten()

    t = np.arange(0,x.shape[0]**2)
    out = np.random.choice(t, num_points, p= prob)
    data = np.zeros((num_points, 2))

    # Map the index back
    for i in range(num_points):
        data[i,0] = x[out[i]//num]
        data[i,1] = x[out[i]%num]

    return data


def get_kld(p, q):
    epsilon = 0.0000001
    p = p + epsilon
    q = q + epsilon

    divergence = np.sum(p*np.log(p/q))
    return divergence

def get_our_data(folder):
    nme = np.load(os.path.join(folder,'nme_new_box_per_image_per_landmark.npy'))
    gt  = np.load(os.path.join(folder,'ground_truth.npy'))

    prd = np.load(os.path.join(folder,'means.npy'))
    L_vect = np.load(os.path.join(folder,'cholesky.npy'))

    covar, det_cov = cholesky_to_covar(L_vect)
    unc = np.concatenate(det_cov, axis=0)
    return gt, prd, covar, unc, nme

def plot_seaborn(x, y, xN, yN, save_name= "all", limit= (-10,10)):
    temp = np.stack([x, y], axis=1)
    df = pd.DataFrame(temp, columns=["x", "y"])

    temp1 = np.stack([xN, yN], axis= 1)
    df1 = pd.DataFrame(temp1, columns=["x", "y"])

    xedges = np.arange(-20, 20, 0.1)
    H,_,_ = np.histogram2d(x,  y , bins= xedges, normed= True) 
    H1,_,_ = np.histogram2d(x1, y1, bins= xedges, normed= True)
    div = get_kld(H1, H)
    #print(div)

    # References
    # https://seaborn.pydata.org/generated/seaborn.JointGrid.html
    # https://stackoverflow.com/questions/35920885/how-to-overlay-a-seaborn-jointplot-with-a-marginal-distribution-histogram-fr
    # https://stackoverflow.com/questions/55839345/seaborn-jointplot-group-colour-coding-for-both-scatter-and-density-plots
    sns.set(style="whitegrid")
    p = sns.JointGrid(x = df['x'], y = df['y'], xlim= limit, ylim= limit, height= 8)
    p.plot_joint(sns.scatterplot, color= "orange", s= 5, edgecolor= None)

    # Plot the normal first
    sns.kdeplot(df1['x'], alpha = alpha, color= "black",  linewidth= lw, ax= p.ax_marg_x, legend= False)
    sns.kdeplot(df1['y'], alpha = alpha, color= "black",  linewidth= lw, ax= p.ax_marg_y, legend= False, vertical= True)

    # Then plot our datapoints
    sns.kdeplot(df['x'] , alpha = alpha, color= "orange", linewidth= lw, ax= p.ax_marg_x, legend= False, shade=True)
    sns.kdeplot(df['y'] , alpha = alpha, color= "orange", linewidth= lw, ax= p.ax_marg_y, legend= False, vertical= True, shade=True)

    # Control the properties
    legend_properties = {'weight':'bold','size':8}
    p.ax_joint.legend(['Obtained'])
    p.ax_marg_x.set_ylim([0, 0.6])
    p.ax_marg_y.set_xlim([0, 0.6])
    p.ax_marg_x.legend(labels= ['Standard', 'Obtained'], prop=legend_properties, loc='upper right')
    p.ax_marg_y.legend(labels= ['Standard', 'Obtained'], prop=legend_properties, loc='lower right')
    p.ax_joint.set_xticks(np.arange(limit[0]+1, limit[1]+1, 3))
    p.ax_joint.set_yticks(np.arange(limit[0]+1, limit[1]+1, 3))
    plt.text(limit[0]+1, limit[1]-1, r'$D_{KL}$' + "(Normal, Obtained) = " + str(round(div, 2)), fontsize= 12)
    #plt.show()

    path = os.path.join(IMAGES_FOLDER, save_name + EXTENSION)
    print("Saving to {}".format(path))
    plt.savefig(path)
    plt.close()


#===============================================================================
# Argument Parsing
#===============================================================================
ap     = argparse.ArgumentParser()
ap.add_argument('-i', '--exp_id', default= 'run_940_evaluate/300W_test', help= 'path of the input folder')
ap.add_argument('-p', '--prefix', default= 'test', help= 'prefix to the saved images')
ap.add_argument(      '--laplacian', action = 'store_true'      , help= 'use laplacian likelihood instead of Gaussian')
args   = ap.parse_args()
input_folder = args.exp_id


folder= os.path.join("abhinav_model_dir", input_folder)
gt, prd, covar, _, nme = get_our_data(folder)
print("Loading done.")

num_images = gt.shape[0]
num_points = gt.shape[1]

transformed_pts = np.zeros(prd.shape)
for i in range(num_images):
    for j in range(num_points):
		# Ground - prediction as in paper
        transformed_pts[i,j] = fractional_matrix_power(covar[i,j], -0.5).dot(gt[i,j]-prd[i,j])
print("Transformation done.")

# Add some points from the standard 2D bivariate normal/ standard 2D simplified 
# laplacian
num_points_new = 10000
if args.laplacian:
    print("Getting standard Laplacian data from our function...")
    data = sample_from_simplified_laplacian(num_points_new, True)
    x1 = data[:,0]
    y1 = data[:,1]
    limit = (-10, 10)
else:
    print("Getting standard Gaussian data from numpy inbuilt function...")
    x1, y1 = np.random.multivariate_normal([0, 0], [[1, 0],[0, 1]], num_points_new).T
    limit = (-10, 10)

# Plot the individual landmarks first
#for i in range(num_points):
#    x = transformed_pts[:, i, 0]
#    y = transformed_pts[:, i, 1]
#    plot_seaborn(x, y, x1, y1, prefix + "_landmark_"+ str(i+1))

# Now plot all the points
x = transformed_pts[:,:,0].flatten()
y = transformed_pts[:,:,1].flatten()
plot_seaborn(x, y, x1, y1, prefix + "_all", limit)
