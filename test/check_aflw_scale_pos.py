

"""
    Script to check which version of the mean is used to get the scale
    Conclusion - There is no 5% noise addition and then taking the geometric mean
"""
import numpy as np
from numpy import genfromtxt

def get_scale(pts_data):
	width  = np.max(pts_data[:,0]) - np.min(pts_data[:,0])
	height = np.max(pts_data[:,1]) - np.min(pts_data[:,1])
	scale_est = 0.005 * np.max([width, height])

	return scale_est, width, height

# This file is obtained by removing the filenames and 
# first header row from the aflw files.
path = "/home/abhinav/Desktop/data/aflw/test_coords.csv"
data = genfromtxt(path, delimiter=',')
print(data.shape)

# Order
# scale,box_size,center_w,center_h
pts_data = data[:,4:]
meta     = data[:,:4]
pts_data = pts_data.reshape((-1, 19, 2))

for i in range(100):
	scale_est, width, height = get_scale(pts_data[i])
	box1 = np.sqrt(width**2 + height**2)
	box2 = np.sqrt(width * height)
	box3 = (width + height)/2.

	center_est_w = 0.5 * (np.max(pts_data[i,:,0]) + np.min(pts_data[i,:,0]))
	center_est_h = 0.5 * (np.max(pts_data[i,:,1]) + np.min(pts_data[i,:,1]))

	scale    = meta[i,0]
	box_size = meta[i,1]
	center_w = meta[i,2]
	center_h = meta[i,3]

	#print("scale = {:.2f}, scale_calc= {:.2f}, ratio= {:.2f}, box_size= {:.2f}, box_1= {:.2f}, box_2= {:.2f}, box_3= {:.2f}, c_w= {:.2f}, c_est_w= {:.2f}, c_h= {:.2f}, c_est_h= {:.2f}".format(scale, scale_est, scale/scale_est, box_size, box1, box2, box3, center_w, center_est_w, center_h, center_est_h))
	print("box_size= {:.2f}, box_1= {:.2f}, box_2= {:.2f}, box_3= {:.2f}".format(box_size, box1, box2, box3))
