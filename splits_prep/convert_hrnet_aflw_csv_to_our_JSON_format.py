

"""
	Converts the AFLW csv files obtained from HRNET codebase to our JSON format

    Sample Run:
    python splits_prep/convert_hrnet_aflw_csv_to_our_JSON_format.py

"""

import numpy as np
from numpy import genfromtxt
from common_functions import *

def get_data(path):
	read_data = genfromtxt(path, delimiter=',',skip_header= 1, dtype= None)
	names = []
	data  = np.zeros((read_data.shape[0], 42))

	for i in range(read_data.shape[0]):
		names.append(read_data[i][0])
		for j in range(data.shape[1]):
			data[i][j] = read_data[i][j+1]

	return names, data

def get_data_in_JSON_format(names, input_data, isValidation):
	num_points = input_data.shape[0]
	print("Number of points = {}".format(num_points))

	data = []
	# path, scale, box_size, center_w,center_h
	meta     = input_data[:,:4]
	pts_data = input_data[:,4:]
	pts_data = pts_data.reshape((-1, 19, 2))

	for i in range(num_points):
		filename = os.path.join('bigdata1/zt53/data/face/aflw', names[i])
		pts = pts_data[i].tolist()

		data.append({
                "isValidation"       : isValidation,
                "pts_paths"          : "unknown.xyz",
                "dataset"            : "aflw",
                "objpos_det"         : [meta[i][2], meta[i][3]],
                "scale_provided_det" :  meta[i][0],
                "objpos_grnd"        : [meta[i][2], meta[i][3]],
                "scale_provided_grnd":  meta[i][0],
                "box_size"           :  meta[i][1],
                "pts"                : pts,
                "img_paths"          : filename
            })

	return data

def wrapper(input_path, output_path, isValidation):
	names, input_data = get_data(input_path)
	data = get_data_in_JSON_format(names, input_data, isValidation = isValidation)
	write_data_to_json(data, output_path)
	print("Done...\n")

csv_folder= "dataset_csv"
#================================================
# Training split
#================================================
input_path  = os.path.join(csv_folder, "aflw/face_landmarks_aflw_train.csv")
output_path = "dataset/aflw_train.json"
wrapper(input_path, output_path, isValidation= False)

#================================================
# Testing splits
#================================================
input_path  = os.path.join(csv_folder, "aflw/face_landmarks_aflw_test.csv")
output_path = "dataset/aflw_test_all.json"
wrapper(input_path, output_path, isValidation= True)

input_path  = os.path.join(csv_folder, "aflw/face_landmarks_aflw_test_frontal.csv")
output_path = "dataset/aflw_test_frontal.json"
wrapper(input_path, output_path, isValidation= True)
