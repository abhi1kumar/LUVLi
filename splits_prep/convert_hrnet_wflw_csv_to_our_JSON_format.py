

"""
	Converts the WFLW csv files obtained from HRNET codebase to our JSON format

    The entire CSVs are available at https://1drv.ms/u/s!AiWjZ1LamlxzdmYbSkHpPYhI8Ms

    Sample Run:
    python splits_prep/convert_hrnet_wflw_csv_to_our_JSON_format.py

"""

import numpy as np
from numpy import genfromtxt
from common_functions import *

NUM_POINTS = 98

def get_data(path):
    # Ignore the first line
	read_data = genfromtxt(path, delimiter=',',skip_header= 1, dtype= None)
	names = []
	data  = np.zeros((read_data.shape[0], 2*NUM_POINTS + 3)) # 3 extra columns of meta information excluding names 

	for i in range(read_data.shape[0]):
		names.append(read_data[i][0])
		for j in range(data.shape[1]):
			data[i][j] = read_data[i][j+1]

	return names, data

def get_data_in_JSON_format(names, input_data, isValidation):
	num_points = input_data.shape[0]
	print("Number of points = {}".format(num_points))

	print(input_data.shape[1])
	data = []
	# image_name, scale, center_w,center_h
    # image_names is already in names
	meta     = input_data[:,:3]
	pts_data = input_data[:,3:]
	pts_data = pts_data.reshape((-1, NUM_POINTS, 2))

	for i in range(num_points):
		filename = os.path.join('bigdata1/zt53/data/face/wflw', names[i])
		pts = pts_data[i].tolist()

		data.append({
                "isValidation"       : isValidation,
                "pts_paths"          : "unknown.xyz",
                "dataset"            : "wflw",
                "objpos_det"         : [meta[i][1], meta[i][2]],
                "scale_provided_det" :  meta[i][0],
                "objpos_grnd"        : [meta[i][1], meta[i][2]],
                "scale_provided_grnd":  meta[i][0],
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
input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_train.csv")
output_path = "dataset/wflw_train.json"
wrapper(input_path, output_path, isValidation= False)

#================================================
# Testing splits
#================================================
input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_test.csv")
output_path = "dataset/wflw_test.json"
wrapper(input_path, output_path, isValidation= True)

input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_test_blur.csv")
output_path = "dataset/wflw_test_blur.json"
wrapper(input_path, output_path, isValidation= True)

input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_test_expression.csv")
output_path = "dataset/wflw_test_expression.json"
wrapper(input_path, output_path, isValidation= True)

input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_test_illumination.csv")
output_path = "dataset/wflw_test_illumination.json"
wrapper(input_path, output_path, isValidation= True)

input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_test_largepose.csv")
output_path = "dataset/wflw_test_largepose.json"
wrapper(input_path, output_path, isValidation= True)

input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_test_makeup.csv")
output_path = "dataset/wflw_test_makeup.json"
wrapper(input_path, output_path, isValidation= True)

input_path  = os.path.join(csv_folder, "wflw/face_landmarks_wflw_test_occlusion.csv")
output_path = "dataset/wflw_test_occlusion.json"
wrapper(input_path, output_path, isValidation= True)
