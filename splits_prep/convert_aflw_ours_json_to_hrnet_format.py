

"""
	Converts AFLW_ours json to file in HRNET AFLW format
	Only run after we have aflw_ours_frontal_train.json in the dataset
	directory

	python splits_prep/convert_aflw_ours_json_to_hrnet_format.py

"""

import numpy as np
from numpy import genfromtxt
from common_functions import *
import json

NUM_POINTS = 68
SEP = ","
np.set_printoptions(threshold=np.inf,suppress=True)

def prepare_csv(json_path, output_file):
	# Prepare the headers
	output_string=  "image_name,scale,box_size,center_w,center_h,"

	for i in range(NUM_POINTS):
		output_string += "original_" + str(i+1) + "_x" + SEP + "original_" + str(i+1) + "_y"
		if i != NUM_POINTS-1:
			output_string += SEP
		else:
			output_string += "\n"

	print("Loading JSON...")
	with open(json_path) as f:
		json_data = json.load(f)
	print("Done")

	#========================================================
	print("Writing data to {}...".format(output_file))
	with open(output_file, "w") as text_file:

		# Start getting the data
		for i in range(len(json_data)):
			img_path =  json_data[i]['img_paths']
			s = json_data[i]['scale_provided_det']
			c = json_data[i]['objpos_det']
			pts=json_data[i]['pts']
			pts=np.array(pts).flatten()

			# image_name, scale, box_size, center_w, center_h,
			# Replace the name of the directory initially since
			# we only need the relative path
			img_path = img_path.replace('./bigdata1/zt53/data/face/aflw_ours_organized/','')

			# box_size as -1
			output_string += img_path + SEP + str(s) + SEP + str(-100) + SEP + str(c[0]) + SEP + str(c[1]) + SEP

			for j in range(NUM_POINTS*2):
				output_string += str(pts[j])
				if j < (NUM_POINTS*2-1):
					output_string += SEP
				else:
					output_string += "\n" 

			if i % 100 == 0 or i == len(json_data)-1:
				print("{} images done".format(i))
				text_file.write(output_string)
				output_string = ""
			
	print("Done\n")

#prepare_csv("dataset/aflw_ours_frontal_train.json", "aflw_ours_frontal_train.csv")
#prepare_csv("dataset/aflw_ours_frontal_val.json", "aflw_ours_frontal_val.csv")
prepare_csv("dataset/aflw_ours_all_train.json"       , "aflw_ours_all_train.csv")
prepare_csv("dataset/aflw_ours_all_val.json"         , "aflw_ours_all_val.csv")
prepare_csv("dataset/aflw_ours_half_profile_val.json", "aflw_ours_half_profile_val.csv")
prepare_csv("dataset/aflw_ours_profile_val.json"     , "aflw_ours_profile_val.csv")
