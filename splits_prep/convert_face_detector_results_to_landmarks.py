

"""
    Sample Run:
    python convert_face_detector_results_to_landmarks.py

    Converts a face detector results to landmark points file for further processing.
    The face detector should be in the following format of path, top,right,bottom,left
    eg:
    tim_pictures_3/195539828.jpg,150,384,390,143
    tim_pictures_3/5330-swatch-all_530x.jpg,233,380,549,40

    Version 1 2020-01-28 Abhinav Kumar
"""

import argparse
import os
import numpy as np
import pandas as pd

SEP      = " "
n_points = 68

file_path     = "tim_pictures_3/face_detector_results.txt"
output_folder = "tim_pictures_3"

df = pd.read_csv(file_path, header= None)

for i in range(df.shape[0]):
    # Get the basename of the file
    basename = os.path.basename(df.iloc[i,0])
    
    # Remove the extension and get the full path
    landmarks_file_full_path = os.path.join(output_folder, basename[:-4] + ".pts")

    with open(landmarks_file_full_path, 'w') as f:
        f.write("version: 1\n")
        f.write("n_points: " + str(n_points) + "\n")
        f.write("{\n")

        top    = float(df.iloc[i, 1])
        right  = float(df.iloc[i, 2])
        bottom = float(df.iloc[i, 3])
        left   = float(df.iloc[i, 4])

        for j in range(n_points):
            if j % 2 == 0:
                f.write(str(left)  + SEP + str(top) + "\n")
            else:
                f.write(str(right) + SEP + str(bottom) + "\n")

        f.write("}\n")

    if i % 1000 == 0 or i == df.shape[0]-1:
        print("{} images done".format(i+1))
