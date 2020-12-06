

"""
    Sample Run
    python splits_prep/extract_frames_from_video.py

    Version 1 2019-08-14 Abhinav Kumar
"""

import os, sys
sys.path.insert(0, os.getcwd())
import json
import argparse
import cv2

from CommonPlottingOperations import *

#===============================================================================
# Argument Parsing
#===============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input_video_path', help= "path of the input video" , default= "video/face.mp4")
ap.add_argument(      '--output_image_dir', help= "output directory in which frames are extracted", default= "bigdata1/zt53/data/face/demo")
args = ap.parse_args()

vidcap = cv2.VideoCapture(args.input_video_path)
success,image = vidcap.read()
count = 0

makedir(args.output_image_dir)
while success:
  cv2.imwrite(os.path.join(args.output_image_path, str(count+1).zfill(6) + ".png"), image)    
  success,image = vidcap.read()
  count += 1
  if count % 100 == 0:
      print('Read frames {}'.format(count))

