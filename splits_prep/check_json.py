
"""
    Checks the two json - one given by Wenxuan and the one created by Abhinav's
    script - get_json_from_config.py

    Version 1 Abhinav Kumar 2019-05-27
"""

import os
import json
import numpy as np
import random

# Samples to query in normal_val.json
num_to_check   = 10

with open('./dataset/face.json') as json_file:
    data1 = json.load(json_file)
num_samples = len(data1)

with open('./dataset/normal_val.json') as json_file:
    val_data = json.load(json_file)
num_val      = len(val_data)

# Randomly select indices of the array
indices_random = random.sample(xrange(num_val), num_to_check)

for ind in range(num_to_check):
    j = indices_random[ind]
    
    scale   = val_data[j]["scale_provided_grnd"]
    name1   = val_data[j]["img_paths"]
    key_ds  = val_data[j]["dataset"]
    key     = os.path.basename(name1)
    key_val = val_data[j]["isValidation"]

    flag = False
    for i in range(num_samples):
        objpos_det = np.array(data1[i]["objpos_det"])
        pts        = np.array(data1[i]["pts"])
        curr_scale = np.array(data1[i]["scale_provided_grnd"])

        name2      = data1[i]["img_paths"]
        curr_name  = os.path.basename(name2)
        curr_ds    = data1[i]["dataset"]
        curr_val   = data1[i]["isValidation"]

        if (key == curr_name and key_ds == curr_ds):

            if key_ds == "ibug": # Do not check since ibug has been put in training in face.json
                print("name1= {:60s} name2= {:60s} Ground_truth= {:6f} Computed= {:6f}\tDiff= {:.6f}".format(name1, name2, curr_scale, scale, np.abs(curr_scale-scale)))
                flag = True
                break
            else:
                if (key_val == curr_val):
                    print("name1= {:60s} name2= {:60s} Ground_truth= {:6f} Computed= {:6f}\tDiff= {:.6f}".format(name1, name2, curr_scale, scale, np.abs(curr_scale-scale)))
                    flag = True
                    break

    if (flag == False):
        print("Not found name1= {:60s}".format(name1))
