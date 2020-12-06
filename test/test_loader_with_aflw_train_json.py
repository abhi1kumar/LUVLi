"""
    Check the functioning of how face loader works
"""
import os, sys
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import random

sys.path.insert(0, os.getcwd())
from data.face_bbx import FACE
from plot.CommonPlottingOperations import *
from pylib import HumanPts, FaceAug, HumanAug, FacePts

batch_size = 24
index = 0
train_loader = torch.utils.data.DataLoader(
         FACE("dataset/aflw_train.json", ".", is_train= True),
         batch_size= batch_size, shuffle= False,
         num_workers=1, pin_memory= True)

# Visualize some images
# inp, heatmap, pts_input_res, heatmap_mask, s_rand, visible_multiclass, meta_box_size_input_res
for i, (img, _, points, _, s, visible_multiclass, meta_box_size_input_res) in enumerate(train_loader):
    #print(i)
    image = img[index].numpy()
    pts   = points[index].numpy()

    #plt.figure(figsize=(16, 8))
    #plt.subplot(121)
    #plt.imshow(swap_channels(image))
    #plt_pts(plt, pts)
    
    #plt.show()
    plt.close()
    #if i > 10:
    #    break
    if (i% int(1000./batch_size) == 0):
        print ("{} images done".format(i)) 
