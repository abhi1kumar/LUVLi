

"""
    Check the functioning of how face loader works on our dataset and if the
    point are same as the 300W dataset
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
from torch.autograd import Variable

index = 0
val_loader = torch.utils.data.DataLoader(
         FACE("dataset/aflw_ours_left_val.json", ".", is_train= False, keep_pts_inside= True),
         batch_size= 1, shuffle= True,
         num_workers=1, pin_memory= True)

# Visualize some images
for i, (img, _, points, _, _, _, _, s, visible_multiclass, _) in enumerate(val_loader):
    print(i)


    vis        = visible_multiclass.clone()
    vis[vis > 1] = 1
    # vis with zero points was producing weird error. Add a small constant 
    # to the invisibile points
    vis[vis<1] = 0.0001

    vis        = Variable(vis[:,:, None].float()).cuda()
    pts_var    = Variable(points.float()).cuda() * vis

    image = img[index].numpy()

    # Conver back to numpy
    pts_back   = pts_var.data.cpu().float().numpy()
    vis_back   = vis.data.cpu().float().numpy().astype(int)

    plt.figure(figsize=(16, 8))

    # Get visible points which have 1 in the visibility
    landmark_id_list_1 = np.where(vis_back[index] == 1)[0]
    pts_1 = pts_back[index,landmark_id_list_1]

    # Get externally occluded points which have zero in the visibility
    landmark_id_list_2 = np.where(vis_back[index] == 0)[0]
    pts_2 = pts_back[index,landmark_id_list_2]

    plt.subplot(111)
    plt.imshow(swap_channels(image))
    plt_pts(plt, pts_2, color= "red"      , text= landmark_id_list_2, shift= 2, text_color= "magenta")
    plt_pts(plt, pts_1, color= "limegreen", text= landmark_id_list_1, shift= 2, text_color= "blue")
    

    plt.show()
    plt.close()
    if i > 10:
        break    
