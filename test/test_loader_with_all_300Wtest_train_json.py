

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


index = 0
val_loader = torch.utils.data.DataLoader(
         FACE("dataset/all_300Wtest_train.json", ".", is_train= True),
         batch_size= 1, shuffle= False,
         num_workers=1, pin_memory= True)

# Visualize some images
for i, (img, _, points, _, s) in enumerate(val_loader):
    print(i)
    image = img[index].numpy()
    pts   = points[index].numpy()

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(swap_channels(image))
    plt_pts(plt, pts)
    
    image = torch.from_numpy(HumanAug.fliplr(img[index].numpy())).float()
    pts = HumanAug.shufflelr(points[index], width=image.size(2), dataset='face')
    plt.subplot(122)
    plt.imshow(swap_channels(image))
    plt_pts(plt, pts)

    plt.show()
    plt.close()
    if i > 10:
        break
    

val_loader = torch.utils.data.DataLoader(
         FACE("dataset/all_300Wtest_train.json", ".", is_train= True),
         batch_size= 10, shuffle= True,
         num_workers=1, pin_memory= True)

"""
num = 1000
c_all = np.zeros((num, ))
scale_rand = np.zeros((num, ))
cnt=0

for i, (img, _, points, _, s) in enumerate(val_loader):
    print(i)
    #plt.figure(figsize=(16, 8))
    image = img[index].numpy()
    pts   = points[index].numpy()

    scale_rand[cnt:cnt+s.shape[0]] = s.numpy()
    cnt += s.shape[0]
    if i > 100:
        break

plt.figure(figsize=(8, 8))
plt.hist(scale_rand)
plt.show()
"""
