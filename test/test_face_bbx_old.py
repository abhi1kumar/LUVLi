import os, sys
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, './pylib')
sys.path.insert(0, './utils')

from utils import imutils
from pylib import HumanPts, FaceAug, HumanAug, FacePts


def plt_pts(plt, pts, color='green', marker=True):
    for i in range(pts.shape[0]):
        x = pts[i,0]
        y = pts[i,1]
        plt.plot(x, y, marker='x', color=color)
        if marker:
            plt.text(x+0.3, y+0.3, str(i+1), fontsize=9)


def crop(img, center, scale, rot, res, size, check=False):
    """
        Called with scale = random number, rot = random number
        res = 256
        size = 200
    """

    scale_factor = float(scale * size) / float(res)
    # height, width = img.shape[0], img.shape[1]

    if check:
        new_img_size = np.floor(max(img.shape[0], img.shape[1]) / scale_factor)
        if new_img_size < 2:
            return img
        else:
            img = scipy.misc.imresize(img, size=1/scale_factor, interp='bilinear')
    else:
        if scale_factor < 2:
            scale_factor = 1
            print("we are here")
        else:
            new_img_size = np.floor(max(img.shape[0], img.shape[1]) / scale_factor)
            if new_img_size < 2:
                return img
            else:
                img = scipy.misc.imresize(img, size=1/scale_factor, interp='bilinear')

            # height, width = tmp_img.shape[0], tmp_img.shape[1]
    center = center / scale_factor
    scale = scale / scale_factor

    # Upper left point
    ul = np.array(HumanAug.TransformSinglePts([0,0], center, scale, 0, res, size, invert=1)).astype(int)
    # Bottom right point
    br = np.array(HumanAug.TransformSinglePts([res,res], center, scale, 0, res, size, invert=1)).astype(int)

    # make sure br - ul = (res, res)
    if scale_factor >= 2:
        br = br - (br - ul - res)

    # Padding so that when rotated proper amount of context is included
    pad = np.ceil(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2).astype(int)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if img.ndim > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    ht = img.shape[0]
    wd = img.shape[1]
    new_x = max(0, -ul[0]), min(br[0], wd) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], ht) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(wd, br[0])
    old_y = max(0, ul[1]), min(ht, br[1])

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        # print new_img.shape
        new_img = scipy.misc.imrotate(new_img, rot, interp='bilinear')
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = Image.fromarray(new_img.astype('uint8'), 'RGB')
    # new_img = scipy.misc.toimage(new_img.astype('uint8'))
    # new_img.save('tmp.jpg')
    return scipy.misc.imresize(new_img, (res, res))

def transform(img, pts, c, s, r, inp_res, std_size, check=False):
    #if np.random.random() <= 0.5:
    #img = torch.from_numpy(HumanAug.fliplr(img.numpy())).float()
    #pts = HumanAug.shufflelr(pts, width=img.size(2), dataset='face')
    #c[0] = img.size(2) - c[0]

    # Prepare image and groundtruth map
    inp = crop(imutils.im_to_numpy(img), c.numpy(), s.numpy(), r, inp_res, std_size, check=check)
    inp_vis=inp

    inp = imutils.im_to_torch(inp).float() # 3*256*256
    #print('img shape is ', inp.size())
    # inp = self.color_normalize(inp, self.mean, self.std)
    # pts_aug = HumanAug.TransformPts(pts.numpy(), c.numpy(), s.numpy(), r, self.out_res, self.std_size)
    pts_input_res = HumanAug.TransformPts(pts.numpy(), c.numpy(), s.numpy(), r, inp_res, std_size)
    pts_aug = pts_input_res * (1.*out_res/inp_res)

    # Generate ground truth
    heatmap, pts_aug = HumanPts.pts2heatmap(pts_aug, [out_res, out_res], sigma=1)
    heatmap = torch.from_numpy(heatmap).float()
    # pts_aug = torch.from_numpy(pts_aug).float()
    heatmap_mask = HumanPts.pts2mask(pts_aug, [out_res, out_res], bb = 10)

    return inp_vis, pts_input_res

inp_res=256
out_res=64
std_size=256
sigma=1
scale_factor=0.25
rot_factor=30
img_folder="."

# Load json
jsonfile = "dataset/normal_val.json"

with open(jsonfile, 'r') as anno_file:
    anno = json.load(anno_file)

a = anno[0]

img_path = os.path.join(img_folder, a['img_paths'])
pts_path = os.path.join(img_folder, a['pts_paths'])
#print('im_path---', img_path)

if pts_path[-4:] == '.txt':
    pts = np.loadtxt(pts_path)  # L x 2
elif pts_path[-4:] == '.pts':
    pts = FacePts.Pts2Lmk(pts_path)  # L x 2
else:
    pts = a['pts']

#read image
image = cv2.imread(img_path)

plt.subplot(3, 5, 1)
plt.imshow(image)
plt_pts(plt, pts)

pts = torch.Tensor(pts) # size is 68*2
assert torch.sum(pts - torch.Tensor(a['pts'])) == 0
s = torch.Tensor([a['scale_provided_det']])
print(s)

c = torch.Tensor(a['objpos_det'])
#print('pts shape is ', pts.size(), 's is ', s, 's shape is ', s.size())
# For single-person pose estimation with a centered/scaled figure
img = imutils.load_image(img_path) # the image in the original size
#print('img shape is ', img.size())
# print img.size()
# exit()
# img = scipy.misc.imread(img_path, mode='RGB') # CxHxW
# img = torch.from_numpy(img)

r = 0
width_height_gd = torch.Tensor(a["width_height_grnd"])

inp_vis, pts_input_res =  transform(img, pts, c, s, r, inp_res, std_size)
print(inp_vis.shape)
plt.subplot(3, 5, 2)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('res = 200')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)


inp_vis, pts_input_res =  transform(img, pts, c, s*1.1, r, inp_res, std_size)
print(inp_vis.shape)
plt.subplot(3, 5, 3)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('Times 1.1 , res = 200')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)


inp_vis, pts_input_res =  transform(img, pts, c, s*200./256, r, inp_res, inp_res)
print(inp_vis.shape)
plt.subplot(3, 5, 4)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('res = 256')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s*1.1*200./256, r, inp_res, inp_res)
print(inp_vis.shape)
plt.subplot(3, 5, 5)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('Times 1.1 , res = 256')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s, r, inp_res, std_size, check=True)
print(inp_vis.shape)
plt.subplot(3, 5, 7)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('res = 200, no weird check')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s*1.1, r, inp_res, std_size, check=True)
print(inp_vis.shape)
plt.subplot(3, 5, 8)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('Times 1.1 , res = 200, no weird check')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s*200./256, r, inp_res, inp_res, check=True)
print(inp_vis.shape)
plt.subplot(3, 5, 9)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('res = 256, no weird check')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s*1.1*200./256, r, inp_res, inp_res, check=True)
print(inp_vis.shape)
plt.subplot(3, 5, 10)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('Times 1.1 , res = 256, no weird check')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)


inp_vis, pts_input_res =  transform(img, pts, c, s, r, inp_res, std_size, check=False)
print(inp_vis.shape)
plt.subplot(3, 5, 12)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('res = 200')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s*1.2, r, inp_res, std_size, check=False)
print(inp_vis.shape)
plt.subplot(3, 5, 13)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('Times 1.2 , res = 200,')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s*200./256, r, inp_res, inp_res, check=False)
print(inp_vis.shape)
plt.subplot(3, 5, 14)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('res = 256')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)

inp_vis, pts_input_res =  transform(img, pts, c, s*1.2*200./256, r, inp_res, inp_res, check=False)
print(inp_vis.shape)
plt.subplot(3, 5, 15)
plt.imshow(inp_vis)
plt_pts(plt, pts_input_res)
plt.title('Times 1.2 , res = 256')
pts_input_res = pts_input_res.astype(int)
plt_pts(plt, pts_input_res, "red", False)
plt.show()
