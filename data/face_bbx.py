# Zhiqiang Tang, May 2017
import os, sys
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
from utils import imutils
from pylib import HumanPts, FaceAug, HumanAug, FacePts 
from pylib.ScaleDownCalculator import *
import pdb
from random import randint


#==================================================================
# Samples a normal random number between two standard deviations 
# from the normal
#==================================================================
def sample_from_bounded_gaussian(x):
    return max(-2*x, min(2*x, np.random.randn()*x))

class FACE(data.Dataset):
    def __init__(self, jsonfile, img_folder, inp_res= 256, out_res= 64, is_train= True, sigma= 1,
                 scale_factor= 0.25, rot_factor= 30, std_size= 200, use_occlusion= False, use_flipping= False, keep_pts_inside= False):

        self.img_folder = img_folder  # root image folders
        self.is_train = is_train  # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.std_size = std_size
        self.use_occlusion = use_occlusion
        self.keep_pts_inside = keep_pts_inside
        self.use_flipping = use_flipping

        # create train/val split
        with open(jsonfile, 'r') as anno_file:
            self.anno = json.load(anno_file)
        print("Loading JSON done")
        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            #if val['dataset'] == 'afw':
            #    print(idx, '----', val)

            if val['dataset'] != '300w_cropped':
                if val['isValidation'] == True: # or val['dataset'] == 'ibug': #The json script is done, so no need of this. In some of experiments, ibug may be train
                    self.valid.append(idx)
                else:
                    self.train.append(idx)
        # self.mean, self.std = self._compute_mean()
        if self.is_train:
            print("Total training images                 = {}".format(len(self.train)))
        else:
            print("Total validation images               = {}".format(len(self.valid)))

    def _compute_mean(self):
        meanstd_file = 'dataset/face.pth.tar'
        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = imutils.load_image(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def color_normalize(self, x, mean, std):
        if x.size(0) == 1:
            x = x.repeat(3, x.size(1), x.size(2))

        for t, m, s in zip(x, mean, std):
            t.sub_(m).div_(s)
        return x

    def __getitem__(self, index):

        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        
        if a['pts_paths'] == "unknown.xyz":
            pts = a['pts']
        else:
            pts_path = os.path.join(self.img_folder, a['pts_paths'])

            if pts_path[-4:] == '.txt':
                pts = np.loadtxt(pts_path)  # L x 2
            else:
                pts = a['pts']

        pts = np.array(pts)
        # Assume all points are visible for a dataset. This is a multiclass
        # visibility
        visible_multiclass = np.ones(pts.shape[0])

        if a['dataset'] == 'aflw_ours' or a['dataset'] == 'cofw_68':                        
            # The pts which are labelled -1 in both x and y are not visible points
            self_occluded_landmark     = (pts[:,0] == -1) & (pts[:,1] == -1)
            external_occluded_landmark = (pts[:,0] <  -1) & (pts[:,1] < -1)

            visible_multiclass[self_occluded_landmark]     = 0
            visible_multiclass[external_occluded_landmark] = 2
            
            # valid landmarks are those which are external occluded and not occluded
            valid_landmark             = (pts[:,0] != -1) & (pts[:,1] != -1)
                        
            # The points which are partially occluded have both coordinates as negative but not -1
            # Make them positive
            pts = np.abs(pts)
        
            # valid_landmark is 0 for to be masked and 1 for not to be masked
            # mask is 1 for to be masked and 0 for not to be masked
            pts_masked = np.ma.array(pts, mask = np.column_stack((1-valid_landmark, 1-valid_landmark)))
            pts_mean   = np.mean(pts_masked, axis=0)
        
            # Replace -1 by mean of valid landmarks. Otherwise taking min for
            # calculating geomteric mean of the box can create issues later.
            pts[self_occluded_landmark] = pts_mean.data
            
            scale_mul_factor = 1.1

        elif a['dataset'] == "aflw" or a['dataset'] == "wflw":
            self_occluded_landmark     = (pts[:,0] <= 0) | (pts[:,1] <= 0)
            valid_landmark             = 1 - self_occluded_landmark
            visible_multiclass[self_occluded_landmark]     = 0

            # valid_landmark is 0 for to be masked and 1 for not to be masked
            # mask is 1 for to be masked and 0 for not to be masked
            pts_masked = np.ma.array(pts, mask = np.column_stack((1-valid_landmark, 1-valid_landmark)))
            pts_mean   = np.mean(pts_masked, axis=0)

            # Replace -1 by mean of valid landmarks. Otherwise taking min for
            # calculating geomteric mean of the box can create issues later.
            pts[self_occluded_landmark] = pts_mean.data

            scale_mul_factor = 1.25
        
        else:
            scale_mul_factor = 1.1

        pts = torch.Tensor(pts) # size is 68*2  
        s = torch.Tensor([a['scale_provided_det']]) * scale_mul_factor
        c = torch.Tensor(a['objpos_det'])

        # For single-person pose estimation with a centered/scaled figure
        # the image in the original size
        img = imutils.load_image(img_path)

        r = 0
        s_rand = 1
        if self.is_train: #data augmentation for training data
            s_rand = (1 + sample_from_bounded_gaussian(self.scale_factor/2.))
            s = s*s_rand

            r = sample_from_bounded_gaussian(self.rot_factor/2.)
            
            #print('s shape is ', s.size(), 's is ', s)
            #if np.random.uniform(0, 1, 1) <= 0.6:
            #    r = np.array([0])

            if self.use_flipping:
                # Flip
                if np.random.random() <= 0.5:
                    img = torch.from_numpy(HumanAug.fliplr(img.numpy())).float()
                    pts = HumanAug.shufflelr(pts, width= img.size(2), dataset= 'face')
                    c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(np.random.uniform(0.6, 1.4)).clamp_(0, 1)
            img[1, :, :].mul_(np.random.uniform(0.6, 1.4)).clamp_(0, 1)
            img[2, :, :].mul_(np.random.uniform(0.6, 1.4)).clamp_(0, 1)

            if self.use_occlusion:
                # Apply a random black occlusion
                # C x H x W
                patch_center_row = randint(1, img.size(1))
                patch_center_col = randint(1, img.size(2))

                patch_height     = randint(1, img.size(1)/2)
                patch_width      = randint(1, img.size(2)/2)

                row_min          = max(0, patch_center_row - patch_height)
                row_max          = min(img.size(1), patch_center_row + patch_height)
                col_min          = max(0, patch_center_col - patch_width)
                col_max          = min(img.size(2), patch_center_col + patch_width)

                img[:, row_min:row_max, col_min:col_max] = 0

        # Prepare points first        
        pts_input_res = HumanAug.TransformPts(pts.numpy(), c.numpy(), s.numpy(), r, self.inp_res, self.std_size)

        # Some landmark points can go outside after transformation. Determine the
        # extra scaling required.
        # This can only be done for the training points. For validation, we do
        # not know the points location.
        if self.is_train and self.keep_pts_inside:
            # visible copy takes care of whether point is visible or not.
            visible_copy = visible_multiclass.copy()
            visible_copy[visible_multiclass > 1] = 1
            scale_down = get_ideal_scale(pts_input_res, self.inp_res, img_path, visible= visible_copy)
            s = s/scale_down
            s_rand = s_rand/scale_down
            pts_input_res = HumanAug.TransformPts(pts.numpy(), c.numpy(), s.numpy(), r, self.inp_res, self.std_size)

        if a['dataset'] =="aflw":
            meta_box_size = a['box_size']
            # We convert the meta_box size also to the input res. The meta_box
            # is not formed by the landmark point but is supplied externally.
            # We assume the meta_box as two points [meta_box_size, 0] and [0, 0]
            # apply the transformation on top of it
            temp = HumanAug.TransformPts(np.array([[meta_box_size, 0], [0, 0]]), c.numpy(), s.numpy(), r, self.inp_res, self.std_size)
            # Passed as array of 2 x 2
            # we only want the transformed distance between the points
            meta_box_size_input_res = np.linalg.norm(temp[1]- temp[0])
        else:
            meta_box_size_input_res = -10 # some invalid number

        # pts_input_res is in the size of 256 x 256
        # Bring down to 64 x 64 since finally heatmap will be 64 x 64
        pts_aug = pts_input_res * (1.*self.out_res/self.inp_res)

        # Prepare image
        inp = HumanAug.crop(imutils.im_to_numpy(img)         , c.numpy(), s.numpy(), r, self.inp_res, self.std_size)
        inp_vis=inp
        inp = imutils.im_to_torch(inp).float() # 3*256*256

        # Generate proxy ground truth heatmap
        heatmap, pts_aug = HumanPts.pts2heatmap(pts_aug, [self.out_res, self.out_res], sigma= self.sigma)
        heatmap = torch.from_numpy(heatmap).float()
        heatmap_mask = HumanPts.pts2mask(pts_aug, [self.out_res, self.out_res], bb = 10)

        if self.is_train:
            return inp, heatmap, pts_input_res, heatmap_mask, s_rand, visible_multiclass, meta_box_size_input_res
        else:
            return inp, heatmap, pts_input_res,  c, s, index, inp_vis, s_rand, visible_multiclass, meta_box_size_input_res

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
