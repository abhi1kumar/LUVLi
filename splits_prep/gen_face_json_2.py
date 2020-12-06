import sys, os, random
import numpy as np
import json
from pylib import FileIO, FacePts, FaceAug
from scipy.io import loadmat

def list_files(path, format):
    list = []
    for root, dirs, files in os.walk(path):
        if len(dirs) != 0:
            for fold in dirs:
                files = os.listdir(root + fold)
                for file in sorted(files):
                    if file.endswith(format):
                        list.append(fold + '/' + file)
        else:
            for file in sorted(files):
                if file.endswith(format):
                    list.append(file)
        return list

def convert_to_list(dict, img_names, bbs_detector, bbs_ground_truth, name_prefix=''):
    for i in range(0, dict['bounding_boxes'][0].shape[0]):
        img_names.append(name_prefix + dict['bounding_boxes'][0, i]['imgName'][0, 0][0])
        bbs_detector.append(dict['bounding_boxes'][0, i]['bb_detector'][0, 0][0])
        bbs_ground_truth.append(dict['bounding_boxes'][0, i]['bb_ground_truth'][0, 0][0])


if __name__ == '__main__':
    data_path = '/home/don/projects/face-data/'
    std_size = 200
    annot_arr = []
    out_of_bbx_count = 0
    # list
    dsource = ['helen/', 'afw/', 'ibug/', 'lfpw/']
    for ds in dsource:
        img_list = list_files(data_path+ds, '.jpg')
        if len(img_list) == 0:
            img_list = FileIO.ListFileInFolderRecursive(data_path + ds, '.png')
        print '%s: %d' % (ds, len(img_list))
        # print img_list[0:10]
        # exit()
        img_names = []
        bbs_detector = []
        bbs_ground_truth = []
        if 'helen' in ds or 'lfpw' in ds:
            bbxoes = loadmat(data_path + 'Bounding Boxes/bounding_boxes_' + ds[0:-1] + '_trainset.mat')
            convert_to_list(bbxoes, img_names, bbs_detector, bbs_ground_truth, 'trainset/')
            bbxoes = loadmat(data_path + 'Bounding Boxes/bounding_boxes_' + ds[0:-1] + '_testset.mat')
            convert_to_list(bbxoes, img_names, bbs_detector, bbs_ground_truth, 'testset/')
        else:
            bbxoes = loadmat(data_path + 'Bounding Boxes/bounding_boxes_' + ds[0:-1] + '.mat')
            convert_to_list(bbxoes, img_names, bbs_detector, bbs_ground_truth)
        for one_img in img_list:
            token = one_img.split('/')
            img_name = token[-1]
            # print token
            if len(token) == 2:
                fold = token[-2]
                img_fpath = ds+fold+'/' + img_name
                pts_fpath = ds+fold+'/' + img_name[:-4] + '.pts'
                line = '%s %s' % (img_fpath, pts_fpath)
                print line
            elif len(token) == 1:
                img_fpath = ds+img_name
                pts_fpath = ds+img_name[:-4] + '.pts'
                line = '%s %s' % (img_fpath, pts_fpath)
                print line
            # exit()
            tmp_annot = {}
            tmp_annot['dataset'] = ds[:-1]
            # print tmp_annot['dataset']
            # exit()
            tmp_annot['img_paths'] = img_fpath
            tmp_annot['pts_paths'] = pts_fpath
            if 'test' in one_img:
                # print one_img
                # exit()
                tmp_annot['isValidation'] = True
            else:
                tmp_annot['isValidation'] = False

            # pts = FacePts.Pts2Lmk(data_path+pts_fpath)  # L x 2
            pts = np.genfromtxt(data_path+pts_fpath, delimiter=' ', skip_header=3, skip_footer=1)
            # print type(pts)
            # pts = pts.tolist()
            tmp_annot['pts'] = pts.tolist()
            index = img_names.index(one_img)
            bb_detector = bbs_detector[index]
            bb_ground_truth = bbs_ground_truth[index]
            # check whether pts out of bbx
            x_indicator = (pts[:, 0] < bb_ground_truth[0]) | (pts[:, 0] > bb_ground_truth[2])
            y_indicator = (pts[:, 1] < bb_ground_truth[1]) | (pts[:, 1] > bb_ground_truth[3])
            tmp_count = np.sum(x_indicator | y_indicator)
            if tmp_count > 0:
                print tmp_count
            out_of_bbx_count += tmp_count

            c_det = [(bb_detector[0]+bb_detector[2])/2., (bb_detector[1]+bb_detector[3])/2.]
            c_grnd = [(bb_ground_truth[0] + bb_ground_truth[2]) / 2., (bb_ground_truth[1] + bb_ground_truth[3]) / 2.]
            tmp_annot['objpos_det'] = c_det
            tmp_annot['objpos_grnd'] = c_grnd
            size_det = max(bb_detector[2]-bb_detector[0], bb_detector[3]-bb_detector[1])
            size_grnd = max(bb_ground_truth[2] - bb_ground_truth[0], bb_ground_truth[3] - bb_ground_truth[1])
            tmp_annot['scale_provided_det'] = 1. * size_det / std_size
            tmp_annot['scale_provided_grnd'] = 1. * size_grnd / std_size
            annot_arr.append(tmp_annot)
            # # check whether pts out of bbx after rescaling
            # bb_detector_rescaled = np.copy(bb_detector)
            # x_len = bb_detector[2] - bb_detector[0]
            # x_len *= 1.5
            # y_len = bb_detector[3] - bb_detector[1]
            # y_len *= 1.5
            # bb_detector_rescaled[0] = c_det[0] - x_len / 2
            # bb_detector_rescaled[2] = c_det[0] + x_len / 2
            # bb_detector_rescaled[1] = c_det[1] - y_len / 2
            # bb_detector_rescaled[3] = c_det[1] + y_len / 2
            # x_indicator = (pts[:, 0] <= bb_detector_rescaled[0]) | (pts[:, 0] >= bb_detector_rescaled[2])
            # y_indicator = (pts[:, 1] <= bb_detector_rescaled[1]) | (pts[:, 1] >= bb_detector_rescaled[3])
            # tmp_count = np.sum(x_indicator | y_indicator)
            # if tmp_count > 0:
            #     print tmp_count
            # out_of_bbx_count += tmp_count
            # exit()
    print 'total out of bbx pts number: ', out_of_bbx_count
    with open(data_path+'face.json', 'w') as fp:
        json.dump(annot_arr, fp)
