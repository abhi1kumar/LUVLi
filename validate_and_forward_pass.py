

"""
    Sample Run:
    python validate_and_forward_pass.py --exp_dir abhinav_model_dir/ --exp_id run_496_eval      --pp "relu"    --saved_wt_file abhinav_model_dir/run_496/lr-0.0000025-29.pth.tar --bs 4 --gpu_id 1 --val_json dataset/all_300Wtest_val.json
    srun --gres gpu:1 --cpus-per-task 4 -X python validate_and_forward_pass.py --exp_dir abhinav_model_dir/ --exp_id run_462_test      --pp "relu"    --saved_wt_file abhinav_model_dir/run_462/lr-0.0000025-29.pth.tar| tee -a abhinav_model_dir/run_462_test/val.log
    srun --gres gpu:1 --cpus-per-task 4 -X python validate_and_forward_pass.py --exp_dir abhinav_model_dir/ --exp_id run_207_test      --pp "" --smax --saved_wt_file abhinav_model_dir/run_207/lr-0.0000025-29.pth.tar
    srun --gres gpu:1 --cpus-per-task 4 -X python validate_and_forward_pass.py --exp_dir abhinav_model_dir/ --exp_id run_272_300W_test --pp "" --smax --saved_wt_file abhinav_model_dir/run_272/lr-0.0000025-29.pth.tar --val_json dataset/all_300Wtest_val.json | tee abhinav_model_dir/run_272_300W_test/val.log     
    srun --gres gpu:1 --cpus-per-task 3 -X python validate_and_forward_pass.py --exp_dir abhinav_model_dir/ --exp_id run_272_menpo     --pp "" --smax --saved_wt_file abhinav_model_dir/run_272/lr-0.0000025-29.pth.tar --val_json dataset/menpo_val.json | tee abhinav_model_dir/run_272_menpo/val.log    
    srun --gres gpu:1 --cpus-per-task 3 -X python validate_and_forward_pass.py --exp_dir abhinav_model_dir/ --exp_id run_276_menpo     --pp "" --smax --saved_wt_file abhinav_model_dir/run_276/lr-0.0000025-29.pth.tar --val_json dataset/menpo_val.json | tee abhinav_model_dir/run_276_menpo/val.log

    Saves the corresponding images, heatmaps and everything by doing a forward pass

    Version 2 2019-06-24 Abhinav Kumar Tight bounding boxes used from the pts rather than from the val_loader since val_loader has box recorded in original image space rather than 256 x 256 space.
    Version 1 2019-06-20 Abhinav Kumar
"""

import os, time, sys
import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from data.face_bbx import FACE
from loss.gaussian_loss import *
from loss.gaussian_regularization_loss import *
from models.cu_net_prev_version_cholesky_common_for_HG import create_cu_net 
from options.train_options import TrainOptions
from pylib import FaceAcc, Evaluation, HumanAug
from pylib.HeatmapStats import get_spatial_mean_and_covariance
from utils.checkpoint import Checkpoint
from utils.logger import Logger
from utils.util import *
from utils.visualizer import Visualizer

cudnn.benchmark = True
flip_index = np.array([[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], # outline
            [17, 26], [18, 25], [19, 24], [20, 23], [21, 22],                                 # eyebrow
            [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],                       # eye
            [31, 35], [32, 34],                                                               # nose
            [48, 54], [49, 53], [50, 52], [59, 57], [58, 56],                                 # outer mouth
            [60, 64], [61, 63], [67, 65]])                                                    # inner mouth

images_save_path                        = "images"
gt_save_path                            = "ground_truth"
means_save_path                         = "means"
covar_save_path                         = "covar"
cholesky_save_path                      = "cholesky"
heatmaps_save_path                      = "heatmaps"
vis_gt_save_path                        = "vis_gt"
vis_estimated_save_path                 = "vis_estimated"

nme_new_per_image_path                  = "nme_new_per_image"
nme_new_box_per_image_path              = "nme_new_box_per_image"
nme_new_per_image_per_landmark_path     = "nme_new_per_image_per_landmark"
nme_new_box_per_image_per_landmark_path = "nme_new_box_per_image_per_landmark"

# The below variables are assigned values in the main function
f_path            = ""
weights_HG        = [0, 0, 0, 0, 0, 0, 0, 1.0]

def main():
    opt = TrainOptions().parse() 
    train_history = TrainHistoryFace()
    checkpoint = Checkpoint()
    visualizer = Visualizer(opt)
    exp_dir = os.path.join(opt.exp_dir, opt.exp_id)
    log_name = opt.vis_env + '_val_log.txt'
    visualizer.log_name = os.path.join(exp_dir, log_name)
    num_classes = opt.class_num

    if not opt.slurm:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    layer_num = opt.layer_num
    order     = opt.order
    net = create_cu_net(neck_size= 4, growth_rate= 32, init_chan_num= 128, 
                class_num= num_classes, layer_num= layer_num, order= order, 
                loss_num= layer_num, use_spatial_transformer= opt.stn, 
                mlp_tot_layers= opt.mlp_tot_layers, mlp_hidden_units= opt.mlp_hidden_units,
                get_mean_from_mlp= opt.get_mean_from_mlp)

    # Load the pre-trained model
    saved_wt_file = opt.saved_wt_file
    print("Loading weights from " + saved_wt_file)
    checkpoint_t = torch.load(saved_wt_file)
    state_dict = checkpoint_t['state_dict']

    for name, param in state_dict.items():
        name = name[7:]
        if name not in net.state_dict():
            print("=> not load weights '{}'".format(name))
            continue
        if isinstance(param, Parameter):
            param = param.data
        net.state_dict()[name].copy_(param)

    net = torch.nn.DataParallel(net).cuda() # use multiple GPUs

    # Optimizer
    if opt.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr, alpha=0.99,
                                        eps=1e-8, momentum=0, weight_decay=0)
    elif opt.optimizer == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr)
    else:
        print("Unknown Optimizer. Aborting!!!")
        sys.exit(0)
    print(type(optimizer))

    # Optionally resume from a checkpoint
    if opt.resume_prefix != '':
        # if 'pth' in opt.resume_prefix:
        #     trunc_index = opt.resume_prefix.index('pth')
        #     opt.resume_prefix = opt.resume_prefix[0:trunc_index - 1]
        checkpoint.save_prefix = os.path.join(exp_dir, opt.resume_prefix)
        checkpoint.load_prefix = os.path.join(exp_dir, opt.resume_prefix)[0:-1]
        checkpoint.load_checkpoint(net, optimizer, train_history)
    else:
        checkpoint.save_prefix = exp_dir + '/'
    print("Save prefix                           = {}".format(checkpoint.save_prefix))

    # Load data
    json_path  = opt.json_path
    train_json = opt.train_json
    val_json   = opt.val_json

    print("Path added to each image path in JSON = {}".format(json_path))
    print("Train JSON path                       = {}".format(train_json))
    print("Val JSON path                         = {}".format(val_json))

    # This train loader is useless
    train_loader = torch.utils.data.DataLoader(
         FACE(train_json, json_path, is_train= True),
         batch_size= opt.bs, shuffle= True,
         num_workers= opt.nThreads, pin_memory= True)

    val_loader = torch.utils.data.DataLoader(
         FACE(val_json, json_path, is_train= False),
         batch_size= opt.bs, shuffle= False,
         num_workers= opt.nThreads, pin_memory= True)

    if not opt.is_train:
        visualizer.log_path = os.path.join(opt.exp_dir, opt.exp_id, 'val_log.txt')
        val_loss, val_rmse, predictions = validate(val_loader, net,
                train_history.epoch[-1]['epoch'], visualizer, num_classes, flip_index)
        checkpoint.save_preds(predictions)
        return

    global f_path
    global weights_HG

    f_path      = exp_dir
    weights_HG  = [float(x) for x in opt.hg_wt.split(",")] 

    print ("Postprocessing applied                = {}".format(opt.pp)) 
    if (opt.smax):
        print("Scaled softmax used with tau          = {}".format(opt.tau))
    else:
        print("No softmax used")

    if opt.is_covariance:
        print("Covariance used from the heatmap")
    else:
        print("Covariance calculated from MLP")

    print("Individual Hourglass loss weights")
    print(weights_HG)
    print("wt_MSE (tradeoff between GLL and MSE in each hourglass)= " + str(opt.wt_mse))
    print("wt_gauss_regln (tradeoff between GLL and Gaussian Regularisation in each hourglass)= " + str(opt.wt_gauss_regln))

    # Optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume_prefix != '':
        start_epoch = train_history.epoch[-1]['epoch'] + 1

    # Training and validation
    start_epoch = 0
    if opt.resume_prefix != '':
        start_epoch = train_history.epoch[-1]['epoch'] + 1

    train_loss_orig_epoch   = []
    train_loss_gau_t1_epoch = []
    train_loss_gau_t2_epoch = []
    train_nme_orig_epoch    = []
    train_nme_gau_epoch     = []
    train_nme_new_epoch     = []

    val_loss_orig_epoch     = []
    val_loss_gau_t1_epoch   = []
    val_loss_gau_t2_epoch   = []
    val_nme_orig_epoch      = []
    val_nme_gau_epoch       = []
    val_nme_new_epoch       = []

    for epoch in range(1):       
        # Evaluate on validation set
        val_loss, val_loss_mse, val_loss_gau_t1, val_loss_gau_t2 , val_rmse_orig, val_rmse_gau, val_rmse_new_box, predictions= validate(val_loader, net, epoch, visualizer, opt, num_classes, flip_index)
        val_loss_orig_epoch.append(val_loss_mse)
        val_loss_gau_t1_epoch.append(val_loss_gau_t1)
        val_loss_gau_t2_epoch.append(val_loss_gau_t2)
        val_nme_orig_epoch.append(val_rmse_orig)
        val_nme_gau_epoch.append(val_rmse_gau)


def validate(val_loader, net, epoch, visualizer, opt, num_classes, flip_index):
    batch_time       = AverageMeter()

    rmses0           = AverageMeter()
    rmses1           = AverageMeter()
    rmses2           = AverageMeter()
    rmses_orig       = AverageMeter()
    rmses_gau        = AverageMeter()
    rmses_new        = AverageMeter()
    rmses_new_box    = AverageMeter()
    rmses_new_meta_box = AverageMeter()
    rmses_new_0      = AverageMeter()
    rmses_new_1      = AverageMeter()
    rmses_new_2      = AverageMeter()

    img_batch_list   = []
    pts_batch_list   = []

    # Objects which keep track of the loss across the entire epoch
    losses           = AverageMeter()
    losses_gau       = AverageMeter()
    losses_gau_t1    = AverageMeter()
    losses_gau_t2    = AverageMeter()
    losses_mse       = AverageMeter()
    losses_regln     = AverageMeter()
    losses_vis       = AverageMeter()

    # Default downsampling is 4 times.
    downsample   = 4.
    downsample_save = 1.
    img_dim = int(256/downsample_save)

    # Variables for saving Image, groundtruths, means, covariance and heatmaps   
    num_val_images                           = len(val_loader.dataset)
    if opt.save_image_heatmaps:
        img_save                             = np.zeros((num_val_images, 3, img_dim, img_dim))
        #heatmaps_save                        = np.zeros((8, num_val_images, 68, img_dim, img_dim))

    gt_save                                  = np.zeros((num_val_images, num_classes, 2))
    means_save                               = np.zeros((num_val_images, num_classes, 2))
    cholesky_save                            = np.zeros((num_val_images, num_classes, 3))
    covar_save                               = np.zeros((num_val_images, num_classes, 4))
    vis_gt_save                              = np.zeros((num_val_images, num_classes))    
    vis_estimated_save                       = np.zeros((num_val_images, num_classes))

    rmse_new_per_image_save                  = np.zeros((num_val_images,))
    rmse_new_box_per_image_save              = np.zeros((num_val_images,))
    rmse_new_per_image_per_landmark_save     = np.zeros((num_val_images, num_classes))
    rmse_new_box_per_image_per_landmark_save = np.zeros((num_val_images, num_classes))

    img_cnt          = 0

    # Loss functions
    mse_loss         = nn.MSELoss()
    bce_loss         = nn.BCELoss()
    if not opt.use_heatmaps:
        loss_fn          = FaceAlignLoss(laplacian= opt.laplacian, form= opt.laplacian_form)
        gauss_regln_loss = GaussianRegularizationLoss()

    # Switch to evaluate mode
    net.eval()

    end              = time.time()

    wt_gau_new         = opt.wt_gau
    wt_gauss_regln_new = opt.wt_gauss_regln
    wt_mse_new         = opt.wt_mse
    if opt.use_heatmaps:
       wt_mse_new      = 1.
       wt_gau_new      = 0.
    print("weight_MSE= {} weight_GLL= {} weight_GR= {}".format(wt_mse_new, wt_gau_new, wt_gauss_regln_new))
    
    for i, (img, heatmap, pts, _, _, _, htp_mask, _, visible_multiclass, meta_box_input_res) in enumerate(val_loader):
     
        # Input and Groundtruth
        img_var    = Variable(img).cuda()
        vis        = visible_multiclass.clone()
        vis[vis > 1] = 1
        vis        = Variable(vis[:,:, None].float()).cuda()
        # vis with zero points was producing weird error. Add a small constant 
        # to the invisibile points
        vis[vis<1] = constants.EPSILON 

        # pts contain the invalid points in the center. If you use pts to calculate
        # MSE it is going to be bad. Make a masked version of points for MSE
        pts_masked = pts.float() * vis.data.cpu().float()
        pts_var    = Variable(pts.float()/4.0).cuda() * vis
        heatmap    = heatmap.cuda(async=True)
        target_heatmap = Variable(heatmap)

        # Tensor at the output and neck of every hourglass
        output, out_y = net(img_var) # output batch_size x 68 x 64 x 64

        batch_size = pts.shape[0]
        num_points = pts.shape[1]

        # Scalar loss values
        loss       = 0.
        loss_term1 = 0.
        loss_term2 = 0.
        loss_mse   = 0.
        loss_regln = 0.
        loss_vis   = 0.

        hg_cnt     = 0

        for per_out in output:
            # Weight of this Hourglass
            weight_hg = weights_HG[hg_cnt]

            if opt.use_heatmaps:
                # Calculate MSE between the heatmaps
                tmp_loss = (per_out - target_heatmap) ** 2
                loss_t   = tmp_loss.sum() / tmp_loss.numel()
                
                # All other loss as zeros
                loss_gau = Variable(torch.zeros(1,).float()).cuda()
                loss_gauss_regln = loss_gau.clone()
                loss_term1_temp  = loss_gau.clone()
                loss_term2_temp  = loss_gau.clone()
                loss_vis_hg      = loss_gau.clone()

            else:
                if isinstance(out_y, list):
                    if len(out_y) > 1:
                        cholesky = out_y[-1]
                    else:
                        cholesky = out_y[0]
                else:
                    cholesky = out_y 
            
                pred_pts_new, covar, normalized_heatmaps = get_spatial_mean_and_covariance(per_out, use_softmax= opt.smax, tau= opt.tau, postprocess= opt.pp)
                pred_pts_new = pred_pts_new * vis
                covar = covar.view(covar.shape[0], covar.shape[1], 4)

                vis_estimated = cholesky[:,:,  3]
                # Concat the calculations for each image and each landmark.
                # pred_pts_new: batch_size x 68 x 2
                # out_y       : batch_size x 68 x 3
                # covar       : batch_size x 68 x 4
                if opt.is_covariance:
                    # pre_pts : batch_size x 68 x 6
                    pre_pts = torch.cat((pred_pts_new, covar)  ,2)
                else:
                    if opt.get_mean_from_mlp:
                        pred_pts_new = cholesky[:, :, 4:6]
                    
                    cholesky     = cholesky[:, :, 0:3]
                    
                    # pre_pts : batch_size x 68 x 5
                    pre_pts = torch.cat((pred_pts_new,cholesky),2)

                # Gaussian_loss, Gaussian Loss in stages, Loss_term1, Loss_term2, Loss_term1 in stages, Loss_term2 in stages
                loss_gau, loss_stages, loss_term1_temp, loss_term2_temp, loss_term1_stages, loss_term2_stages = loss_fn([pre_pts], pts_var, is_covariance=opt.is_covariance)

                # Loss used is MSE but error metric used is NME
                loss_t       = mse_loss(pred_pts_new, pts_var)

                # Regularization to force heatmaps to be gaussian
                loss_gauss_regln = gauss_regln_loss(normalized_heatmaps, pred_pts_new, covar)
                
                if opt.use_visibility:
                    loss_vis_hg = bce_loss(vis_estimated, vis.squeeze())
                else:
                    loss_vis_hg = Variable(torch.zeros(1,).float()).cuda()

            loss        += weight_hg * (wt_gau_new*loss_gau + wt_mse_new*loss_t + wt_gauss_regln_new*loss_gauss_regln + loss_vis_hg)
            loss_term1  += weight_hg *  wt_gau_new*loss_term1_temp
            loss_term2  += weight_hg *  wt_gau_new*loss_term2_temp
            loss_mse    += weight_hg *  wt_mse_new*loss_t
            loss_regln  += weight_hg *  wt_gauss_regln_new * loss_gauss_regln
            loss_vis    += weight_hg *  loss_vis_hg

            #if opt.save_image_heatmaps:
            #    heatmaps_save[hg_cnt, img_cnt:img_cnt + img.shape[0] ] = normalized_heatmaps.clone().cpu().data.numpy() # Clone so that the original Variable doesnot get detached from the graph
            hg_cnt   += 1

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Update the losses
        N = img.shape[0]
        losses.update       (loss.data[0]      , N)
        losses_gau_t1.update(loss_term1.data[0], N)
        losses_gau_t2.update(loss_term2.data[0], N)
        losses_mse.update   (loss_mse.data[0]  , N)
        losses_regln.update (loss_regln.data[0], N)
        losses_vis.update   (loss_vis.data[0]  , N)

        # pred_pts_0: Integer coordinates where heatmap is maximum
        # pred_pts_1: pred_pts_0       + 0.25*sign(gradient of pred_pts_0 at pt shifted by 1 pixel)
        # pred_pts_2: pred_pts_0 + 0.5 + 0.25*sign(gradient of pred_pts_0 at pt shifted by 1 pixel)
        # Each coords is batch_size x 68 x 2
        pred_pts_0, pred_pts_1, pred_pts_2 = FaceAcc.heatmap2pts(output[-1].data.cpu(), flag=0)
        pred_pts_2 -= 1
        pred_pts_2 = Variable(pred_pts_2).cuda()*vis
        pred_pts_2 = pred_pts_2.data.cpu()

        # NME in reality since the code calculate norm-2 values using numpy
        # rmse_orig = NME(pred_pts_0 - 0.5 + 0.25*sign(gradient of pred_pts_0 at pt shifted by 1 pixel), ground_truth)
        rmse_orig = np.sum(FaceAcc.per_image_rmse(pred_pts_2.numpy() * 4., pts_masked.numpy())) / img.size(0)
        rmses_orig.update(rmse_orig, img.size(0))

        if opt.use_heatmaps:
            # We will use the prediction from the heatmap since there is no spatial mean
            # available
            pred_pts_new = Variable(torch.Tensor(pred_pts_2)).cuda() * vis
        
        # rmse_new  = NME(centroid, ground_truth)
        rmse_new_per_image_per_landmark     = FaceAcc.per_image_rmse(pred_pts_new.data.cpu().numpy() * downsample, pts_masked.numpy(), per_landmark=True)
        rmse_new_per_image                  = FaceAcc.per_image_rmse(pred_pts_new.data.cpu().numpy() * downsample, pts_masked.numpy())
        rmse_new                            = np.sum(rmse_new_per_image) / img.size(0)
        rmses_new.update(rmse_new, img.size(0))

        # rmse_new_box = NME(centroid, ground_truth, ground_bounding_box)
        # For box calculation remember to use pts and not pts_masked
        # since min of pts_masked is always zero
        max_box, _ = torch.max(pts, 1)
        min_box, _ = torch.min(pts, 1)
        width_height_gd = max_box - min_box
        rmse_new_box_per_image_per_landmark = FaceAcc.per_image_rmse_with_bounding_box(pred_pts_new.data.cpu().numpy() * downsample, pts_masked.numpy(), width_height_gd.numpy(), per_landmark=True)        
        rmse_new_box_per_image              = FaceAcc.per_image_rmse_with_bounding_box(pred_pts_new.data.cpu().numpy() * downsample, pts_masked.numpy(), width_height_gd.numpy())        
        rmse_new_box                        = np.sum(rmse_new_box_per_image) / img.size(0)
        rmses_new_box.update(rmse_new_box, img.size(0))

        # rmse_new_meta_box = NME(centroid, ground_truth, meta_bounding_box)
        rmse_new_meta_box = np.sum(FaceAcc.per_image_rmse_with_bounding_box(pred_pts_new.data.cpu().numpy() * 4., pts_masked.numpy(), meta_box_input_res.numpy(), is_scale= True)) / img.size(0)
        rmses_new_meta_box.update(rmse_new_meta_box, img.size(0))

        # Applies rounding to the coordinates of the predicted heatmap. The next
        # 'new' variables with underscore study effect of quantization of maximum
        # value of predictions
        # pred_pts_new_0: Rounds them to integer and then adds 1
        # pred_pts_new_1: pred_pts_new_0       + 0.25*sign(gradient of pred_pts_new_0 at pt shifted by 1 pixel)
        # pred_pts_new_2: pred_pts_new_0 + 0.5 + 0.25*sign(gradient of pred_pts_new_0 at pt shifted by 1 pixel)
        pred_pts_new_0, pred_pts_new_1, pred_pts_new_2 = FaceAcc.pts_trans(output[-1].data.cpu(),pred_pts_new.data.cpu())

        # rmse_new_0 = NME(pred_pts_new_0, ground_truth)
        # rmse_new_1 = NME(pred_pts_new_1, ground_truth)
        # rmse_new_2 = NME(pred_pts_new_2, ground_truth)
        rmse_new_0 = np.sum(FaceAcc.per_image_rmse(pred_pts_new_0.numpy() * downsample, pts_masked.numpy())) / img.size(0)
        rmse_new_1 = np.sum(FaceAcc.per_image_rmse(pred_pts_new_1.numpy() * downsample, pts_masked.numpy())) / img.size(0)
        rmse_new_2 = np.sum(FaceAcc.per_image_rmse(pred_pts_new_2.numpy() * downsample, pts_masked.numpy())) / img.size(0)

        rmses_new_0.update(rmse_new_0, img.size(0))
        rmses_new_1.update(rmse_new_1, img.size(0))
        rmses_new_2.update(rmse_new_2, img.size(0))

        loss_dict = OrderedDict([('loss_val', losses.avg), ('loss_vis', losses_vis.avg), ('loss_val_t1', losses_gau_t1.avg), ('loss_val_t2', losses_gau_t2.avg), ('loss_mse', losses_mse.avg), ('loss_regln', losses_regln.avg), ('rmse_orig', rmses_orig.avg), ('rmse_new', rmses_new.avg), ('rmse_new_box', rmses_new_box.avg), ('rmse_new_meta_box', rmses_new_meta_box.avg), ('rmse_new_0', rmses_new_0.avg), ('rmse_new_1', rmses_new_1.avg), ('rmse_new_2', rmses_new_2.avg) ])

        if i % opt.print_freq == 0 or i==len(val_loader)-1:
            visualizer.print_log(epoch, i, len(val_loader), value1=loss_dict)


        if i == 0:
            tt = pred_pts_2
        if i>0:
            tt = torch.cat((tt,pred_pts_2),0)
        predictions = tt
        
        # Save the values the numpy arrays.
        if opt.save_image_heatmaps:
            img_save                            [img_cnt:img_cnt + img.shape[0]] = resize_4d_numpy(img.cpu().numpy(), downsample= downsample_save)

        gt_save                                 [img_cnt:img_cnt + img.shape[0]] = pts.float().numpy() / downsample_save
        means_save                              [img_cnt:img_cnt + img.shape[0]] = pred_pts_new.data.cpu().numpy() * 4./ downsample_save
        vis_gt_save                             [img_cnt:img_cnt + img.shape[0]] = visible_multiclass.cpu().numpy()
        
        if not opt.use_heatmaps:
            covar_save                          [img_cnt:img_cnt + img.shape[0]] = covar.data.cpu().numpy() * (4./ downsample_save) * (4./ downsample_save)
            cholesky_save                       [img_cnt:img_cnt + img.shape[0]] = cholesky.data.cpu().numpy() * (4./ downsample_save)  # Cholesky are similar to standard deviations. So multiplied by one time only
            vis_estimated_save                  [img_cnt:img_cnt + img.shape[0]] = vis_estimated.data.cpu().numpy()

        rmse_new_per_image_save                 [img_cnt:img_cnt + img.shape[0]] = rmse_new_per_image
        rmse_new_box_per_image_save             [img_cnt:img_cnt + img.shape[0]] = rmse_new_box_per_image
        rmse_new_per_image_per_landmark_save    [img_cnt:img_cnt + img.shape[0]] = rmse_new_per_image_per_landmark
        rmse_new_box_per_image_per_landmark_save[img_cnt:img_cnt + img.shape[0]] = rmse_new_box_per_image_per_landmark

        img_cnt += img.shape[0]

    # Write numpy arrays to file
    print("Started saving everything...")
    if opt.save_image_heatmaps:
        print("Saving images and heatmaps as well")
        np.save(os.path.join(f_path, images_save_path)                   , img_save)
        #np.save(os.path.join(f_path, heatmaps_save_path)                 , heatmaps_save)

    np.save(os.path.join(f_path, gt_save_path)                           , gt_save)
    np.save(os.path.join(f_path, means_save_path)                        , means_save)
    np.save(os.path.join(f_path, vis_gt_save_path)                       , vis_gt_save)
    
    if not opt.use_heatmaps:
        np.save(os.path.join(f_path, vis_estimated_save_path)            , vis_estimated_save)

        if opt.is_covariance: 
            np.save(os.path.join(f_path, covar_save_path)                , covar_save)
        else:
            np.save(os.path.join(f_path, cholesky_save_path)             , cholesky_save)

    np.save(os.path.join(f_path, nme_new_per_image_path)                 , rmse_new_per_image_save)
    np.save(os.path.join(f_path, nme_new_box_per_image_path)             , rmse_new_box_per_image_save)
    np.save(os.path.join(f_path, nme_new_per_image_per_landmark_path)    , rmse_new_per_image_per_landmark_save)
    np.save(os.path.join(f_path, nme_new_box_per_image_per_landmark_path), rmse_new_box_per_image_per_landmark_save)
    print("Saving done.")

    return losses.avg, losses_mse.avg,losses_gau_t1.avg, losses_gau_t2.avg, rmses_orig.avg, rmses_new.avg, rmses_new_box.avg, predictions

def resize_4d_numpy(input, downsample= 4):
    new_dim = int(256/downsample)
    output = np.zeros((input.shape[0], input.shape[1],  new_dim, new_dim))                   # b x c x h x w

    input   = input.transpose(0, 2, 3, 1)                                                    # b x h x w x c
    for i in range(input.shape[0]):
        temp = cv2.resize(input[i], dsize=(new_dim, new_dim), interpolation=cv2.INTER_CUBIC) # h x w x c
        output[i] = temp.transpose(2, 0, 1)                                                  # c x h x w

    return output

if __name__ == '__main__':
    main()
