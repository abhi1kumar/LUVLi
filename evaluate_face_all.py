

"""
    Sample Run:
    python evaluate_face_all.py --exp_id run_108  -s 1 --pp "relu" --bs 12 --laplacian
    python evaluate_face_all.py --exp_id run_109  -s 2 --pp "relu" --bs 12 --laplacian
    python evaluate_face_all.py --exp_id run_507  -s 3 --pp "relu" --bs 12 --laplacian
    python evaluate_face_all.py --exp_id run_1005 -s 4 --pp "relu" --bs 12 --laplacian
    python evaluate_face_all.py --exp_id run_5004 -s 5 --pp "relu" --bs 12 --laplacian
    
    python evaluate_face_all.py --exp_id run_272  -s 2 -w lr-0.0000025-29.pth.tar
    python evaluate_face_all.py --exp_id run_474  -s 2 --pp "relu"
    python evaluate_face_all.py --exp_id run_740  -s 2 --pp "relu" --mlp_tot_layers 1 --mlp_hidden_units 4096 --bs 12 --slurm
    python evaluate_face_all.py --exp_id run_728  -s 2 --get_mean_from_mlp -w lr-0.0000025-49.pth.tar --bs 6 --slurm
    python evaluate_face_all.py --exp_id run_78   -s 2 --pp "relu" --bs 12 --laplacian --layer_num 4

    Evaluates multiple datasets in one go over the model.
    Split-1 300-W
    Split-2 300-W
    Split-3 AFLW-19
    Split-4 WFLW-98
    Split-5 AFLW_ours-68

    Version 7 2019-11-01 Abhinav Kumar Support for AFLW_ours added(Split 5)
    Version 6 2019-11-01 Abhinav Kumar Support for WFLW-98 added (Split 4)
    Version 6 2019-10-30 Abhinav Kumar Support for AFLW-19 added (Split 3)
    Version 5 2019-07-30 Abhinav Kumar Original parser used
    Version 4 2019-07-17 Abhinav Kumar Supports NME-AUC plots and uncertainty ellipses as well
    Version 3 2019-07-13 Abhinav Kumar Single directory to store all evaluations now.
    Version 2 2019-07-12 Abhinav Kumar Support for split 1 added
    Version 1 2019-06-23 Abhinav Kumar
"""

import os, sys
import numpy as np
import torch
import argparse
from options.train_options import TrainOptions

join_char = "_"
suffix    = "evaluate"
log_file  = "val.log"
exp_dir   = "abhinav_model_dir"
nme_log   = "nme_auc.log"

def execute(command, print_flag= True):
    if print_flag:
        print(command)
    os.system(command)

#===============================================================================
# Argument Parsing
#===============================================================================
args = TrainOptions().parse()

if args.saved_wt_file == "face-layer-num-8-order-1-model-best.pth.tar":
    args.saved_wt_file = "lr-0.00002-49.pth.tar"


#===============================================================================
# Get what all jsons we have to run depending on the split
#===============================================================================
if args.split == 1:
    print("Running models trained on 300-W Split " + str(args.split))
    keyword   = ["test"]
    jsons     = ["dataset/normal_val.json"]
    class_num = 68
elif args.split == 2:    
    print("Running models trained on 300-W Split " + str(args.split))
    keyword   = ["300W_test", "menpo", "cofw_68", "multi_pie"]
    jsons     = ["dataset/all_300Wtest_val.json", "dataset/menpo_val.json", "dataset/cofw_68_val.json", "dataset/multi_pie_val.json"]
    class_num = 68
elif args.split == 3:    
    print("Running models trained on original AFLW-19 " + str(args.split))
    keyword   = ["aflw_full", "aflw_frontal"]
    jsons     = ["dataset/aflw_test_all.json", "dataset/aflw_test_frontal.json"]
    class_num = 19
elif args.split == 4:    
    print("Running models trained on WFLW-98 " + str(args.split))
    keyword   = ["wflw_full", "wflw_pose", "wflw_expression", "wflw_illumination", "wflw_makeup", "wflw_occlusion", "wflw_blur"]
    jsons     = ["dataset/wflw_test.json", "dataset/wflw_test_largepose.json", "dataset/wflw_test_expression.json", "dataset/wflw_test_illumination.json", "dataset/wflw_test_makeup.json", "dataset/wflw_test_occlusion.json", "dataset/wflw_test_blur.json"]
    class_num = 98
elif args.split == 5:    
    print("Running models trained on AFLW-68 " + str(args.split))
    keyword   = ["aflw_ours_all", "aflw_ours_frontal", "aflw_ours_half_profile", "aflw_ours_profile"]
    jsons     = ["dataset/aflw_ours_all_val.json", "dataset/aflw_ours_frontal_val.json", "dataset/aflw_ours_half_profile_val.json", "dataset/aflw_ours_profile_val.json"]
    class_num = 68
else:
    print("Some unknown split. Aborting!!!")
    sys.exit(0)

#===============================================================================
# Get all the options for the command_command
#===============================================================================
if args.pp == "":
    post_process = ""
else:
    post_process = " --pp " + args.pp

if args.smax:
    smax = " --smax --tau " + str(args.tau)
else:
    smax = ""  

if args.get_mean_from_mlp:
    get_mean_from_mlp = " --get_mean_from_mlp "
else:
    get_mean_from_mlp = ""

if args.use_heatmaps:
    use_heatmaps = " --use_heatmaps "
else:
    use_heatmaps = ""

if args.laplacian:
    laplacian = " --laplacian "
else:
    laplacian = ""

layer_num = " --layer_num " + str(args.layer_num)

# We only use the last hourglass to get the NLL on the final prediction.
# The final prediction is only taken from the last hourglass
# So, hg_wt_string should be like "0,0,0,0,0,0,0,1"
hg_wt_array     = np.zeros((args.layer_num, )).astype(np.uint8)
hg_wt_array[-1] = 1
hg_wt_string    = np.array2string(hg_wt_array, separator=',')[1:-1] # Remove two brackets at the end
hg_wt           = " --hg_wt " + hg_wt_string + " "

class_num       = " --class_num " + str(class_num)
use_visibility  = " --use_visibility "

folder = args.exp_id
gpu_id = str(args.gpu_id) #str(torch.cuda.device_count() - 2)

#===============================================================================
# Do the forward pass on the individual datasets
#===============================================================================
num_datasets = len(keyword)
for i in range(num_datasets):
    exp_id      = os.path.join(folder + join_char + suffix, keyword[i])
    exp_id_full = os.path.join(exp_dir    , exp_id)
    out_log     = os.path.join(exp_id_full, log_file)
    wt_file     = os.path.join(os.path.join(exp_dir, folder), args.saved_wt_file)

    if not os.path.isdir(exp_id_full):
        print("Making_directory= {}".format(exp_id_full))
        os.makedirs(exp_id_full)
    else:
        print("Directory_present= {}".format(exp_id_full))

    # Save images only for split 2,3,4,5 and not the first one
    if args.split >= 2 and i == 0:
            save_image_heatmaps = " --save_image_heatmaps "
    else:
        save_image_heatmaps = ""

    # Get common command for normal machine and slurm managed machines
    common_command = " --exp_id " + exp_id + " " + " --val_json "      + jsons[i] \
        + class_num + save_image_heatmaps \
        + " --saved_wt_file " + wt_file \
        + layer_num + post_process + smax + laplacian + use_visibility + get_mean_from_mlp + use_heatmaps \
        + hg_wt + " --wt_gau " + str(args.wt_gau) + " --wt_mse " + str(args.wt_mse) \
        + " --bs " + str(args.bs) \
        + " --mlp_tot_layers " + str(args.mlp_tot_layers) + " --mlp_hidden_units " +  str(args.mlp_hidden_units) \
        + " | tee " + out_log

    if not args.slurm:
        print("Not using slurm")
        command = "python validate_and_forward_pass.py --gpu_id " + gpu_id + common_command  
    else:    
        print("Using slurm")
        command = "srun --gres gpu:1 --cpus-per-task 4 -X python validate_and_forward_pass.py --slurm " + common_command
    execute(command)

#===============================================================================
# Now run NME and AUC calculations
#===============================================================================
for i in range(num_datasets):
    print("")
    print(keyword[i])
    exp_id      = os.path.join(folder + join_char + suffix, keyword[i])
    exp_id_full = os.path.join(exp_dir, exp_id)

    # Add split name for WFLW
    split_name = " --split_name " + keyword[i]
    
    # NME AUC calculation for inter-ocular normalized errors
    if args.split == 4:
        # WFLW benchmarks used 10% for NME_{inter-ocular}
        threshold  = " --threshold " + str(10)
    else:
        # Other benchmarks used 8% for NME_{inter-ocular}
        threshold  = " --threshold " + str(8)
        
    command = "python pylib/calculate_auc_from_nme.py -i " + os.path.join(exp_id_full, "nme_new_per_image.npy")     + split_name + threshold + " | tee "    + os.path.join(exp_id_full, nme_log)
    execute(command, print_flag= False)

    # For WFLW-98 we do not need to run box accuracies
    if args.split != 4:
        # NME and AUC calculation for box normalized errors
        # For box errors, threshold used is 7%
        threshold  = " --threshold " + str(7)
        command = "python pylib/calculate_auc_from_nme.py -i " + os.path.join(exp_id_full, "nme_new_box_per_image.npy") + split_name + threshold + " | tee -a " + os.path.join(exp_id_full, nme_log)
        execute(command, print_flag= False)
        
    # For these two datasets also do the external occlusion analysis    
    if keyword[i] == "cofw_68" or "aflw_ours" in keyword[i]:
        # Calculate NME for landmarks as well
        command = "python pylib/calculate_nme_uncetainty_on_landmarks.py -f " + exp_id_full + laplacian + " | tee -a " + os.path.join(exp_id_full, nme_log)
        execute(command, print_flag= False)

#===============================================================================
# Now plot some graphs as well for Split 2
#===============================================================================
if args.split == 2000:
    # Get the NME-AUC over landmark plot over 300W_Test
    exp_id_rel = folder + join_char + suffix
    command = "python plot_uncertainty_NeurlPS/plot_uncertainty.py -i "  + exp_id_rel
    execute(command)

    # Plot uncertainties from Multi-PIE
    command = "python plot/show_multi_pie_landmark_uncertainties.py -i " + exp_id_rel + laplacian
    execute(command)
