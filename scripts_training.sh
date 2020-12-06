#===============================================================================
# LUVLi Training Scripts
#===============================================================================
cd abhinav_model_dir
mkdir run_61 run_10{7..9} run_503 run_507 run_1001 run_1005 run_5000 run_5004
cd ..

# === 300-W Split 1 ====
python train_face_gll.py --gpu_id 0 --exp_id run_108  --pp "relu" --laplacian --use_visibility | tee abhinav_model_dir/run_108/train.log

# === 300-W Split 2 ====
# Train on 300W-LP-2D heatmaps
python train_face_gll.py --gpu_id 0 --exp_id run_61   --pp ""     --train_json dataset/300W_LP_train.json      --val_json dataset/all_300Wtest_val.json --bulat_aug --use_heatmaps --lr 0.00025 --lr_policy 3 --nEpochs 41 --saved_wt_file ""  | tee abhinav_model_dir/run_61/train.log

# Finetune with 300W-LP-2D heatmap weights
python train_face_gll.py --gpu_id 0 --exp_id run_109  --pp "relu" --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json --bulat_aug --laplacian --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar | tee abhinav_model_dir/run_109/train.log

# Finetune with 300-W heatmap weights
python train_face_gll.py --gpu_id 0 --exp_id run_107  --pp "relu" --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json --bulat_aug --laplacian --use_visibility | tee abhinav_model_dir/run_107/train.log

# === AFLW-19 ====
# Train on AFLW-19 heatmaps
python train_face_gll.py --gpu_id 0 --exp_id run_503  --pp ""     --train_json dataset/aflw_train.json --val_json dataset/aflw_test_all.json --bulat_aug --use_heatmaps --lr 0.00025     --class_num 19 --lr_policy 3 --nEpochs 100 --saved_wt_file "" | tee abhinav_model_dir/run_503/train.log

# Finetune with AFLW-19 heatmap weights
python train_face_gll.py --gpu_id 0 --exp_id run_507  --pp "relu" --train_json dataset/aflw_train.json --val_json dataset/aflw_test_all.json --bulat_aug --laplacian    --use_visibility --class_num 19 --saved_wt_file abhinav_model_dir/run_503/lr-0.0000125-99.pth.tar | tee abhinav_model_dir/run_507/train.log

# === WFLW ====
# Train on WFLW heatmaps
python train_face_gll.py --gpu_id 0 --exp_id run_1001 --pp ""     --train_json dataset/wflw_train.json --val_json dataset/wflw_test.json --bulat_aug --use_heatmaps --lr 0.00025     --class_num 98 --lr_policy 3 --nEpochs 100 --saved_wt_file "" | tee abhinav_model_dir/run_1001/train.log

# Finetune with WFLW heatmap weights
python train_face_gll.py --gpu_id 0 --exp_id run_1005 --pp "relu" --train_json dataset/wflw_train.json --val_json dataset/wflw_test.json --bulat_aug --laplacian    --use_visibility --class_num 98 --saved_wt_file abhinav_model_dir/run_1001/lr-0.0000125-99.pth.tar | tee abhinav_model_dir/run_1005/train.log

# === MERL-RAV (AFLW_ours) ====
# Train on MERL-RAV (AFLW_ours) heatmaps
python train_face_gll.py --gpu_id 0 --exp_id run_5000 --pp ""     --train_json dataset/aflw_ours_all_train.json --val_json dataset/aflw_ours_all_val.json --bulat_aug --use_heatmaps --lr 0.00025 --lr_policy 3 --nEpochs 100 --saved_wt_file ""  | tee abhinav_model_dir/run_5000/train.log

# Finetune with MERL-RAV (AFLW_ours) heatmap weights
python train_face_gll.py --gpu_id 0 --exp_id run_5004 --pp "relu" --train_json dataset/aflw_ours_all_train.json --val_json dataset/aflw_ours_all_val.json --bulat_aug --use_visibility --laplacian --saved_wt_file abhinav_model_dir/run_5000/lr-0.0000125-99.pth.tar  | tee abhinav_model_dir/run_5004/train.log


#===============================================================================
# Ablation Studies
#===============================================================================
mkdir abhinav_model_dir/run_7{0..9}

# last hourglass wt
python train_face_gll.py --gpu_id 0 --exp_id run_70 --pp "relu"    --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug --laplacian --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar --hg_wt "0,0,0,0,0,0,0,1" | tee abhinav_model_dir/run_70/train.log

# LUVLi to MSE
python train_face_gll.py --gpu_id 0 --exp_id run_71 --pp "relu"    --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug --laplacian --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar --wt_mse 1 --wt_gau 0     | tee abhinav_model_dir/run_71/train.log

# LUVLi to UGGLI
python train_face_gll.py --gpu_id 0 --exp_id run_72 --pp "relu"    --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug                              --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar                           | tee abhinav_model_dir/run_72/train.log

# LUVLi to Gaussian Likelihood + visibility
python train_face_gll.py --gpu_id 0 --exp_id run_73 --pp "relu"    --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug             --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar                           | tee abhinav_model_dir/run_73/train.log

# No visibility
python train_face_gll.py --gpu_id 0 --exp_id run_74 --pp "relu"    --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug --laplacian                  --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar                           | tee abhinav_model_dir/run_74/train.log

# Get mean from MLP
python train_face_gll.py --gpu_id 0 --exp_id run_75 --pp "relu"    --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug --laplacian --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar --get_mean_from_mlp       | tee abhinav_model_dir/run_75/train.log

# relu --> smax 1
python train_face_gll.py --gpu_id 0 --exp_id run_76 --smax --tau 1 --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug --laplacian --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar                           | tee abhinav_model_dir/run_76/train.log

# relu --> smax 0.02
python train_face_gll.py --gpu_id 0 --exp_id run_77 --smax         --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug --laplacian --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar                           | tee abhinav_model_dir/run_77/train.log

# 8 hourglass --> 4 hourglass
python train_face_gll.py --gpu_id 0 --exp_id run_78  --pp "relu"   --train_json dataset/all_300Wtest_train.json --val_json dataset/all_300Wtest_val.json  --bulat_aug --laplacian --use_visibility --saved_wt_file abhinav_model_dir/run_61/lr-0.00005-40.pth.tar --layer_num 4 --hg_wt "1,1,1,1" | tee abhinav_model_dir/run_78/train.log
