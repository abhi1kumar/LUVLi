#===============================================================================
# LUVLi Evaluation Scripts
#===============================================================================

# === 300-W Split 1 ====
python evaluate_face_all.py --exp_id run_108  -s 1 --pp "relu" --bs 12 --laplacian

# === 300-W Split 2 ====
python evaluate_face_all.py --exp_id run_109  -s 2 --pp "relu" --bs 12 --laplacian

# === AFLW-19 Split 3 ====
python evaluate_face_all.py --exp_id run_507  -s 3 --pp "relu" --bs 12 --laplacian

# === WFLW Split 4 ====
python evaluate_face_all.py --exp_id run_1005 -s 4 --pp "relu" --bs 12 --laplacian

# === MERL-RAV (AFLW-Ours) Split 5 ====
python evaluate_face_all.py --exp_id run_5004 -s 5 --pp "relu" --bs 12 --laplacian
