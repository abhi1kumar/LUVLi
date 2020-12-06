#===============================================================================
# LUVLi Dataset Split Preparation Scripts
#===============================================================================

# === 300-W Split 1 and 2 ==== 
# Images already prepared and JSONs in the repo

# === 300W-LP-2D ====
python splits_prep/organize_300W_LP.py
python splits_prep/get_jsons_from_config.py -i splits_prep/config_300W_LP.txt

# === Menpo      ====
python splits_prep/organize_menpo_train.py
# JSON in the repo

# === COFW-68    ====
python splits_prep/organize_cofw_68_test.py
# JSON in the repo

# === Multi-PIE  ====
python splits_prep/organize_multi_pie.py
# JSON in the repo

# === AFLW-19    ====
# Image there. Make JSON
python splits_prep/convert_hrnet_aflw_csv_to_our_JSON_format.py

# === WFLW       ====
# Images there. Make JSON
python splits_prep/convert_hrnet_wflw_csv_to_our_JSON_format.py

# === MERL-RAV (AFLW-Ours) ====
# Already organized. Simply Rename
mv bigdata1/zt53/data/face/merl_rav_organized bigdata1/zt53/data/face/aflw_ours_organized
# JSONs in the repo
