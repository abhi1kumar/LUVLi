# LUVLi and UGLLI Face Alignment
[LUVLi Face Alignment: Estimating Landmarks' Location, Uncertainty, and Visibility Likelihood](https://arxiv.org/pdf/2004.02980.pdf), [CVPR 2020](http://cvpr2020.thecvf.com/)

[[slides](https://docs.google.com/presentation/d/1jnvrBZWsX7PsAYFDBC-qQlQS8bLo_6PPEtB5gr7gMRU/edit)], [[1min_talk](https://www.youtube.com/watch?v=ZbKJvD_tO7Y)], [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Kumar_LUVLi_Face_Alignment_CVPR_2020_supplemental.zip)],[[demo](https://www.cse.msu.edu/computervision/kumar_marks_mou_wang_jones_cherian_akino_liu_feng_cvpr2020.mp4)]

[UGLLI Face Alignment: Estimating Uncertainty with Gaussian Log-Likelihood Loss](http://www.merl.com/publications/docs/TR2019-117.pdf#page=3), [ICCV](https://iccv2019.thecvf.com/) Workshops on Statistical Deep Learning in Computer Vision 2019 

[[slides](https://docs.google.com/presentation/d/1gHu5D0sb5dVvEyiqpJXry2oHqpfrMbmecJpCS8JDets/edit)], [[poster](https://docs.google.com/presentation/d/1aDgnO7aAvqbhvlAxoLitmXRS-Wj7tCfCbSt5nGY6LMU/edit)], [[news](https://www.merl.com/news/award-20191027-1293)], [[Best Oral Presentation Award](https://www.merl.com/public/img/news/photo-1293.jpg)]

This repository is based on the [DU-Net](https://github.com/zhiqiangdon/CU-Net) code.

### References
Please cite the following papers if you find this repository useful:
```
@inproceedings{kumar2020luvli,
  title={LUVLi Face Alignment: Estimating Landmarks' Location, Uncertainty, and Visibility Likelihood},
  author={Kumar, Abhinav and Marks, Tim K. and Mou, Wenxuan and Wang, Ye and Jones, Michael and Cherian, Anoop and Koike-Akino, Toshiaki and Liu, Xiaoming and Feng, Chen},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}

@inproceedings{kumar2019uglli,
  title={UGLLI Face Alignment: Estimating Uncertainty with Gaussian Log-Likelihood Loss},
  author={Kumar, Abhinav and Marks, Tim K and Mou, Wenxuan and Feng, Chen and Liu, Xiaoming},
  booktitle={ICCV Workshops on Statistical Deep Learning in Computer Vision},
  year={2019}
}
```

### Requirements
1. Python 2.7
2. [Pytorch](http://pytorch.org) 0.3.0 or 0.3.1
3. Torchvision 0.2.0
4. Cuda 8.0
5. Ubuntu 18.04

Other platforms have not been tested.

### Installation
Clone the repo first. Unless otherwise stated the scripts and instructions assume working directory is the project root. 

There are two ways to run this repo - through Conda or through pip.

##### Conda install
Install conda first:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
source ~/.bashrc
conda list
```

Then install the desired packages:
```bash
conda env create --file conda_py27.yml
conda activate py27
```

##### Pip install

```bash
virtualenv --python=/usr/bin/python2.7 py27
source py27/bin/activate
pip install torch==0.3.1 -f https://download.pytorch.org/whl/cu80/stable
pip install torchvision==0.2.0
pip install sklearn opencv-python 
sudo apt-get install libfreetype6-dev
sudo apt-get install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev
pip install configparser seaborn
```


### Directory structure
We need to make some extra directories to store the datasets and the models
```bash
cd $PROJECT_DIR

# This directory stores the models of the training in its sub-directories
mkdir abhinav_model_dir

# For storing train datasets
mkdir -p bigdata1/zt53/data/face

# For storing csv
mkdir dataset_csv
```

### Extra files
We also use the [DU-Net](https://github.com/zhiqiangdon/CU-Net) 300-W Split 1 heatmap model for training. Please contact [Zhiqiang Tang](https://sites.google.com/site/zhiqiangtanghomepage) to get this.
1. ```face-layer-num-8-order-1-model-best.pth.tar``` - Base 300-W Split1 face model from which everything is finetuned 

Now copy this file:
```bash
cp face-layer-num-8-order-1-model-best.pth.tar $PROJECT_DIR
```


<br/><br/>
***
## Facial Landmark Localization
<br/>

### Download the datasets

The following Face datasets are used for training and testing - 
1. [AFW](https://ibug.doc.ic.ac.uk/download/annotations/afw.zip) 
2. [HELEN](https://ibug.doc.ic.ac.uk/download/annotations/helen.zip)
3. [IBUG](https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip)
4. [LFPW](https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip)
5. [300W Cropped indoor and outdoor](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) - available in 4 parts
6. [Menpo](https://www.dropbox.com/s/o3615gx23xohs00/menpo_challenge_trainset.zip)
7. [COFW-Color](http://www.vision.caltech.edu/xpburgos/ICCV13/Data/COFW_color.zip)
8. [Multi-PIE](https://www.flintbox.com/public/project/4742/)
9. [300W_LP](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing)
10. AFLW-19 Drop an email to michael.opitz@icg.tugraz.at to get the dataset mailed to you
11. [WFLW-98](https://drive.google.com/uc?id=1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC&export=download)
12. [MERL-RAV](https://github.com/abhi1kumar/MERL-RAV_dataset) (We refer to MERL-RAV as AFLW_ours in this repo) 


The Splits are made as follows

| Splits | Name                 | Datasets |
|--------|--------------------- |----------| 
| 1      | 300-W Split 1        | 1-4|
| 2      | 300-W Split 2        | 1-9|
| 3      | AFLW-19              | 10 |
| 4      | WFLW                 | 11 |
| 5      | MERL-RAV (AFLW_ours) | 12 |

Extract and move all the datasets to the ```bigdata1/zt53/data/face``` directory. Follow the [MERL-RAV dataset instructions](https://github.com/abhi1kumar/MERL-RAV_dataset#instructions) to get the ```merl_rav_organized``` directory.

Next download the [HR-Net](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) processed annotations of AFLW and WFLW dataset from [one-drive](https://1drv.ms/u/s!AiWjZ1LamlxzdmYbSkHpPYhI8Ms). Extract and move them to the ```dataset_csv``` directory.

The directory structure should look like this:
```bash
./FaceAlignmentUncertainty/
|--- abhinav_model_dir/
|
|--- bigdata1/
|      |---zt53/
|             |---data/
|                    |---face/
|                          |---300W
|                          |     |--- 01_Indoor/
|                          |     |--- 02_Outdoor/
|                          |---300W_LP/
|                          |---aflw/
|                          |     |---flickr/
|                          |            |---0/
|                          |            |---1/
|                          |            |---2/
|                          |---afw/
|                          |---COFW_color/
|                                 |---COFW_test_color.mat
|                          |---helen/
|                          |---ibug/
|                          |---lfpw/
|                          |---menpo/
|                          |---merl_rav_organized/
|                          |---Multi-PIE_original_data/
|                          |---wflw/
|
|--- Bounding\ Boxes/
|--- data/
|--- dataset/
|
|--- dataset_csv/
|          |---aflw/
|          |     |---face_landmarks_aflw_test.csv
|          |     |---face_landmarks_aflw_test_frontal.csv
|          |     |---face_landmarks_aflw_train.csv
|          |---wflw/
|                |---face_landmarks_wflw_test.csv
|                |---face_landmarks_wflw_test_blur.csv
|                |---face_landmarks_wflw_test_expression.csv
|                |---face_landmarks_wflw_test_illumination.csv
|                |---face_landmarks_wflw_test_largepose.csv
|                |---face_landmarks_wflw_test_makeup.csv
|                |---face_landmarks_wflw_test_occlusion.csv
|                |---face_landmarks_wflw_train.csv
|
|--- images/
|--- models/
|--- options/
|--- plot/
|--- pylib/
|--- splits_prep/
|--- test/
|--- utils/
|
|--- face-layer-num-8-order-1-model-best.pth.tar
|  ...

```

Next type the following:
```bash
chmod +x *.sh
./scripts_dataset_splits_preparation.sh
```

### Training
```bash
./scripts_training.sh
```

### Evaluation of our pre-trained models

| Split | Directory |  LUVLi                | UGLLI                 |
|-------|-----------|-----------------------|-----------------------|
| 1     | run_108   | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1cCcC5bCPLT2zllgLlHrAgmkx5QvCs12z/view?usp=sharing) | - |
| 2     | run_109   | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1D1Y7R8-67dPn-n1_DwQQ04RiKyccdF80/view?usp=sharing) | - |
| 3     | run_507   | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1AilmsnZtpLirsfkgcbaHCc_ylrVWFTJh/view?usp=sharing) | - |
| 4     | run_1005  | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1fyxPnp3Dm3oy2IvhqSGEVNM_pTew8NY_/view?usp=sharing) |- |
| 5     | run_5004  | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/15L9Ss1zpai2FaK54tqCWGlm7dfOk7dVV/view?usp=sharing) |- |
| 1     | run_924   | -                     |  lr-0.00002-39.pth.tar |
| 2     | run_940   | -                     |  lr-0.00002-39.pth.tar |

Copy the pre-trained models to the ```abhinav_model_dir``` first. The directory structure should look like this:
```bash
./FaceAlignmentUncertainty/
|--- abhinav_model_dir/
|           |--- run_108
|           |       |--lr-0.00002-49.pth.tar
|           |
|           |--- run_109
|           |       |--lr-0.00002-49.pth.tar
|           |
|           |--- run_507
|           |       |--lr-0.00002-49.pth.tar
|           |
|           |--- run_1005
|           |       |--lr-0.00002-49.pth.tar
|           |
|           |--- run_5004
|           |       |--lr-0.00002-49.pth.tar
|  ...

```

Next type the following:
```bash
./scripts_evaluation.sh
```

In case you want to get our qualitative plots and also the transformed figures, type:
```bash
python plot/show_300W_images_overlaid_with_uncertainties.py --exp_id abhinav_model_dir/run_109_evaluate/ --laplacian
python plot/plot_uncertainties_in_transformed_space.py          -i run_109_evaluate/300W_test --laplacian
python plot/plot_residual_covariance_vs_predicted_covariance.py -i run_109_evaluate --laplacian
python plot/plot_histogram_smallest_eigen_value.py              -i run_109_evaluate --laplacian
```

***
<br/><br/>

## Extras
#### Training options

| Options                             |  Command                          |
|-------------------------------------|-----------------------------------|
| UGLLI                               | Default                           |
| LUVLi                               |```--laplacian --use_visibility``` |
| Post processing by ReLU             | ```--pp "relu"```                 |
| Aug scheme of Bulat et al, ICCV 2017| ```--bulat_aug```                 |
| Use slurm                           | ```--slurm```                     |

#### Preparing the JSONs from the splits
Images of splits are assumed to be in different directories with images and landmarks groundtruth of the same name with pts/mat extension. The bounding box ground truth for the first four face datasets is a mat file. The bounding boxes for other datasets is calculated by adding 5% noise to the tightest bounding box.

Go to the ```splits_prep``` directory and open ```config.txt```. 
```bash
input_folder_path    = ./bigdata1/zt53/data/face/          #the base path of all the images in the folder
annotations_path     = ./Bounding Boxes/                   #the bounding box groundtruths are in this folder 
num_keypoints        = 68                                  # assumed to be constant for a particular split

train_datasets_names = lfpw, helen, afw                    #train datasets name
train_folders        = lfpw/trainset, helen/trainset, afw  #folders path relative to input_folder_path
train_annotations    = bounding_boxes_lfpw_trainset.mat, bounding_boxes_helen_trainset.mat, bounding_boxes_afw.mat #paths relative to annotations_path

val_datasets_names   = lfpw, helen, ibug	               # val datasets name
val_folders          = lfpw/testset, helen/testset, ibug   #folders path relative to input_folder_path
val_annotations      = bounding_boxes_lfpw_testset.mat, bounding_boxes_helen_testset.mat, bounding_boxes_ibug.mat #paths relative to annotations_path

output_folder        = ./dataset                           # folder in which JSONs is to be stored
output_prefix        = normal_                             # prefix of the JSONs
```

We have **already placed** the bounding box initializations and the annotations of the train images of 300W dataset in the ```Bounding Box``` directory. In case, you are wondering the source of these annotations, you too can downloaded this from [here](https://ibug.doc.ic.ac.uk/media/uploads/competitions/bounding_boxes.zip).

#### Getting Split 1
```bash
python splits_prep/get_jsons_from_config.py -i splits_prep/config.txt
```

#### Getting Split 2
```bash
python splits_prep/get_jsons_from_config.py -i splits_prep/config_split2.txt
```

To get the JSONs for Menpo and Multi-PIE, type
```bash
python splits_prep/get_jsons_from_config.py -i splits_prep/config_menpo.txt
python splits_prep/get_jsons_from_config.py -i splits_prep/config_multi_pie.txt
```

<br/><br/>
***
## Contact
Feel free to drop an email to this address -
```abhinav3663@gmail.com```
