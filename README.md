# LUVLi and UGLLI Face Alignment
The code is officially available [here](https://github.com/merlresearch/LUVLi)

[LUVLi Face Alignment: Estimating Landmarks' Location, Uncertainty, and Visibility Likelihood](https://arxiv.org/pdf/2004.02980.pdf), [CVPR 2020](http://cvpr2020.thecvf.com/)

[[slides](https://docs.google.com/presentation/d/1jnvrBZWsX7PsAYFDBC-qQlQS8bLo_6PPEtB5gr7gMRU/edit)], [[1min_talk](https://www.youtube.com/watch?v=ZbKJvD_tO7Y)], [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Kumar_LUVLi_Face_Alignment_CVPR_2020_supplemental.zip)],[[demo](https://www.cse.msu.edu/computervision/kumar_marks_mou_wang_jones_cherian_akino_liu_feng_cvpr2020.mp4)]

[UGLLI Face Alignment: Estimating Uncertainty with Gaussian Log-Likelihood Loss](http://www.merl.com/publications/docs/TR2019-117.pdf#page=3), [ICCV](https://iccv2019.thecvf.com/) Workshops on Statistical Deep Learning in Computer Vision 2019 

[[slides](https://docs.google.com/presentation/d/1gHu5D0sb5dVvEyiqpJXry2oHqpfrMbmecJpCS8JDets/edit)], [[poster](https://docs.google.com/presentation/d/1aDgnO7aAvqbhvlAxoLitmXRS-Wj7tCfCbSt5nGY6LMU/edit)], [[news](https://www.merl.com/news/award-20191027-1293)], [[Best Oral Presentation Award](https://www.merl.com/public/img/news/photo-1293.jpg)]

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

### Evaluation of our pre-trained models

| Split | Name                 | Directory |  LUVLi                | UGLLI                 |
|-------|--------------------- |-----------|-----------------------|-----------------------|
| 1     | 300-W Split 1        | run_108   | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1cCcC5bCPLT2zllgLlHrAgmkx5QvCs12z/view?usp=sharing) | - |
| 2     | 300-W Split 2        | run_109   | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1D1Y7R8-67dPn-n1_DwQQ04RiKyccdF80/view?usp=sharing) | - |
| 3     | AFLW-19              | run_507   | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1AilmsnZtpLirsfkgcbaHCc_ylrVWFTJh/view?usp=sharing) | - |
| 4     | WFLW                 | run_1005  | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/1fyxPnp3Dm3oy2IvhqSGEVNM_pTew8NY_/view?usp=sharing) |- |
| 5     | MERL-RAV (AFLW_ours) | run_5004  | [lr-0.00002-49.pth.tar](https://drive.google.com/file/d/15L9Ss1zpai2FaK54tqCWGlm7dfOk7dVV/view?usp=sharing) |- |
| 1     | 300-W Split 1        | run_924   | -                     |  lr-0.00002-39.pth.tar |
| 2     | 300-W Split 2        | run_940   | -                     |  lr-0.00002-39.pth.tar |

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

