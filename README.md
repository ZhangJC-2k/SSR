# SSR
This is a repo for our work: "**[Improving Spectral Snapshot Reconstruction with Spatial-Spectral Rectification](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Improving_Spectral_Snapshot_Reconstruction_with_Spectral-Spatial_Rectification_CVPR_2024_paper.html)**".

#### News
Our work has been accepted by CVPR, codes and results are coming soon (July or August).

The codes and pre-trained weights have been released. More details and instructions will be continuously updated.

### 1. Environment Requirements
```shell
Python>=3.6
scipy
numpy
einops
```

### 2. Train:

Download the cave dataset of [MST series](https://github.com/caiyuanhao1998/MST) from [Baidu disk](https://pan.baidu.com/share/init?surl=X_uXxgyO-mslnCTn4ioyNQ?pwd=fo0q)`code:fo0q` or [here](https://pan.baidu.com/s/1gyIOfmUWKrjntKobUjwTjw?pwd=lup6), put the dataset into the corresponding folder "SSR/CAVE_1024_28/" as the following form:

	|--CAVE_1024_28
        |--scene1.mat
        |--scene2.mat
        ：
        |--scene205.mat
        |--train_list.txt
Then run the following command
```shell
cd SSR
python Train.py
```

### 3. Test:

Download the test dataset from [here](https://pan.baidu.com/s/1KqMo3CY8LU9HRU2Lak9yfQ?pwd=c0a2), put the dataset into the corresponding folder "SSR/Test_data/" as the following form:

	|--Test_data
        |--scene01.mat
        |--scene02.mat
        ：
        |--scene10.mat
        |--test_list.txt
Then run the following command
```shell
cd SSR
python Test.py
```
For testing pre-trained models, run the following command
```
python Test_pretrain.py
```
Finally, run 'cal_psnr_ssim.m' in Matlab to get the performance metrics.

### Citation
If this repo helps you, please consider citing our work:


```shell
@InProceedings{SSR,
    author    = {Zhang, Jiancheng and Zeng, Haijin and Chen, Yongyong and Yu, Dengxiu and Zhao, Yin-Ping},
    title     = {Improving Spectral Snapshot Reconstruction with Spectral-Spatial Rectification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {25817-25826}
}
```
