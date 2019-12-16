# HistoSegNet (V1)

## Introduction
![](/img.png)

We propose a new approach to Weakly-Supervised Semantic Segmentation (WSSS) with image label supervision for histopathology images, which trains on only patch-level annotations to infer pixel-level labels, called HistoSegNet (published in ICCV 2019). WSSS is useful for histopathology images because pixel-level prediction of tissue types facilitates further analysis by shape and texture, which can be indicative of disease.

![](/method.png)

Unlike other approaches, no additional training is required beyond training the classification network and only simple modifications are applied to Grad-CAM to perform segmentation. Our approach involves four stages:

1. Patch-level Classification CNN
2. Pixel-level Segmentation (i.e. Grad-CAM)
3. Inter-HTT Adjustments
4. Segmentation Post-Processing (i.e. dense CRF)

We have provided the code here for easy installation and verification of the results presented in our paper submission. Pretrained models and sample images are provided for the ADP tuning set and the GlaS dataset.

## Citing this repository

If you find this code useful in your research, please consider citing us:

        @InProceedings{chan2019histosegnet,
          author = {Chan, Lyndon and Hosseini, Mahdi S. and Rowsell, Corwyn and Plataniotis, Konstantinos N. and Damaskinos, Savvas},
          title = {HistoSegNet: Semantic Segmentation of Histological Tissue Type in Whole Slide Images},
          booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
          month = {October},
          year = {2019}
        }


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

Mandatory

* `python` (checked on 3.5)
* `keras` (checked on 2.2.4)
* `tensorflow` (checked on 1.13.1)
* `numpy` (checked on 1.16.2)
* `pydensecrf` (checked on 1.0rc3)
* `cv2` / `opencv-python` (checked on 3.4.4.19)
* `scipy` (checked on 1.2.0)
* `skimage` / `scikit-image` (checked on 0.14.2)

Optional

* `matplotlib` (checked on 3.0.2)
* `jupyter`

## Downloading data

Download `hsn_data.zip` (226 MB) from OneDrive containing pretrained models, ground-truth annotations, and images [here](https://drive.google.com/open?id=1jG1ojQKmvGjjjrRhCkaH0FDWM61tSgjL) and extract the contents into your `hsn_v1` directory (i.e. three folders `data`, `gt`, `img`).

## Run the demo batch scripts

To run on the ADP tuning set:
```
python demo_01_segment_patches.py
```

To run on the GlaS set:
```
python demo_02_segment_glas_patches.py
```

## Run the demo notebooks
Note: this requires Jupyter notebooks to be set up
* `demo_01_segment_patches.ipynb`
* `demo_02_segment_glas_patches.ipynb`