# hsn_v1 (HistoSegNetV1)

Code used for the HistoSegNet introduced in Chan *et al.*'s ICCV 2019 submission "HistoSegNet: Semantic Segmentation of Histological Tissue Type in Whole Slide Images". HistoSegNet produces pixel-level predictions of histological tissue type on whole-slide digital pathology images for both morphological and tissue types, using the same histological taxonomy as that introduced by Hosseini *et al.*'s CVPR 2019 paper "Atlas of Digital Pathology: A Generalized Hierarchical Histological Tissue Type-Annotated Database for Deep Learning".

![](/img.png)

We have provided the code here for easy installation and verification of the results presented in our paper submission.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

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

### Installing

TODO
```
cd 
pip install -r requirements.txt
pip install .
```

### Downloading data

TODO

### Changing paths

TODO

### Run the demo

To run on the tuning set:
```
python demo_01_segment_patches.py
```

To run on the GlaS set:
```
python demo_02_segment_glas_patches.py
```

To run on scanned WSI set (no overlap, not reported in paper):
```
python demo_03_segment_wsi.py
```

To run on scanned WSI set (with 25% overlap, as reported in paper):
```
python demo_03_segment_wsi_overlap.py
```

To run on the ADP set (not reported in paper):
```
python demo_05_segment_adp.py
```

To train and run Gaudet *et al.*'s segmentation network on the GlaS set:
```
python external/Gaudet_SegmentationNet/train.py
```