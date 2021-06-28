

# SMOKED

## Requirements
All codes are tested under the following environment:
*   Ubuntu 16.04
*   Python 3.7
*   Pytorch 1.3.1
*   CUDA 10.0

## Dataset
We train and test our model on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Please first download the dataset and organize it as following structure:
```
datasets
└──kitti
   │──training
   │    ├──calib 
   │    ├──label_2 
   │    ├──image_2
   │    └──ImageSets
   └──testing
        ├──calib 
        ├──image_2
        └──ImageSets
```

## Setup
1. Build codes:

```
python3 setup.py build develop
```

2. Link to dataset directory (For only training, if you just want to run demo datasets, you can skip this) :

```
mkdir datasets
ln -s /path_to_kitti_dataset datasets/kitti
```

## Getting started
First check the config file under `configs/`. 

For single GPU training, simply run:
```
python3 tools/plain_train_net.py --config-file "configs/smoke_AddDepth.yaml"
```

We currently only support single GPU testing:
```
python3 tools/plain_train_net.py --eval-only --config-file "configs/smoke_AddDepth.yaml"
```

For running demo video:

```
wget -O datasets/kitti/demo.zip https://www.dropbox.com/s/eyugn1rm0afd2sa/demo.zip?dl=1
unzip datasets/kitti/demo.zip -d datasets/kitti/
python3 tools/plain_train_net.py --eval-only --config-file "configs/smoke_demo.yaml"
python3 visualize_3DBox.py
python3 imgs2video.py
```

