
# SSCNav: Confidence-Aware Semantic Scene Completion for Visual Semantic Navigation

[Yiqing Liang](https://yiqing-liang.netlify.app/)<sup>1</sup>,
[Boyuan Chen](http://www.cs.columbia.edu/~bchen/)<sup>1</sup>,
[Shuran Song](https://www.cs.columbia.edu/~shurans/)<sup>1</sup>
<br>
<sup>1</sup>Columbia University
<br>
[ICRA 2021](http://www.icra2021.org/)

### [Project Page](https://sscnav.cs.columbia.edu/) | [Video](https://youtu.be/tfBbdGS72zg) | [arXiv](https://arxiv.org/pdf/2012.04512)

## Overview
PyTorch implementation for paper "SSCNav: Confidence-Aware Semantic Scene Completion for Visual Semantic Navigation".

![teaser](https://sscnav.cs.columbia.edu/images/teaser.gif)

## Content

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Models](#pretrained-models)
- [Usage](#usage)

## Installation
Recommend working on conda environment, Python 3.7+
1. Download Matterport3d dataset following instruction [here](https://github.com/niessner/Matterport).
2. ```conda create -p /path/to/your/env python=3.7```
3. Install [PyTorch](https://pytorch.org/).
4. Install Habitat environment following instruction [here](https://github.com/facebookresearch/habitat-lab).
5. Clone and navigate to this repository.
6. ```conda install --yes --file requirements.txt --channel default --channel menpo --channel conda-forge```

## Data Preparation
For training and testing Semantic Scene Completion Model and Confidence Model, download [data](https://sscnav.cs.columbia.edu/download/SceneCompletionData.zip) (634M):
```
SceneCompletionData.zip
    |-- data_gen_1 \
    |-- data_gen_2 - training data for groundtruth semantic segmentation
    |-- data_gen_3 /
    |-- test_data_gen_1 \
    |-- test_data_gen_2 - testing data for groundtruth semantic segmentation
    |-- test_data_gen_3 /
    |-- train - training data for ACNet semantic segmentation output
    |-- test - testing data for ACNet semantic segmentation output
```
## Pretrained Models
Download our pretrained models [here](https://sscnav.cs.columbia.edu/download/pretrained.zip) (1.2G).
Model | Description
--- | --- 
```cmplt.pth``` | semantic scene completion model weight for groundtruth semantic segmentation
```conf.pth``` | confidence model weight for groundtruth semantic segmentation
```cmplt_seg.pth``` | semantic scene completion model weight for ACNet semantic segmentation output
```conf_seg.pth``` | confidence model weight for ACNet semantic segmentation output
```final_seg.pth``` | ACNet semantic segmentation model weight
```final_Q.pth``` | SSCNav navigation model weight

## Usage

### Semantic Scene Completion

The experiment settings are controlled by file ```configs/config.json```. Each experiment is assigned a unique key in the file. Check the JSON file for details.

**Note:** data are provided in the form of list.
For example:
```
"root_dirs": ["../resized_buffer/data_gen_1", "../resized_buffer/data_gen_2", "../resized_buffer/data_gen_3"]    
"test_dirs": ["../resized_buffer/test_data_gen_1", "../resized_buffer/test_data_gen_2", "../resized_buffer/test_data_gen_3"]
```

#### Training & Evaluating a Semantic Scene Completion Model
The command to train the provided model for groundtruth semantic segmentation:
```
CUDA_VISIBLE_DEVICES=? python train_cmplt.py train_cmplt_resized
```
The command to train the provided model for ACNet semantic segmentation output:
```
CUDA_VISIBLE_DEVICES=? python train_cmplt.py train_cmplt_resized_seg
```
#### Training & Evaluating a Confidence Model
The command to train the provided model for groundtruth semantic segmentation:
```
CUDA_VISIBLE_DEVICES=? python train_conf.py train_conf_4
```
The command to train the provided model for ACNet semantic segmentation output:
```
CUDA_VISIBLE_DEVICES=? python train_conf.py train_conf_seg_6
```
### Navigation

#### Training a Navigation Model
```
CUDA_VISIBLE_DEVICES=? python train_agent.py --user_semantics False --cmplt True 
    --cmplt_pretrained /local/crv/yiqing/result/train_cmplt_resized/17_cd.pth
    --conf True --conf_pretrained /local/crv/yiqing/result/train_conf_4/14_fd.pth
                
```
#### Evaluating a Navigation Model

## BibTeX
```
@inproceedings{liang2021sscnav,
    title={SSCNav: Confidence-Aware Semantic Scene Completion for Visual Semantic Navigation},
    author={Liang, Yiqing and Chen, Boyuan and Song, Shuran},
    booktitle = {Proc. of The International Conference in Robotics and Automation (ICRA)},
    year={2021}
}
```

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.


## Acknowledgement

This work was supported in part by the Amazon Research Award and Columbia School of Engineering.
