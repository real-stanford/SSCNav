
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

Navigation Experiments are controlled by argparse.

Here are some fields that matter most (feel free to check the rest in the code though!):
* **title:** experiment title, determines where to save the model & visualization & log.
* **config_paths:** configuration file for Habitat, ./configs/agent_train.yaml for training and ./configs/agent_test.yaml for testing.
* **user_semantics:** True -> use ACNet segmentation, groundtruth otherwise.
* **cmplt & cmplt_pretrained:** if semantic completion model is used, must set former to be True and optionally assign initial weight for the model.
* **conf & conf_pretrained:** if confidence model is used, must set former to be True and optionally assign initial weight for the model.
* **save_dir:** place to store model and visualization.
* **log_dir:** place to store Tensorboard log.
* **load_json:** dir to the JSON file which stores the evaluation episodes.
```
usage: train_agent.py [-h] --title TITLE [--seed SEED] [--device DEVICE]
                      [--config_paths CONFIG_PATHS] [--flip FLIP]
                      [--seg_threshold SEG_THRESHOLD] [--pano]
                      [--user_semantics] [--seg_pretrained SEG_PRETRAINED]
                      [--cmplt] [--cmplt_pretrained CMPLT_PRETRAINED] [--conf]
                      [--conf_pretrained CONF_PRETRAINED] [--targets TARGETS]
                      [--aggregate] [--memory_size MEMORY_SIZE]
                      [--num_channel NUM_CHANNEL]
                      [--success_threshold SUCCESS_THRESHOLD]
                      [--collision_threshold COLLISION_THRESHOLD]
                      [--ignore IGNORE] [--Q_pretrained Q_PRETRAINED]
                      [--offset OFFSET] [--floor_threshold FLOOR_THRESHOLD]
                      [--lr LR] [--momentum MOMENTUM]
                      [--weight_decay WEIGHT_DECAY] [--gamma GAMMA]
                      [--batch_size BATCH_SIZE] [--buffer_size BUFFER_SIZE]
                      [--height HEIGHT] [--area_x AREA_X] [--area_z AREA_Z]
                      [--h H] [--w W] [--h_new H_NEW] [--w_new W_NEW]
                      [--max_step MAX_STEP] [--navigable_base NAVIGABLE_BASE]
                      [--max_transition MAX_TRANSITION]
                      [--start_replay START_REPLAY]
                      [--update_target UPDATE_TARGET] [--start_eps START_EPS]
                      [--end_eps END_EPS] [--fix_transition FIX_TRANSITION]
                      [--success_reward SUCCESS_REWARD]
                      [--step_penalty STEP_PENALTY]
                      [--approach_reward APPROACH_REWARD]
                      [--collision_penalty COLLISION_PENALTY]
                      [--save_dir SAVE_DIR] [--save_interval SAVE_INTERVAL]
                      [--log_dir LOG_DIR] [--train_thin TRAIN_THIN]
                      [--loss_thin LOSS_THIN] [--train_vis TRAIN_VIS]
                      [--scene_types SCENE_TYPES] [--double_dqn] [--TAU TAU]
                      [--soft_update] [--count COUNT] [--preconf]
                      [--load_json LOAD_JSON] [--checkpoint CHECKPOINT]
                      [--shortest] [--tsplit TSPLIT] [--new_eval]
                      [--fake_conf] [--discrete] [--att] [--rc] [--unconf]
                      [--full_map]


```
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
