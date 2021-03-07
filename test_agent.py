from SCNav_agent import SCNavAgent
from utils.utils import d3_41_colors_rgb, ScalarMeanTracker

import torch

import random
import numpy as np
import argparse
import cv2
import shutil
import os
import copy
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm

from train_agent import parser
import quaternion as nq

from habitat.tasks.nav.object_nav_task import (
            ObjectGoal,
                ObjectGoalNavEpisode,
                    ObjectViewLocation,
                    )

def main():
    args = parser.parse_args()
    assert args.title in ['debug', 'random', 'randoms','single', 's', 's_c', 's_cu', 'ps', 'ps_cu',
            'ss_c_1100', 'ss_cu_1100', 'random-seg', 'random-seg-1000',
            's_cf', 'ss', 'ss_c', 'ss_cu',
            'random_new', "s_new", 's_c_new', 's_cu_new',
            'ss_new', 'ss_c_new', "ss_cu_new",
            'ss_1', 'ss_cu_1', 's_1', 's_cu_1',
            's_cu_2', 'ss_cu_2', 'ss_cu_2_1000', 'ss_cu_2_2500',
            's_c_3', 's_2', 's_cu_chal', 's_chal', 's_c_chal',
            'ss_3', 'ss_c_3', 'ss_cu_3', 'ss_4', 'ss_c_4', 'ss_cu_4', 'ss_5',
            'ss_c_5', 'ss_cu_5', 's_shortest', 's_c_shortest', 's_cu_shortest',
            's_chal_2', 's_c_chal_2', 's_cu_chal_2',
            's_chal_3', 's_c_chal_3', 's_cu_chal_3', 'fake_conf',
            's_cu_chal_5',
            'fake_conf_gt', 'discrete', 
            's_cu_chal_4', 'ss_cu_6', 'ss_cu_7', 'ss_cu_8',
            'ss_cu_att', 'ss_cu_9', 'fake_conf_2', 's_chal_final',
            's_cu_chal_final', 's_chal_final_2', 's_cu_chal_final_2',
            's_cu_chal_final_3', 'ss_cu_10', 's_cu_8', 'sscnav_old'], "Illegal exp id"
    new_eval = False
    fake_conf = False
    discrete = False
    att = False
    rc = False
    unconf = False
    full_map = False
    if args.title ==  'debug':
        args.Q_pretrained = "/local/crv/yiqing/result/s_2/100000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
        args.config_paths = './configs/agent_test.yaml'
    if args.title == 'random':
        args.Q_pretrained = ''
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
    if args.title == 'random_new':
        args.Q_pretrained = ''
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 'randoms':
        args.Q_pretrained = ""
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 'random-seg':
        args.Q_pretrained = ""
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 'random-seg-1000':
        args.Q_pretrained = ""
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = False
        args.conf = False
        args.pano = False
        args.seg_threshold = 1000
        
    elif args.title == 'single':
        args.Q_pretrained = '/local/crv/yiqing/result/single/60000.pth'
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 's':
        args.Q_pretrained = '/local/crv/yiqing/result/s/60000.pth'
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 's_c':
        args.Q_pretrained = '/local/crv/yiqing/result/s_c/60000.pth'
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = False
        args.pano = False
    elif args.title == 's_cf':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cf/33900.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
        args.preconf = True

    elif args.title == 's_cu':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
    elif args.title == 's_new':
        args.Q_pretrained = '/local/crv/yiqing/result/ss/1100.pth'
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 's_c_new':
        args.Q_pretrained = '/local/crv/yiqing/result/s_c/60000.pth'
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = False
        args.pano = False
    elif args.title == 's_cf':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cf/33900.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
        args.preconf = True

    elif args.title == 's_1':
        args.Q_pretrained = "/local/crv/yiqing/result/s_1/80000.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 's_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_2/100000.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 's_cu_1':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_1/80000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
    elif args.title == 's_c_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c_3/100000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = False
        args.pano = False
    elif args.title == 's_cu_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_2/100000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
    elif args.title == 's_chal_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_chal/60000.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_c_chal_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c_chal/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal_4':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_2/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_2/4_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal_5':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_3/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_4/14_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_chal_final':
        args.Q_pretrained = "/local/crv/yiqing/result/s_chal_final/100000.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal_final':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_final/100000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_5/20_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_chal_final_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_chal_final/60000.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal_final_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_final/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_5/20_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal_final_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_4/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_5/20_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_chal_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_chal/100000.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_c_chal_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c_chal/100000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal/100000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_chal':
        args.Q_pretrained = "/local/crv/yiqing/result/s_2/100000.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_c_chal':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c_3/100000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_chal':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_2/100000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_shortest':
        args.Q_pretrained = "/local/crv/yiqing/result/s_3/60000.pth"
        args.user_semantics = False
        args.shortest = True
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 's_c_shortest':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c_4/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = False
        args.pano = False
        args.shortest = True
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_shortest':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_3/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
        args.shortest = True
        new_eval = True
        args.max_step = 500
    elif args.title == 's_cu_new':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/5_fd.pth'
        args.pano = False
    elif args.title == 'ps':
        args.Q_pretrained = "/local/crv/yiqing/result/ps/.pth"
        args.user_semantics = False
        args.cmplt = False
        args.conf = False
        args.pano = True
    elif args.title == 'ps_cu':
        args.Q_pretrained = "/local/crv/yiqing/result/ps_cu/.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf/_cd.pth'
        args.pano = True
    elif args.title == 'ss':
        args.Q_pretrained = "/local/crv/yiqing/result/ss/1100.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 'ss_new':
        args.Q_pretrained = "/local/crv/yiqing/result/ss_new/6000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 'ss_c':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = "/local/crv/yiqing/result/train_seg/10.pth"
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.pano = False
    elif args.title == 'ss_c_new':
        args.Q_pretrained = "/local/crv/yiqing/result/ss_c_new/10000.pth"
        args.user_semantics = True
        args.seg_pretrained = "/local/crv/yiqing/result/train_seg/10.pth"
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.pano = False
    elif args.title == 'ss_c_1100':
        args.Q_pretrained = "/local/crv/yiqing/result/ss_c/1100.pth"
        args.user_semantics = True
        args.seg_pretrained = "/local/crv/yiqing/result/train_seg/10.pth"
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.pano = False

    elif args.title == 'ss_cu':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu/60000.pth"

        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
    elif args.title == 'ss_1':
        args.Q_pretrained = "/local/crv/yiqing/result/ss_1/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = False
        args.conf = False
        args.pano = False
    elif args.title == 'ss_cu_1':
        args.Q_pretrained = "/local/crv/yiqing/result/ss_cu_1/60000.pth"

        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
    elif args.title == 'ss_cu_new':
        args.Q_pretrained = "/local/crv/yiqing/result/ss_cu_new/.pth"

        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
    elif args.title == 'ss_cu_1100':
        args.Q_pretrained = "/local/crv/yiqing/result/ss_cu/1100.pth"

        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
    elif args.title == 'ss_cu_2':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_2/100000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
    elif args.title == 'ss_cu_2_1000':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_2/100000.pth"
        args.user_semantics = True
        args.seg_threshold = 1000
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
    elif args.title == 'ss_cu_2_2500':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_2/100000.pth"
        args.user_semantics = True
        args.seg_threshold = 2500
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
    elif args.title == 'ss_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_2/100000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_c_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c_3/100000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_cu_3':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_2/100000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_5':
        args.Q_pretrained = "/local/crv/yiqing/result/s_chal/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = False
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_c_5':
        args.Q_pretrained = "/local/crv/yiqing/result/s_c_chal/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_cu_5':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg/5_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_cu_6':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_2/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg_3/7_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_cu_7':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_3/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg_6/6_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_cu_8':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_4/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg_6/6_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'sscnav_old':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_4/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg_6/6_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
        full_map = True
        
    elif args.title == 's_cu_8':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_4/60000.pth"
        args.user_semantics = False
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized/17_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_4/14_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_cu_9':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_5/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg_6/6_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
        rc = True
    elif args.title == 'ss_cu_10':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_6/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg_7/13_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
    elif args.title == 'ss_cu_att':
        args.Q_pretrained = "/local/crv/yiqing/result/s_cu_chal_att/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
                = '/local/crv/yiqing/result/train_conf_seg_6/6_fd.pth'
        args.pano = False
        new_eval = True
        args.max_step = 500
        att = True
    elif args.title == 'fake_conf_gt':
        args.Q_pretrained = "/local/crv/yiqing/result/fake_conf/60000.pth"
        args.user_semantics = False
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.conf_pretrained = ''
        args.pano = False
        new_eval = True
        fake_conf = True
        args.max_step = 500
    elif args.title == 'fake_conf':
        args.Q_pretrained = "/local/crv/yiqing/result/fake_conf/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.conf_pretrained = ''
        args.pano = False
        new_eval = True
        fake_conf = True
        args.max_step = 500
    elif args.title == 'fake_conf_2':
        args.Q_pretrained = "/local/crv/yiqing/result/fake_conf_2/60000.pth"
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
                = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = False
        args.conf_pretrained = ''
        args.pano = False
        new_eval = True
        fake_conf = True
        unconf = True
        args.max_step = 500

    elif args.title == 'discrete':
        args.Q_pretrained = '/local/crv/yiqing/result/discrete/60000.pth'
        args.user_semantics = True
        args.seg_pretrained = '/local/crv/yiqing/result/train_seg/10.pth'
        args.seg_threshold = 5000
        args.cmplt = True
        args.cmplt_pretrained\
        = '/local/crv/yiqing/result/train_cmplt_resized_seg/16_cd.pth'
        args.conf = True
        args.conf_pretrained\
        = '/local/crv/yiqing/result/train_conf_seg_6/6_fd.pth'
        args.pano = False
        new_eval = True
        fake_conf = False
        discrete = True
        args.max_step = 500

    
    
    args.config_paths = './configs/agent_test.yaml'
    #if new_eval:
    #    args.height = 0.88
    #    args.config_paths='./configs/agent_test_habitat.yaml'
    args.save_dir = '/local/crv/yiqing/result/exps/'
    args.count = 150



    save_dir = os.path.join(args.save_dir, args.title)
    if os.path.exists(save_dir):
        assert False, "Dir exists!"
    os.makedirs(save_dir)

    configs = None
    args.load_json = '/local/crv/yiqing/SCNav/data/vis.json'
    if new_eval:
        args.load_json = '/local/crv/yiqing/SCNav/val.json'
    #args.load_json = '/local/crv/yiqing/result/exps/s_cu_2/vis.json'
    if args.load_json != "":
        with open(args.load_json, 'r') as f:
            configs = json.load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    agent = SCNavAgent(
            device = torch.device(args.device),
            #device = torch.device("cpu"),
            config_paths = args.config_paths,
            flip = args.flip,
            save_dir = save_dir,
            pano = bool(args.pano),
            #pano = False,
            user_semantics = bool(args.user_semantics),
            seg_pretrained = args.seg_pretrained,
            cmplt = bool(args.cmplt),
            cmplt_pretrained = args.cmplt_pretrained,
            conf = bool(args.conf),
            conf_pretrained = args.conf_pretrained,
            targets = args.targets,
            aggregate = bool(args.aggregate),
            #aggregate = True,
            memory_size = args.memory_size,
            num_channel = args.num_channel,
            success_threshold = args.success_threshold,
            collision_threshold = args.collision_threshold,
            ignore = args.ignore,
            training = False,
            Q_pretrained = args.Q_pretrained,
            #Q_pretrained = '../result/1.pth',
            offset = args.offset,
            floor_threshold = args.floor_threshold,
            lr = args.lr,
            momentum = args.momentum,
            weight_decay = args.weight_decay,
            gamma = args.gamma,
            batch_size = args.batch_size,
            #batch_size = 1,
            buffer_size = args.buffer_size,
            height = args.height,
            area_x = args.area_x,
            area_z = args.area_z,
            h = args.h,
            w = args.w,
            h_new = args.h_new,
            w_new = args.w_new,
            max_step = args.max_step,
            #max_step = 5,
            navigable_base = args.navigable_base,
            success_reward = args.success_reward,
            step_penalty = args.step_penalty,
            approach_reward = args.approach_reward,
            collision_penalty = args.collision_penalty,
            max_dist=args.max_dist,
           # max_dist = float("inf"),
            scene_types = args.scene_types,
            double_dqn = bool(args.double_dqn),
            TAU = args.TAU,
            preconf=args.preconf,
            seg_threshold=args.seg_threshold,
            min_dist=0.,
            current_position = None,
            new_eval=new_eval,
            shortest=args.shortest,
            fake_conf=fake_conf,
            discrete=discrete,
            att=att,
            rc=rc,
            unconf=unconf,
            full_map = full_map)
    targets = copy.deepcopy(agent.targets)
    train_scalars = {}
    for target in targets + ['all']:
        train_scalars[target] = ScalarMeanTracker()


    '''
    scene_id = agent.env.episodes[0].scene_id
    start_position = agent.env.episodes[0].start_position
    start_rotation = agent.env.episodes[0].start_rotation
    print(scene_id)
    print(start_position)
    #observations = agent.env.reset()
    

    
    #cv2.imwrite("test.png", observations['rgb'][..., [2, 1, 0]])
    scene_id = "/local/crv/yiqing/mp3d/v1/tasks/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
    start_position =  [11.16225, 0.17163, -2.66615] 
    start_rotation = [0, 0.13569, 0, 0.99075]
    episode = ObjectGoalNavEpisode(
            goals = [],
            episode_id='0',
            scene_id=scene_id,
            start_position=list(start_position),
            start_rotation=list(start_rotation)
            )
    agent.env.episode_iterator=iter([episode])
    observations = agent.env.reset()
    cv2.imwrite("test_2.png", observations['rgb'][..., [2, 1, 0]])
    assert False, "Pause"
    '''
    max_test_epoch = len(targets) * args.count if not new_eval else len(list(configs.keys()))
    print("start to evaluate on %s episodes..." % max_test_epoch)
    pbar = tqdm(total=max_test_epoch)
    path_records = {}
    for ep_id in range(max_test_epoch):
        path_records[str(ep_id)] = {}
        if not new_eval:
            current_target = targets[ep_id % len(targets)]
            if new_eval and current_target in ['table', 
                'sofa', 'door']:
                continue
            if configs is None:
                agent.reset(current_target)
                while agent.best_path_length < 1.25 or agent.best_path_length > 25.:
                    agent.reset(current_target)
            else:
                agent.reset_config(configs[str(ep_id)])
        else:
            config = configs[str(ep_id)]
         #   config['start_position'][1]\
          #          = str(float(config['start_position'][1]) + 2.)
            current_target = config['target']
            agent.reset_config(config)

        
            

        path_records[str(ep_id)]['target'] = current_target
        path_records[str(ep_id)]['scene_id'] = agent.episode.scene_id
        path_records[str(ep_id)]['start_position']\
        = [str(x) for x in list(agent.env.sim.get_agent_state().position)]
        path_records[str(ep_id)]['start_rotation']\
        = [str(x) for x in
                list(nq.as_float_array(agent.env.sim.get_agent_state().rotation))]
        path_records[str(ep_id)]['best_path_length']\
                = str(agent.best_path_length)
        path_records[str(ep_id)]['actions'] = []

        step_id = 0
        innerbar = tqdm(total=args.max_step)
        while not agent.done:
            
            cv2.imwrite(os.path.join(save_dir, "%s_%s_obs_%s.png"
                    % (ep_id, step_id, agent.target)),
                    d3_41_colors_rgb[torch.argmax(agent.state[0,
                        :agent.num_channel,...] , dim=0)])

            if agent.user_semantics:
                cv2.imwrite(os.path.join(save_dir, "%s_%s_seg_%s.png"
                    % (ep_id, step_id, agent.target)),
                    d3_41_colors_rgb[agent.raw_semantics])
            if agent.conf:
                cv2.imwrite(os.path.join(save_dir, "%s_%s_conf_obs_%s.png"
                    % (ep_id, step_id, agent.target)),
                    agent.conf_obs[0,0].numpy() * 255.)
                    
            cv2.imwrite(os.path.join(save_dir, "%s_%s_rgb_%s.png"
                    % (ep_id, step_id, agent.target)),
                    agent.get_observations()['rgb'][..., [2, 1, 0]])

            
            
            tmp_depth = agent.get_observations()['depth'][..., 0]
            depth = tmp_depth * agent.d2x[..., 0]
            tmp_depth[depth == 0.] = 255.
            tmp_depth[(depth >0.) & (depth <= 1.)] = 122
            tmp_depth[depth > 1.] = 255.

            '''
            targets = [agent.target] if agent.user_semantics else agent.target_objects
            semantic = agent.get_observations()['semantic'] if not\
            agent.user_semantics else agent.raw_semantics
            for i in range(depth.shape[0]):
                for j in range(depth.shape[1]):
                    if semantic[i, j] not in targets:
                        depth[i, j] = 0.
                    depth[i, j] = 255. if depth[i,j] > agent.success_threshold else 122.
                    
            '''

            cv2.imwrite(os.path.join(save_dir, '%s_%s_dep_%s.png'
                % (ep_id, step_id, agent.target)),
                tmp_depth)
            if args.title not in ['random', 'randoms', 'random-seg',
                    'random_new']:
                dreward = agent.step(args.end_eps)
            else:
                dreward = agent.step(1.)

            if agent.user_semantics:
                cv2.imwrite(os.path.join(save_dir, "%s_%s_next_seg_%s.png"
                    % (ep_id, step_id, agent.target)),
                    d3_41_colors_rgb[agent.raw_semantics])
            tmp_depth = agent.get_observations()['depth'][..., 0]
            depth = tmp_depth * agent.d2x[..., 0]
            tmp_depth[depth == 0.] = 255.
            tmp_depth[(depth >0.) & (depth <= 1.)] = 122
            tmp_depth[depth > 1.] = 255.

            '''
            targets = [agent.target] if agent.user_semantics else agent.target_objects
            semantic = agent.get_observations()['semantic'] if not\
            agent.user_semantics else agent.raw_semantics
            for i in range(depth.shape[0]):
                for j in range(depth.shape[1]):
                    if semantic[i, j] not in targets:
                        depth[i, j] = 0.
                    depth[i, j] = 255. if depth[i,j] > agent.success_threshold else 122.
                    
            '''

            cv2.imwrite(os.path.join(save_dir, '%s_%s_next_dep_%s.png'
                % (ep_id, step_id, agent.target)),
                tmp_depth)
            if not discrete:
                path_records[str(ep_id)]['actions'].append((int(agent.action[0]),
                int(agent.action[1])))

                cmap = agent.q_map.numpy()
                if np.max(cmap) == np.min(cmap):
                    cmap = np.zeros(cmap.shape).astype(np.uint8)
                else:
                    cmap = (cmap - np.min(cmap)) / (np.max(cmap)
                                - np.min(cmap)) * 255.
                    cmap = cmap.astype(np.uint8)
                cmap = cv2.cvtColor(cmap, cv2.COLOR_GRAY2BGR)
                cmap = cv2.applyColorMap(cmap, cv2.COLORMAP_JET)
                cv2.circle(cmap, (agent.action[1], agent.action[0]), 3, (20, 20, 20), -1)
                cv2.imwrite(os.path.join(save_dir, "%s_%s_q_map_%s.png"
                    % (ep_id, step_id, agent.target)),
                    cmap)
                
            while step_id < agent.eps_len:
                step_id += 1
                innerbar.update(1)

        
        path_records[str(ep_id)]['path_length'] = str(agent.path_length)
        path_records[str(ep_id)]['success'] = str(agent.success)
        path_records[str(ep_id)]['reward'] = str(agent.reward)
        path_records[str(ep_id)]['eps_len'] = str(agent.eps_len)
        path_records[str(ep_id)]['action_list'] = [str(t) for t in
                agent.action_list]
        innerbar.close()
        pbar.update(1)


        torch.cuda.empty_cache() 
        results = {
               "path_length": agent.path_length,
               "reward": agent.reward,
               "success": int(agent.success),
               "eps_len": agent.eps_len,
               "SPL": int(agent.success) * agent.best_path_length\
                       / max(agent.path_length, agent.best_path_length)}
      #  if agent.best_path_length >= 0.25 and agent.best_path_length < 1.25:
        train_scalars['all'].add_scalars(results)
        train_scalars[current_target].add_scalars(results)
      
      #  if agent.best_path_length >= 1.25:
      #      train_scalars['all']["L>=5"].add_scalars(results)
      #      train_scalars[current_target]["L>=5"].add_scalars(results)


    data = {}
    for cat in (targets + ['all']):
        data[cat] = train_scalars[cat].pop_and_reset()
    with open(os.path.join(save_dir, 'data.json'), 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
    with open(os.path.join(save_dir, "vis.json"), 'w') as fp:
        print(path_records)
        json.dump(path_records, fp)
    pbar.close()


if __name__ == "__main__":

    main()
