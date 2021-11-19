"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms
from config import *

n_sigma=2
args = dict(

    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='./fish_SE/full_set',
    # resume_path='./pointTrack_weights/best_seed_model.pthCar',
    # resume_path='./mots_finetune_car2/checkpoint.pth',
    # resume_path = './fish_SE/exp0/best_iou_model.pth0.8987708287747176',

    train_dataset = {
        'name': 'mots_fish',
        'kwargs': {
            'root_dir': '/home/ubuntu/git-repo/VideoAmodal/dataset/data/',
            'type': 'train',
            #'size': 7000,
            'transform': my_transforms.get_transform([
                {
                    'name': 'AdjustBrightness',
                    'opts': {}
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
                {
                    'name': 'Flip',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                    }
                },
            ]),
        },
        #'batch_size': 4,
        #'workers': 0,
        'batch_size': 64,
        'workers': 8
    },

    val_dataset = {
        'name': 'mots_fish',
        'kwargs': {
            'root_dir': '/home/ubuntu/git-repo/VideoAmodal/dataset/data/',
            'type': 'val',
            # 'size': 500,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 512,
        'workers': 32,
        #'batch_size': 2,
        #'workers': 1
    },

    model={
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1],
            'input_channel': 3
        }
    },

    lr=5e-4,
    milestones=[5],
    n_epochs=5,
    start_epoch=1,
    max_disparity=192.0,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': n_sigma,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
    loss_type='MOTSSeg2Loss'
)


def get_args():
    return copy.deepcopy(args)
