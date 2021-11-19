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
    n_sigma=n_sigma,

    save=True,
    #checkpoint_path='./fish_SE/full_set/best_iou_model.pth0.904403054939011',
    checkpoint_path='./fish_SE/full_set/best_seed_model.pth0.26857656598091123',

    min_pixel=64,
    threshold=0.5,
    model={
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [2 + n_sigma, 1],
            'input_channel': 3
        }
    },

    save_dir='./fish_SE/full_set/val_prediction/',
    dataset= {
        'name': 'mots_fish',
        'kwargs': {
            'root_dir': '/home/ubuntu/git-repo/VideoAmodal/dataset/data/',
            # 'type': 'train',
            'type': 'val',
            # 'size': 1000,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'type': (torch.FloatTensor, torch.LongTensor, torch.ByteTensor),
                    }
                },
            ]),
        },
        'batch_size': 128,
        'workers': 16
    },

    max_disparity=192.0,
    with_uv=True
)


def get_args():
    return copy.deepcopy(args)
