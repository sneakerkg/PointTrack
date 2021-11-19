"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import shutil
import time
from config import *

os.chdir(rootDir)

from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np

import torch
from config_mots import *
from criterions.mots_seg_loss import *
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer
from file_utils import remove_key_word
from random import random

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from PIL import Image

from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True
config_name = sys.argv[1]

args = eval(config_name).get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")
# clustering
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])
model = model.to(device)

# set criterion
criterion = eval(args['loss_type'])(**args['loss_opts'])
criterion = criterion.to(device)

# resume
start_epoch = 0
best_iou = 0
best_seed = 10
max_disparity = args['max_disparity']
if 'resume_path' in args.keys() and args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])

    if 'start_epoch' in args.keys():
        start_epoch = args['start_epoch']
    elif 'epoch' in state.keys():
        start_epoch = state['epoch'] + 1
    else:
        start_epoch = 1
    # best_iou = state['best_iou']
    for kk in state.keys():
        if 'state_dict' in kk:
            state_dict_key = kk
            break
    new_state_dict = state[state_dict_key]

    has_module_prefix = False
    for k, v in new_state_dict.items():
        if 'module.' in k:
            has_module_prefix = True
            break


    for k, v in model.state_dict().items():
        if 'module.' not in k and has_module_prefix:
            new_state_dict[k] = new_state_dict['module.'+k]
            new_state_dict.pop('module.'+k)

    if not 'state_dict_keywords' in args.keys():
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except:
            print('resume checkpoint with strict False')
            model.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = remove_key_word(state[state_dict_key], args['state_dict_keywords'])
        model.load_state_dict(new_state_dict, strict=False)
        print('resume checkpoint with strict False')
    try:
        logger.data = state['logger_data']
    except:
        pass

def visualization(ims, instance_maps, batch_id):
    for img_id in range(len(instance_maps)):
        canvas = np.zeros((ims[img_id].shape[1], ims[img_id].shape[2], 3)).astype(np.uint8)
        cNorm  = colors.Normalize(vmin=0, vmax=len(instance_maps[img_id]))
        jet = cm = plt.get_cmap('jet')
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        for i_id, instance in enumerate(instance_maps[img_id]):
            colorVal = scalarMap.to_rgba(i_id)
            canvas[instance.squeeze()>0,:] = (np.array(colorVal[:3]) * 255).astype(np.uint8)
        final_im = (ims[img_id].cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        im_path = './vis/batch_' + str(batch_id) + '_' + str(img_id) + '.png'
        print (im_path)
        Image.fromarray(np.concatenate([final_im, canvas], axis=1)).save(im_path)
    exit (0)






def val():
    # define meters
    loss_meter, iou_meter, loss_seed_meter = AverageMeter(), AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(tqdm(val_dataset_it)):
            ims = sample['image'].to(device)
            instances = sample['instance'].squeeze(1).to(device)
            class_labels = sample['label'].squeeze(1).to(device)

            output = model(ims)
            loss, seed_loss, instance_maps = criterion(output, instances, class_labels, **args['loss_w'], iou=True,
                                        iou_meter=iou_meter, show_seed=True, return_pred=True)


            print (len(instance_maps))
            visualization(ims, instance_maps, i)
            exit (0)

            loss = loss.mean()
            seed_loss = seed_loss.mean()

            loss_meter.update(loss.item())
            loss_seed_meter.update(seed_loss.item())
            print (iou_meter.avg)

    return loss_meter.avg, iou_meter.avg, loss_seed_meter.avg




val_loss, val_iou, val_seed_loss = val()
print('===> val loss: {:.4f}, val iou: {:.4f}, val seed: {:.4f}'.format(val_loss, val_iou, val_seed_loss))
