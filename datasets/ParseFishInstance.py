"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import cv2
import os
import pycocotools.mask as maskUtils
import multiprocessing
from PIL import Image
import pickle
import json
from pycocotools import mask as mutils
from pycocotools import _mask as coco_mask

num_frames = 128
height = 320
width = 480

data_root = "/home/ubuntu/git-repo/VideoAmodal/dataset/data/"
data_split = "train"

def mkdir_if_no(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def decode2binarymask(masks):
    mask = coco_mask.decode(masks)
    binary_masks = mask.astype('bool') # (320,480,128)
    binary_masks = binary_masks.transpose(2,0,1)
    return binary_masks

def plot_vid_to_imgs(v_id):
    inmodal = np.zeros((num_frames, height, width), dtype=np.int16)
    V_ID = "%05d" % v_id
    # Read all object
    obj_root = os.path.join(data_root, data_split+'_data', data_split)
    instance_root = os.path.join(data_root, data_split+'_data', data_split+'_instances')
    mkdir_if_no(instance_root)
    V_ID_instance_root = os.path.join(instance_root, V_ID)
    mkdir_if_no(V_ID_instance_root)
    object_file = os.path.join(obj_root, V_ID, 'objects.json')
    # update inmodal for each object
    obj_data = json.load(open(object_file,))
    obj_base = 26000
    for obj in obj_data:
        obj_id = obj['id']
        masks = decode2binarymask(obj['masks'])
        for frame_id in range(num_frames):
            inmodal_ann_mask = masks[frame_id]
            inmodal[frame_id, inmodal_ann_mask > 0] = obj_base + obj_id
    # save imodal
    for frame_id in range(num_frames):
        inmodal_path = os.path.join(V_ID_instance_root, "%05d.png" % frame_id)
        Image.fromarray(inmodal[frame_id]).save(inmodal_path)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=32)
    vids = list(range(10000))
    results = pool.map(plot_vid_to_imgs, vids)
    pool.close()

