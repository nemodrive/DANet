###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from .base import BaseDataset


class CityscapesSegmentation(BaseDataset):
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19

    def __init__(self, root='../datasets', split='train',
                 mode=None, transform=None, target_transform=None, num_class=None, **kwargs):
        if num_class is not None:
            self.NUM_CLASS = num_class

        super(CityscapesSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks = _get_cityscapes_pairs(root, split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        mask = Image.open(self.masks[index])

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_cityscapes_pairs(folder, split='train'):
    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split('\t', line)
                imgpath = os.path.join(folder, ll_str[0].rstrip())
                maskpath = os.path.join(folder, ll_str[1].rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        split_f = os.path.join(folder, 'train_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)
    else:
        split_f = os.path.join(folder, 'trainval_fine.txt')
        img_paths, mask_paths = get_path_pairs(folder, split_f)

    return img_paths, mask_paths
