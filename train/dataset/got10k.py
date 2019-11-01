import os
import sys
import torch
import numpy as np
import csv
import pandas as pd
from collections import OrderedDict
from base_dataset import BaseDataset

sys.path.append(os.path.abspath('../../'))

from train.data.image_loader import default_image_loader as image_loader

import yaml
from train.setting.env_settings import env_settings, get_got10k_root, get_got10k_train, get_got10k_val, get_got10k_test

class Got10k(BaseDataset):
    def __init__(self, root=None, transfrom=None, split=None, seq_ids=None):
        '''
        args:
            root: the path to got10k training data.
            image_loader: the function for loadigf images.
            split: "train" or "val" or "test".
            seq_ids: List including the ids of videos for training
        '''
        if root == None:
            self.root = os.path.join(get_got10k_root())

        if split == "train":
            self.seq_root = os.path.join(self.root, get_got10k_train())            
        elif split == "val":
            self.seq_root = os.path.join(self.root, get_got10k_val())
        elif split == "test":
            self.seq_root = os.path.join(self.root, get_got10k_test())
        elif split == None:
            raise Exception('please select your dataset, "train" or "val" or "test".')
        
        
        super(Got10k, self).__init__(root=root, transform=transform)

        self.seq_ids = seq_ids
        # if seq_ids is not None, we should select all sequence in seq_list
        self.all_seqs = self._get_seq_list()        
        self.seq_list = self._set_seq_list()
        self.seq_meta_info = self._load_meta_info()

        # seq_ids

    def _set_seq_list(self):
        '''
        set the sequence list
        '''
        if self.seq_ids is None:
            seq_list = self.all_seqs
        else:
            seq_list = [self.all_seqs[i] for i in self.seq_ids]

        return seq_list  

    def _get_seq_list(self):
        '''
        get all folders inside the root
        '''
        seq_list_path = os.path.join(self.seq_root, 'list.txt')
        with open(seq_list_path) as f:
            dir_list = list(csv.reader(f))
        
        dir_list = [dir_name[0] for dir_name in dir_list]

        return dir_list

    def _get_name(self):
        return "got10k"

    def _get_seq_path(self, seq_id):
        return os.path.join(self.seq_root, self.seq_list[seq_id])

    def _load_meta_info(self):
        seq_meta_info = {s: self._read_meta(os.path.join(self.seq_root, s)) for s in self.seq_list}
        return seq_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _read_anno(self, seq_path):
        anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, frame_ids, seq_id, anno=None):
        seq_path = self._get_seq_path(seq_id)
        obj_meta = self.seq_meta_info[self.seq_list[seq_id]]

        frame_list = [self._get_frame(seq_path, fra_id) for fra_id in frame_ids]

        if anno is None:
            anno = self._read_anno(seq_path)
        
        anno_frames = [anno[fra_id, :] for fra_idd in frame_ids]

        return frame_list, anno_frames, obj_meta

if __name__ == "__main__":
    got10k = Got10k(split="train")
    # print(got10k._get_seq_list())