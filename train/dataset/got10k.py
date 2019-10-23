import os
import sys
import torch
import numpy as np
import csv
import pandas as pd
from collections import OrderedDict
from base_dataset import BaseDataset
import image_loader

import yaml

sys.path.append(os.path.abspath('..'))
from setting.env_settings import env_settings, get_got10k_root, get_got10k_train, get_got10k_val, get_got10k_test

class Got10k(BaseDataset):
    def __init__(self, root=None, image_loader=image_loader, transfrom=None, split=None, seq_ids=None):
        '''
        args:
            root: the path to got10k training data.
            image_loader: the function for loadigf images.
            split: "train" or "val" or "test".
            seq_ids: List including the ids of videos for training
        '''

        if split=="train":
            root = get_got10k_train()
            if seq_ids is None:
                self.file_path = os.path.join(root, 'list.txt')
        elif split=="val":
            root = get_got10k_val()
            if seq_ids is None:
                self.file_name = os.path.join(root, 'list.txt')
        elif split=="test":
            root = get_got10k_test()
            if seq_ids is None:
                self.file_path = os.path.join(root, 'list.txt')
        elif split==None:
            raise Exception('please select your dataset, "train" or "val" or "test".')
        
        # print(root)
        super(Got10k, self).__init__(root=root, image_loader=image_loader)

        self.seq_list = self._get_seq_list()

        seq_ids

    def set_seq_ids(self):
        '''
        set the sequence ids
        '''
        

    def _get_seq_list(self):
        '''
        get all folders inside the root
        '''
        train_sequence_list_path = os.path.join(self.root, 'list.txt')
        with open(train_sequence_list_path) as f:
            dir_list = list(csv.reader(f))
        
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list



if __name__ == "__main__":
    got10k = Got10k()
    got10k._get_seq_list()