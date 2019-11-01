import torch.utils.data as data
import sys
import os

sys.path.append(os.path.abspath('../../'))
from train.data.image_loader import default_image_loader as image_loader

class BaseDataset(data.Dataset):
    '''
    base dataset for loading datasets
    '''
    def __init__(self, root, transform=None):
        '''
        args:
            root: the path to dataset
        '''
        if root == '':
            raise Exception('the dataset path is not setup.')
        self.root = root
        self.all_seqs = []
        self.transform = transform
    
    def __len__(self):
        return len(self.all_seqs)

    def __getitem__(self, idx):
        '''
        function: support the indexing such that dataset[i] can be used to get ith sample
        args:
            idx: the index of frames
        '''
        return None
        
    def is_video_sequence(self):
        return True

    def get_name(self):
        '''
        get the name of dataset
        '''
        return NotImplementedError

              


if __name__ == "__main__":
    pass