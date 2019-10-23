import torch.utils.data as data
import image_loader

class BaseDataset(data.Dataset):
    '''
    base dataset for loading datasets
    '''
    def __init__(self, root, image_loader=image_loader, transform=None):
        '''
        args:
            root: the path to dataset
            image_loader: the function for reading images
        '''
        if root == '':
            raise Exception('the dataset path is not setup.')
        self.root = root
        self.image_loader = image_loader
        self.seq_list = []
        self.transform = transform
    
    def __len__(self):
        return len(seq_list)

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