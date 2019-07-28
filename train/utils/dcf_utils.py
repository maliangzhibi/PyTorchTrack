import argparse
import torch
import torch.nn as nn
from torch.utils.data import dataloader
import numpy as np
from dataset import datasetvid
import os
import shutil
import time


def train_setting():
    parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
    parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
    parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
    parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-5)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    args = parser.parse_args()
    return args

def gauss_shaped_labels(sigma, sz):
    '''
    for obtaining the gauss_shaped_labels to the target region
    '''
    # the goal of np.arange(1, sz[0]+1) subtracted sz[0]/1 is to set the target center at (0,0) position of meshgrid(...) 
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0])/2), np.arange(1, sz[0]+1) - np.floor(float(sz[1])/2))
    d = x**2 + y**2
    g = np.exp(-0.5/(2*sigma)*d)
    g = np.roll(g, int(-np.floor(float(sz[0])/2.)+1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1])/2.)+1), axis=1)
    
    return g.astype(np.float32)

class TrackerConfig(object):
    crop_sz=125
    output_sz = 121

    lambda0 = 1e-6
    padding = 2.0
    output_sigma_factor = 0.1
    output_sigma = crop_sz/(1+padding)*output_sigma_factor
    
    y = gauss_shaped_labels(output_sigma, [output_sz, output_sz])
    # gauss_shaped_label中将中心滚动到四个角点上可能是因为傅里叶变换之后的时域上中心和空域的中心相反
    yf = torch.rfft(torch.Tensor(y).view(1,1,output_sz, output_sz).cuda(), signal_ndim=2)


def get_gpu_num():
    '''
    return gpu num
    '''
    return torch.cuda.device_count()

def get_data(data_path = None):
    '''
    return data_train_loader and data_val_loader
    '''
    if data_path is None:
        raise Exception('there is no path!')
    
    args = train_setting()
    data_train = datasetvid.VID(root=data_path, train=True, range=args.range)
    data_val = datasetvid.VID(root=data_path, train=False, range=args.range)

    data_train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.batch_size*get_gpu_num(),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True 
    )
    data_val_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size = args.batch_size*get_gpu_num(),
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    return data_train_loader, data_val_loader

def adjust_learning_rate(optimizer, epoch):
    '''
    learning_schedule for dcf by customized
    '''
    args = train_setting()
    lr = np.logspace(-2, -5, num=args.epochs)[epoch]
    # set each learning_rate of param_group to lr
    for param_group in optimizer.param_groups:
        param_group['lr']=lr

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count

def save_checkpoint(state, is_best, file_name=os.path.join(save_path, 'checkpoint.pth.tar')):
    """
    store the checkpoint to save_path
    """
    torch.save(state, file_name)
    # if the checkpoint is the best, store it into save_path  
    if is_best:
        shutil.copyfile(file_name, os.path.join(save_path, 'model_best.pth.tar'))
    

def check_checkpoint():
    """
    start_checkpoint: if you start the checkpoint, you will not train
    from scrach, you will train from the checkpoint.
    return the checkpint
    """
    args = train_setting()
    if args.resume:
        if os.path.isfile(args.resume):
            print('-loading checkpoint [{}]'.format(args.resume))
            
            checkpoint = troch.load(args.resume)
            return True, checkpoint
        else:
            print('no checkpoint found {}'.format(args.resume))
            return False, None
            


