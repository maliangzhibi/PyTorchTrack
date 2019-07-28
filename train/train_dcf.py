"""
for training dcf tracker
"""

from dataset.datasetvid import VID
from model.DCFNet import DCFNet

import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torch.backends.cudnn as cudnn
import numpy as np
import time
import utils.dcf_utils as dcf_util
import trainer.dcf_train as dcf_train
from data.get_train_data import get_train_data
from data.get_val_data import get_val_data

def main():
    """
    main(): no parameters
    train for the tracker
    """
    cudnn.benchmark = True
    # configuration for dcf tracker
    config = dcf_util.TrackerConfig()
    model = DCFNet(config=config)
    
    # convert the model into cuda datatype
    model.cuda()

    # get the num of gpu
    gpu_num = dcf_util.get_gpu_num()
    
    # compute the data by paralleling
    if gpu_num > 1:
        model = nn.DataParallel(model, list(range(gpu_num))).cuda()
    
    # define the loss
    criterion = nn.MSELoss(size_average=False).cuda()

    # define optimizer
    best_loss = 1e6
    args = dcf_util.train_setting()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, 
                                momentum=args.momentum,
                                weight_decay = args.weight_decay)
    
    # get the target
    target = torch.Tensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(args.batch_size=gpu_num, 1, 1, 1)

    flag, checkpoint = dcf_util.check_checkpoint()
    # check whether use the checkpoint
    if flag == True:
        args.start_epoch=checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('==>loaded checkpoint [{}] (epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        print("don't use the checkpoint fofr training.")
    
    # training data
    crop_base_path = join('dataset', 'crop_{:d}_{:1.1f}'.format(args.input_sz, args.padding))
    if not isdir(crop_base_path):
        print('please run gen_training_data.py --output_size {:d} --padding {:.1f}!'.format(args.input_sz, args.padding))
        exit()

    save_path = join(args.save, 'crop_{:d}_{:1.1f}'.format(args.input_sz, args.padding))
    if not isdir(save_path):
        makedirs(save_path)
    
    train_loader = get_train_data(gpu_num, crop_base_path, args)
    val_loader = get_val_data(gpu_num, crop_base_path, args)

    for epoch in range(args.start_epoch, args.eopchs):
        dcf_util.adjust_learning_rate(optimizer, epoch)

        # train for epoch
        dcf_train.train(train_loader, target, model, criterion, optimizer)
        loss = dcf_train.validate(val_loader, target, model, criterion)

        # remember best_loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        dcf_util.save_checkpoint(
            {'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            }, 
            is_best
        )


if __name__ == "__main__":

    print('for training args===>:', train_setting())
    best_loss = 1e6

    
