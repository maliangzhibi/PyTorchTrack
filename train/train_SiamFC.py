from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import GOT10k, ImageNetVID
from dataset.pairwise import Pairwise
from trainer.SiamFC_train import TrackerSiamFC

if __name__ == '__main__':
    # dataset
    name = 'GOT-10K'
    assert name in ['VID', 'GOT-10K']
    if name == 'GOT-10K':
        root_dir = 'G:/got10k/'
        seq_dataset = GOT10k(root_dir, subset='train')
    # elif name == 'VID':
    #     root_dir = 'data/ILSVRC'
    #     seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))
    else:
        print('no got10 dataset.')
        exit()
    pair_dataset = Pairwise(seq_dataset)

    # set data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        pair_dataset, batch_size = 32, shuffle=True, pin_memory=cuda, drop_last=True, num_workers=0 
    )

    # tracker
    tracker = TrackerSiamFC()

    # save checkpoints
    net_dir = './checkpoint/'
    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    # train setting
    epoch_num = 50
    for epoch in range(epoch_num):
        for step, batch in enumerate(loader):
            loss = tracker.step(batch, backward=True, update_lr=(step == 0))
            if step%20 == 0:
                print('Epoch [{}][{}/{}]: Loss: {:.3f}'.format(epoch + 1, step + 1, len(loader), loss))
                sys.stdout.flush()
        

        net_path = os.path.join(net_dir, 'model_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)
