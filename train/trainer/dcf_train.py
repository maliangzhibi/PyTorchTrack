import time
import utils.dcf_utils as dcf_util
import torch
import troch.nn as nn


def train(data_train_loader, data_train_target, model, criterion, optimizer, epoch):
    """
    data_train_loader: the data for training DCFNet
    model: the network for training DCFNet
    criterion: the loss for the outputs of the DCFNet
    optimizer: the schedule for optimize the loss
    epoch: the number for iteration over all data_train 
    """
    args = dcf_util.train_setting()
    batch_time = dcf_util.AverageMeter()
    data_time = dcf_util.AverageMeter()
    losses = dcf_util.AverageMeter()

    # set the train mode
    model.train()

    end = time.time()
    for i, (template, search) in enumerate(data_train_loader):
        # time for data loading
        data_time.update(time.time - end)

        template = template.cuda(non_blocking=True)
        search = search.cuda(non_blocking=True)

        # the data pass the network
        output = model(template, search)
        loss = criterion(output, data_train_target)/template.size(0)

        # record loss
        losses.update(loss.item())

        # perform the optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # elapsed time for a mini_bacth
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val: .3f}({batch_time.avg: .3f})\t'
                'Data {data_time.val: .3f}({data_time.avg: .3f})\t'
                'Loss {loss_time.val: .4f}({loss_time.avg: .3f})\t'.format(epoch, i, 
                len(data_train_loader), batch_time=batch_time, data_time=data_time, 
                loss_time=losses)
            )


def validate(data_val_loader, data_val_target, model, criterion):
    """
    data_val_loader: as validate dataset for the DCFNet
    model: the network for DCFNet
    criterion: the loss for the output of DCFNet
    """
    args = dcf_util.train_setting()

    batch_time = dcf_util.AverageMeter()
    losses = dcf_util.AverageMeter()

    # set the train mode to eval
    model.eval()
    with torch.no_grad():
        end = time.time()

        for i, (template, search) in enumerate(val_loader):
            # convert data to cuda type
            template = template.cuda(non_blocking=True)
            search = search.cuda(non_blocking=True)
            
            # the val_data pass the dcfnet
            output = model(template, search)
            loss = criterion(output, data_val_target)
            losses.update(loss.item())
            
            # no optimization
            
            # time for elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    'Test: [{0}/{1}]\t'
                    'Time {batch_time.val: .3f}({batch_time.avg: .3f})\t'
                    'Loss {loss.val: .4f}({loss.avg: .4f})\t'.format(i, len(data_val_loader), 
                    batch_time=batch_time, loss=losses)
                )
        print('* final loss: {loss.val: .4f}({loss.avg: .4f})'.format(loss=losses))
    
    return losses.avg       




