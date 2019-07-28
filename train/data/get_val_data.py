from dataset.datasetvid import VID
from torch.utils.data import dataloader
from utils.dcf_utils import train_setting as setting

def get_train_data(gpu_num = 1, crop_base_path = None, args = None):
    if args == None:
        args = setting()
    
    if crop_base_path == None:
        print('crop_path is None, please pass the correct argments.')
        exit()
    
    val_dataset = VID(root=crop_base_path, train=False, range=args.range)
    val_loader = dataloader.DataLoader(
        val_dataset, batch_size = args.batch_size*gpu_num, shuffle=False, num_works = args.works, pin_memory = True, drop_last=True
    )
    return val_loader