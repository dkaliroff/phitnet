from __future__ import division
import torch
from path import Path
import datetime

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))

    save_path = Path('_'.join(folder_string))
    timestamp = datetime.datetime.\
        now().strftime("%m_%d_%H_%M")
    return save_path/timestamp

def tensor2array(args, tensor, norm_mm):
    tensor = tensor.detach().cpu()
    if len(norm_mm) == 1 and norm_mm[0]==0 and args.normalize:
        for ch in range(tensor.shape[0]):
            tensor[ch]=args.mean_vals[ch] + tensor[ch]*args.std_vals[ch]

    if len(norm_mm)==0 or len(norm_mm)==2:
        min_value = tensor.min().item() if len(norm_mm)==0 else norm_mm[0]
        max_value = tensor.max().item() if len(norm_mm)==0 else norm_mm[1]
        tensor = (tensor - min_value) / (max_value - min_value)

    if tensor.ndimension() == 2:
        tensor=tensor.unsqueeze(0)

    return tensor.numpy()

def square_center_crop(tensor, tw):
    w = tensor.shape[-1]
    assert(w>=tw)
    if tw<w:
        x1 = int(round((w - tw) / 2.))
        return tensor[:,:,x1:x1 + tw, x1:x1 + tw]
    else:
        return tensor

def center_crop(tensor, th, tw):
    w = tensor.shape[-1]
    h = tensor.shape[-2]
    x1 = int(round((w - tw) / 2.))
    x2 = int(round((h - th) / 2.))
    return tensor[:,:,x1:x1 + tw, x2:x2 + tw]

def scale_center_crop(tensor, scale):
    w = tensor.shape[-1]
    tw = int(w * scale)
    x1 = int(round((w - tw) / 2.))
    return tensor[:, :, x1:x1 + tw, x1:x1 + tw]

def save_checkpoint(save_path, net_state, checkpoint_freq, epoch):
    prefix = 'invrep'
    torch.save(net_state, save_path / '{}_checkpoint.pth.tar'.format(prefix))
    if epoch%checkpoint_freq==0:
        torch.save(net_state, save_path / '{}_epoch_{}.pth.tar'.format(prefix,epoch))
