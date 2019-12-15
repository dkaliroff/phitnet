import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='InvRepNet',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #data
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--grayscale', action='store_true', default=False, help='use grayscale images as input')
    parser.add_argument('--normalize', action='store_true', default=False, help='normalize image to -1 to 1')
    parser.add_argument('--hsv', action='store_true', default=False, help='use HSV images as input')

    #training
    parser.add_argument('--epochs', default=75, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--epoch_size', default=1000, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size')

    #model loading
    parser.add_argument('--pretrained_model', default=None, metavar='PATH', help='path to pre-trained dispnet model')
    parser.add_argument('--freeze_model', action='store_true', default=False, help='freeze freeze nlmasknet model')

    #loss
    parser.add_argument('--dfunc', type=str, choices=['l1', 'l2'], default='l2', help='distance function for loss')
    parser.add_argument('--loss_ver', type=int, metavar='N', help='loss mode: 0-pixel mean 1-patch mean, 2-combined, 3-corr', default=3)
    parser.add_argument('--inter_margin', type=float, help='margin for triplet loss', metavar='W', default=0.1)
    parser.add_argument('--inter_weight', type=float, help='inter weight in total loss', metavar='W', default=4)
    parser.add_argument('--intra_weight', type=float, help='intra weight in total loss', metavar='W', default=1)
    parser.add_argument('--intra_secondary_weight', type=float, help='secondary intra relative weight in total loss', metavar='W', default=1)
    parser.add_argument('--intra_margin_ratio', type=float, help='margin for intra loss (as ratio from inter margin)', metavar='W', default=0)
    parser.add_argument('--border_len_ignore', type=int,metavar='W', default=0,help='ignore part of the patch border')

    #regularization loss
    parser.add_argument('--multi_channel_corr_weight', type=float, metavar='W', default=1, help='multi_channel_corr weight')
    parser.add_argument('--scale_consistency_weight', type=float, metavar='W', default=1, help='scale consistency weight')
    parser.add_argument('--scale_consistency_max', type=float, metavar='W', default=2, help='maximum upsampling, valid range=(1,2]')
    parser.add_argument('--scale_consistency_dfunc', type=str, choices=['l12','l12normed','corr'], default='corr', help='distance function for loss')
    parser.add_argument('--rot_consistency_weight', type=float, metavar='W', default=0, help='rot_consistency weight')
    parser.add_argument('--rot_consistency_dfunc', type=str, choices=['l12','corr'], default='l12', help='distance function for loss rot loss')

    #network arch
    parser.add_argument('--in_channels', default=3, type=int, metavar='N', help='number of input channels')
    parser.add_argument('--out_channels', default=3, type=int, metavar='N', help='number of output channels ')
    parser.add_argument('--channel_wise', action='store_true', default=False, help='channel wise network input')
    parser.add_argument('--final_act', type=str, choices=['clamp01','sigmoid', 'sigmoid4x','None'], default='None', help='last activation layer at networks output')
    parser.add_argument('--final_conv', type=str, choices=['1x1','3x3','group3x3'], default='3x3', help='final convolution to out_channels')
    parser.add_argument('--upmode', type=str, choices=['transpose','bilinear'], default='bilinear', help='upsampling mode')
    parser.add_argument('--downmode', type=str, choices=['conv','maxpool','avgpool'], default='maxpool', help='downsampling mode')
    parser.add_argument('--batch_norm', action='store_true', default=False, help='enables batch_norm in inception layers')
    parser.add_argument('--depth', default=5, type=int, metavar='N', help='depth of model (same depth of encoder and decoder)')
    parser.add_argument('--kernel_size', default=5, type=int, metavar='N', help='kernel size of inception layer')
    parser.add_argument('--num_fmaps_1', default=32, type=int, metavar='N', help='number of channels of 1st inception conv')
    parser.add_argument('--num_fmaps_2', default=64, type=int, metavar='N', help='number of channels of 2nd inception conv')
    parser.add_argument('--reflection_pad', action='store_true', default=False, help='reflection padding for conv layers instead of zero')

    #optimizer
    parser.add_argument('--lr', default=1e-5, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay')

    #execution
    parser.add_argument('--workers', default=10, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--seed', default=2019, type=int, help='seed for random functions, and network initialization')

    #logging
    parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency inside epoch')
    parser.add_argument('--print_freq_epoch', default=1, type=int, metavar='N', help='print frequency of epochs')
    parser.add_argument('--validation_freq', default=1, type=int, metavar='N', help='epoch frequency of validation')
    parser.add_argument('--checkpoint_freq', default=10, type=int, metavar='N', help='save checkpoint epoch freq')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='debug mode - small training')
    parser.add_argument('--skip_validation', action='store_true', default=False, help='disables validation after epochs')
    parser.add_argument('--manual_save_path', type=str, help='manual save folder name - overrides auto-name')

    # inference and tasks
    parser.add_argument('--with_task_val', action='store_true', default=False, help='patch matching validation enable')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu instead of gpu')
    parser.add_argument('--pretrained_path', default=None, metavar='PATH', help='path to pre-trained invrepnet model')
    parser.add_argument('--pm_mode', default=0, type=int, metavar='N', help='patch matching mode, default is 3 channel invrepnet')
    parser.add_argument('--additional_rep', action='store_true', default=False, help='flag to test additional representation')

    return parser

