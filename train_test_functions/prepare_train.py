import sys
import os
from train_test_functions import custom_transforms
import torch
import models
from tensorboardX import SummaryWriter
from train_test_functions.sequence_folders_list import SequenceFolder, SequenceFolder_Validation_Full
from train_test_functions.utils import  save_path_formatter
from path import Path

def initialize_train(parser):

    print("=> initializing train")


    args = parser.parse_args()
    save_path = save_path_formatter(args, parser)
    # workers to 0 in pycharm
    isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
    if isRunningInPyCharm:
        print('=> Pycharm mode - setting workers to 0')
        args.save_path = save_path = 'pycharm_' + save_path
    else:
        print('=> Shell mode - full settings')

    if args.DEBUG:
        args.save_path = save_path = 'debug_' + save_path
        args.print_freq = 5
        args.epoch_size = 10
        args.epochs = 10

    try:
        stats_file = open(os.path.join(args.data, "stats.txt"), "r")
        args.std_vals = [float(val) for val in stats_file.readline().split('[')[1].split(']')[0].split()]
        args.mean_vals = [float(val) for val in stats_file.readline().split('[')[1].split(']')[0].split()]
        stats_file.close()
    except:
        args.std_vals = [0.5, 0.5, 0.5]
        args.mean_vals = [0.5, 0.5, 0.5]


    if args.final_act=='sigmoid' or args.final_act=='sigmoid4x' or args.final_act=='clamp01':
        args.norm_mm_by_act=[0, 1]
    else:
        args.norm_mm_by_act=[]


    #when only 1 channel out dont calc correlation loss
    if args.out_channels==1:
        args.multi_channel_corr_weight = 0

    #when working channelwise use always i=out=1 for network
    if args.channel_wise:
        args.out_channels=1
        args.in_channels=1

    if args.final_act == 'None':
        args.final_act = None

    args.save_path = 'checkpoints' / save_path
    if args.manual_save_path is not None:
        args.save_path = 'checkpoints' / Path(args.manual_save_path)

    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    # saving execution args to file
    with open(args.save_path / 'args_in_run.txt', 'w') as f:
        f.write("command line in run:\n")
        f.write(' '.join(sys.argv[1:]))
        f.write("\n\nargs in run:\n")
        for a in vars(args):
            f.write('--{} {}\n'.format(a, getattr(args, a)))
    f.close()

        # define writers for Tensorboard
    training_writer = SummaryWriter(args.save_path)

    return training_writer, args

def initialize_test(parser):

    print("=> Initializing test mode")


    args = parser.parse_args()
    save_path = save_path_formatter(args, parser)

    #set workers to 0 in pycharm mode to allow debug
    isRunningInPyCharm = "PYCHARM_HOSTED" in os.environ
    if isRunningInPyCharm:
        print('=> Pycharm mode - setting workers to 0')
        args.save_path = save_path = 'pycharm_' + save_path
    else:
        print('=> Shell mode - full settings')
    #use stats for normnalization if stats.txt file is available
    try:
        stats_file = open(os.path.join(args.data, "stats.txt"), "r")
        args.std_vals = [float(val) for val in stats_file.readline().split('[')[1].split(']')[0].split()]
        args.mean_vals = [float(val) for val in stats_file.readline().split('[')[1].split(']')[0].split()]
        stats_file.close()
    except:
        args.std_vals = [0.5, 0.5, 0.5]
        args.mean_vals = [0.5, 0.5, 0.5]

    #set final activation if enabled and normalize inference accordingly
    if args.final_act=='sigmoid' or args.final_act=='sigmoid4x' or args.final_act=='clamp01':
        args.norm_mm_by_act=[0, 1]
    else:
        args.norm_mm_by_act=[]

    #when only 1 channel out dont calc multi channel correlation loss
    if args.out_channels==1:
        args.multi_channel_corr_weight = 0

    #when working channelwise use always i=out=1 for network
    if args.channel_wise:
        args.out_channels=1
        args.in_channels=1
    #for args text file parsing
    if args.final_act == 'None':
        args.final_act = None

    return  args

def create_dataloaders(args):
    print("=> creating data loaders")

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor()
    ])
    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor()])

    if args.normalize:
        normalize = custom_transforms.Normalize(mean=args.mean_vals, std=args.std_vals)
        train_transform.append(normalize)
        valid_transform.append(normalize)

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        grayscale=args.grayscale,
        hsv=args.hsv
    )

    val_set = SequenceFolder_Validation_Full(
        args.data,
        crop_size=(0,0),
        transform=valid_transform,
        grayscale=args.grayscale,
        hsv=args.hsv
    )
    if len(val_set)==0:
        args.skip_validation=True

    #in debug mode, small training config
    if args.DEBUG:
        new_number=min(len(train_set),len(val_set),32)
        train_set.__len__ = new_number
        train_set.samples = val_set.samples[:new_number]
        val_set.__len__ = new_number
        val_set.samples = val_set.samples[:new_number]

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True, drop_last=True)
    #if epoch size is not set use full dataset
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    #save crop size to file
    sample_apbox = train_loader.dataset.samples[0]['apbox']
    sample_apbox = [int(s) for s in sample_apbox.split(',') if s.isdigit()]
    crop_size = sample_apbox[2] - sample_apbox[0], sample_apbox[3] - sample_apbox[1]
    with open(args.save_path / 'crop_size.txt', 'a') as f:
        f.write('{},{}'.format(crop_size[0],crop_size[1]))
    f.close()

    return train_loader, val_loader, args

def create_model(args,device,inference_ref_padded=False):

    print("=> creating model")
    #use grayscale input if enabled
    if args.grayscale == True:
        args.in_channels=1
    #use reflection pad if enabled
    try:
        args.reflection_pad
    except:
        args.reflection_pad=False
    #gather params for model
    inception_params = [args.num_fmaps_1, args.num_fmaps_2, args.kernel_size,args.batch_norm,args.reflection_pad]
    #creating model
    invrep_net = models.InvRepNet(in_channels=args.in_channels, out_channels=args.out_channels,
                                    depth=args.depth, upmode=args.upmode,
                                    downmode=args.downmode, final_act=args.final_act, final_conv=args.final_conv,
                                    channel_wise=args.channel_wise, inception_params=inception_params,
                                    inference_ref_padded=inference_ref_padded).to(device)
    #loading pretrained model if enabled
    if args.pretrained_model:
        print("=> using pre-trained weights for InvRepNet")
        weights = torch.load(args.pretrained_model)
        weights_dict=weights['state_dict']
        for key, val in weights_dict.items():
            weights_dict[key] = val.to(device)
        invrep_net.load_state_dict(weights_dict)


    #Freezing parameters if enabled
    if args.freeze_model:
        print('InvRepNet parameters freezed')
        for param in invrep_net.parameters():
            param.requires_grad = False
    #set cuda configuration
    if device==torch.device("cuda"):
        invrep_net = torch.nn.DataParallel(invrep_net)

    return invrep_net

def create_optimizer(args, model):
    print('=> setting adam solver')
    optim_params = [
        {'params': model.parameters(), 'lr': args.lr},
    ]
    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
    return optimizer
