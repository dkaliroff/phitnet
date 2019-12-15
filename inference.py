import cv2
import argparse
import torch.utils.data
from train_test_functions import prepare_train, custom_transforms
from imageio import imsave
import os
from train_test_functions.utils import tensor2array
import torch
from train_test_functions.sequence_folders_list import SequenceFolderPlane
import numpy as np

parser = argparse.ArgumentParser(description='InvRepNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# data
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--output', metavar='DIR', required=True, help='path to output dir')
parser.add_argument('--pretrained_model', required=True, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--use_cpu', action='store_true', default=False, help='run in train mode (not eval)')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global device
    args = parser.parse_args()
    #args from command line
    pretrained_model = args.pretrained_model
    data_dir = args.data
    args_file = (args.pretrained_model).rsplit('/', 1)[0]

    #loading args from file
    with open(os.path.join(args_file, 'args_in_run.txt')) as fp:
        for line in fp:
            if line.startswith('--'):
                tokens = line[2:].strip().split()
                if tokens[1].isdigit():
                    tokens[1] = int(tokens[1])
                if tokens[1] == "True":
                    tokens[1] = True
                if tokens[1] == "False":
                    tokens[1] = False
                if tokens[1] == "None":
                    tokens[1] = None
                if tokens[1] == "[]":
                    tokens[1] = []
                if tokens[1] is not 'None':
                    # print('arg.{}={}'.format(tokens[0], tokens[1]))
                    vars(args)[tokens[0]] = tokens[1]

    # eval args
    args.freeze_model = True
    args.batch_size = 1

    # set command line args
    args.pretrained_model = pretrained_model
    args.data = data_dir
    if args.use_cpu:
        device = torch.device("cpu")

    # eval args
    args.freeze_model = True
    args.batch_size = 1
    run_net(args)

def run_net(args):
    global device
    inference_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor()])
    #normalize if enabled
    if args.normalize == True:
        normalize = custom_transforms.Normalize(mean=args.mean_vals, std=args.std_vals)
        inference_transform.append(normalize)
    print("=> fetching scenes in '{}'".format(args.data))
    #create inference plain set and data loader (all images is folder, no subfolders)
    inf_set = SequenceFolderPlane(
        args.data,
        transform=inference_transform,
    )
    inf_loader = torch.utils.data.DataLoader(
        inf_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)

    #create model
    invrep_net = prepare_train.create_model(args, device, inference_ref_padded=True).to(device)
    invrep_net.eval()

    #create output dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #inference loop
    for img, img_full_name in inf_loader:
        img_name = img_full_name[0].split('/')[-1]
        run_image(img, img_name, invrep_net, args)
        del img
    return 0

def run_image(img, img_name, invrep_net, args):
    with torch.no_grad():
        img = [im.to(device) for im in img]
        if img[0].shape[1]==1:
            img=[torch.stack((img[0].squeeze(1),img[0].squeeze(1),img[0].squeeze(1)),dim=1)]
        rep1 = invrep_net(img)[0]
        rep_png = (255 * np.transpose(tensor2array(args, rep1[0], args.norm_mm_by_act), (1, 2, 0))).astype(np.uint8)
        rep_file_name = os.path.join(args.output,img_name)
        print(rep_file_name)
        imsave(rep_file_name, rep_png)



if __name__ == '__main__':
    main()
