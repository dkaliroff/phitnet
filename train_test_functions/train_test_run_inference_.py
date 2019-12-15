import argparse
import torch.utils.data
from train_test_functions import prepare_train, custom_transforms
from imageio import imsave
import os
from train_test_functions.utils import tensor2array
import torch
from train_test_functions.sequence_folders_list import SequenceFolder_Inference
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
    #for patch matching
    args = parser.parse_args()
    pretrained_model = args.pretrained_model
    data_dir = args.data
    args_file = (args.pretrained_model).rsplit('/', 1)[0]
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
                    print('arg.{}={}'.format(tokens[0], tokens[1]))
                    vars(args)[tokens[0]] = tokens[1]

    # eval
    args.freeze_model = True
    args.pretrained_model = pretrained_model
    args.data = data_dir
    run_pm_images_rep(args)

##PM INF
def run_pm_images_rep(args):
    global device
    if args.use_cpu:
        device = torch.device("cpu")

    # eval args
    args.freeze_model = True
    #inference transformation - image to tensor
    inference_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor()])
    #normalzie if active
    if args.normalize == True:
        normalize = custom_transforms.Normalize(mean=args.mean_vals, std=args.std_vals)
        inference_transform.append(normalize)
    print("=> fetching scenes in '{}'".format(args.data))
    #create inference set
    inf_set = SequenceFolder_Inference(
        args.data,
        crop_size=(0, 0),
        resize_ratio=0,
        transform=inference_transform,
    )
    print('{} samples found in {} inference scenes'.format(len(inf_set), len(inf_set.scenes)))

    print('=> Inference results will be saved in:\n{}'.format(args.output))
    inf_loader = torch.utils.data.DataLoader(
        inf_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)

    #create model
    invrep_net = prepare_train.create_model(args, device, inference_ref_padded=True).to(device)
    invrep_net.eval()

    #create dirs
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(os.path.join(args.output,'img')):
        os.makedirs(os.path.join(args.output,'img'))
    if not os.path.exists(os.path.join(args.output, 'rep')):
        os.makedirs(os.path.join(args.output,'rep'))


    #inference loop by scenes
    idx=0
    curr_scene_name=None
    name_lines=[]
    for img, img_full_name in inf_loader:
        scene_name = img_full_name[0].split('/')[-2]
        if 'png' in str(img_full_name) or 'jpg' in str(img_full_name):
            if scene_name==curr_scene_name:
                idx+=1
            else:
                curr_scene_name=scene_name
                idx=0
            img_name = img_full_name[0].split('/')[-1]
            img_name_idx = 'im'+str(idx)+'.png'
            name_lines.append('scene@{}@image@{}@image_name_idx@{}'.format(scene_name,img_name,img_name_idx))
            run_image(img, scene_name, img_name_idx, invrep_net, args)
            del img

    #saving log
    with open(os.path.join(args.output,'inference_record.txt'), 'w') as f:
        name_lines.sort()
        for line in name_lines:
            f.write("%s\n" % line)
    f.close()

    return 0

def run_image(img, scene_name, img_name, invrep_net, args):
    with torch.no_grad():
        print("=> scene: {}, image: {}".format(scene_name, img_name))
        img = [im.to(device) for im in img]
        rep1 = invrep_net(img)[0]
        img_png = (255 * np.transpose(tensor2array(args, img[0][0], [0]), (1, 2, 0))).astype(np.uint8)
        img_file_name = os.path.join(args.output,'img',scene_name + '_' + img_name)
        imsave(img_file_name, img_png)
        rep_png = (255 * np.transpose(tensor2array(args, rep1[0], args.norm_mm_by_act), (1, 2, 0))).astype(np.uint8)
        rep_file_name = os.path.join(args.output, 'rep', scene_name + '_' + img_name)
        print(rep_file_name)
        if args.out_channels == 3 or args.out_channels == 1:
            imsave(rep_file_name, rep_png)
        else:
            np.save(rep_file_name,rep_png)


##REG INF
def run_reg_images(args):
    global device
    if args.use_cpu:
        device = torch.device("cpu")

    inference_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor()])

    if args.normalize == True:
        normalize = custom_transforms.Normalize(mean=args.mean_vals, std=args.std_vals)
        inference_transform.append(normalize)
    print("=> fetching scenes in '{}'".format(args.data))

    inf_set = SequenceFolder_Inference(
        args.data,
        crop_size=(0, 0),
        resize_ratio=0,
        transform=inference_transform,
    )

    print('{} samples found in {} inference scenes'.format(len(inf_set), len(inf_set.scenes)))
    inf_loader = torch.utils.data.DataLoader(
        inf_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)

    print('=> Image pairs will be saved in:\n{}'.format(args.output))

    idx=0
    curr_scene_name=None
    name_lines=[]
    for img, img_full_name in inf_loader:
        scene_name = img_full_name[0].split('/')[-2]
        if 'png' in str(img_full_name) or 'jpg' in str(img_full_name):
            if scene_name==curr_scene_name:
                idx+=1
            else:
                curr_scene_name=scene_name
                idx=0
            img_name = img_full_name[0].split('/')[-1]
            img_name_idx = 'im'+str(idx)+'.png'
            name_lines.append('scene@{}@image@{}@image_name_idx@{}'.format(scene_name,img_name,img_name_idx))
            run_image_no_net(img, scene_name, img_name_idx, args)
            del img

    return 0

def run_image_no_net(img, scene_name, img_name, args):
    with torch.no_grad():
        print("=> scene: {}, image: {}".format(scene_name, img_name))
        img = [im.to(device) for im in img]
        img_png = (255 * np.transpose(tensor2array(args, img[0][0], [0]), (1, 2, 0))).astype(np.uint8)
        img_file_name = os.path.join(args.output,scene_name + '_' + img_name)
        imsave(img_file_name, img_png)


if __name__ == '__main__':
    main()
