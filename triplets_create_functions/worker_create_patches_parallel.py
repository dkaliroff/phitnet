import os
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.util import view_as_windows
from PIL import Image

def process_scene(scene,root, single_anchor,ap_neighbors, crop_len, step_len,andist_thresh, apdist_thresh, std_thresh,extended_random,
                  crop_len_random,no_sort,BT_apdist,border_margin=0, crop_file = False,mask_thresh=1):


    if BT_apdist > 0 and 'BT' in scene[0]:
        apdist_thresh=BT_apdist
    total_std=0
    total_mean=0
    step_len=step_len
    path_list = scene
    if not no_sort:
        path_list.sort()
    folder_id = path_list[0].split('/')[0]
    #   number of images in one sequence
    num_imgs = len(path_list)
    #creating file to save triplets (in scene folder)
    triplets_file = open(os.path.join(root,folder_id,"triplets.txt"), "w")
    print("** scene: {} ** => start processing".format(folder_id))

    # sample image
    img_name = path_list[0].split('/')[-1]
    img_path = root + str(folder_id) + "/data/" + img_name
    mask_path = root + str(folder_id) + "/data/" + img_name[:-4] + "_mask.png"

    # load sample image anc check sizes against mask
    rgb_img = Image.open(img_path)
    try:
        mask_img = Image.open(mask_path)
        assert (rgb_img.size == mask_img.size)
    except:
        print("** scene: {} ** => no mask available for this scene".format(folder_id))


    #get image size
    original_w, original_h = rgb_img.size

    new_w=original_w-border_margin
    new_h=original_h-border_margin
    new_box=(border_margin,border_margin,border_margin+new_w,border_margin+new_h)
    if crop_file:
        with open(os.path.join(root,folder_id,'crop.txt')) as cf:
            line = cf.readline().strip()
            line=[int(s)-1 for s in line.split() if s.isdigit()]
            new_box=[line[2],line[0],line[3],line[1]]
            new_w=new_box[2]-new_box[0]
            new_h=new_box[3]-new_box[1]
        assert(crop_len<new_w and crop_len<new_h)

    # for each image in the sequence - create triplets with all other images

    num_anchors = num_imgs if not single_anchor else 1
    triplets_count=0
    # loop over all posible anchor images
    for ii in range(num_anchors):
        a_img_name = path_list[ii].split('/')[-1]
        a_img_path = root + str(folder_id) + "/data/" + a_img_name
        a_img = (Image.open(a_img_path)).crop(new_box)
        a_patches = view_as_windows(np.array(a_img), (crop_len,crop_len,3), step=step_len).squeeze(2)/255.0
        try:
            a_mask = mask_img.crop(new_box)
            a_mask_patches = view_as_windows(np.array(a_mask).astype('uint8'), (crop_len, crop_len), step=step_len)
            a_mask_patches = np.sum(a_mask_patches, axis=(-1, -2)) / (crop_len * crop_len * a_mask_patches.max())
        except:
            a_mask_patches=None

        num_rows = a_patches.shape[0]
        num_cols = a_patches.shape[1]
        positive_range = range(num_imgs)

        for jj in positive_range:
            if ii==jj:
                continue
            # use only next and previous frames (in order of the frames) as positive
            if ap_neighbors and abs(ii-jj)>1:
                continue
            p_img_name = path_list[jj].split('/')[-1]
            p_img_path = root + str(folder_id) + "/data/" + p_img_name
            p_img = (Image.open(p_img_path)).crop(new_box)
            p_patches = view_as_windows(np.array(p_img), (crop_len, crop_len, 3), step=step_len).squeeze(2)/255.0
            try:
                p_mask = mask_img.crop(new_box)
                p_mask_patches = view_as_windows(np.array(p_mask).astype('uint8'), (crop_len, crop_len), step=step_len)
                p_mask_patches = np.sum(p_mask_patches, axis=(-1, -2)) / (crop_len * crop_len* p_mask_patches.max())
            except:
                p_mask_patches = None

            # loop over patches in anchor image and create triplets with current positive
            for row in range(0,num_rows,2):
                for col in range(0,num_cols,2):
                    #first - check if this patch is usable according to mask
                    if a_mask_patches is not None:
                        ampatch=a_mask_patches[row,col]
                        pmpatch=p_mask_patches[row,col]
                        mask_condition = ampatch >= mask_thresh and pmpatch >= mask_thresh
                        if mask_condition == False:
                            break

                    #create crop box for this path
                    a_crop_box=[col*step_len,row*step_len,col*step_len+crop_len,row*step_len+crop_len]

                    #select patches from patches grid
                    apatch = a_patches[row, col]
                    ppatch = p_patches[row, col]

                    #use patches if answers conditions:
                    #std above threshold and difference between anchor and positive above threshold
                    std_condition= apatch.std()>std_thresh and ppatch.std() > std_thresh
                    positive_condition = np.sqrt(np.mean((apatch-ppatch)**2)) > apdist_thresh

                    if (std_condition and positive_condition):

                        for nn in range(0,9):
                            #select negative from patch in the 8-neigborhood
                            if extended_random:
                                move_y, move_x = random_patch_move_extended(row, col, num_rows-1, num_cols-1)
                                nrow = max(0,min(num_rows-1,row + move_y))
                                ncol = max(0,min(num_cols-1,col + move_x))
                            else:
                                move_y, move_x = random_patch_move(row, col, num_rows-1, num_cols-1)
                                nrow=row+move_y
                                ncol=col+move_x

                            n_crop_box=[ncol*step_len,nrow*step_len,ncol*step_len+crop_len,nrow*step_len+crop_len]
                            npatch = p_patches[nrow,ncol]

                            assert(np.count_nonzero(npatch-ppatch)>0)
                            # TODO: think about a negative condition - hard negative mining ?
                            negative_condition = np.mean((apatch - npatch) ** 2) > andist_thresh

                            if negative_condition:
                                total_std = total_std + (apatch.std(axis=(0, 1)) + ppatch.std(axis=(0, 1)) + npatch.std(axis=(0, 1))) / 3.0
                                total_mean = total_mean + (apatch.mean(axis=(0, 1)) + ppatch.mean(axis=(0, 1)) + npatch.mean(axis=(0, 1))) / 3.0

                                if crop_len_random:
                                    new_len_add = random.choice([32, 64, 128]) - crop_len
                                    max_cond_a = (a_crop_box[2] + new_len_add < new_h and a_crop_box[3] + new_len_add < new_w)
                                    min_cond_a = (a_crop_box[2] + new_len_add > 0 and a_crop_box[3] + new_len_add > 0)
                                    max_cond_n = (n_crop_box[2] + new_len_add < new_h and n_crop_box[3] + new_len_add < new_w)
                                    min_cond_n = (n_crop_box[2] + new_len_add > 0 and n_crop_box[3] + new_len_add > 0)
                                    if new_len_add != 0 and max_cond_a and max_cond_n and min_cond_a and min_cond_n:
                                        a_crop_box[2] += new_len_add
                                        a_crop_box[3] += new_len_add
                                        n_crop_box[2] += new_len_add
                                        n_crop_box[3] += new_len_add

                                # writing triplet to file
                                write_triplet_line(a_crop_box, n_crop_box, border_margin, a_img_name, p_img_name,
                                                   triplets_file)
                                triplets_count = triplets_count + 1


    triplets_file.close()
    # print("** scene: {} ** => finished".format(folder_id))
    print("** scene: {} ** => finished, number of triplets = {}".format(folder_id,triplets_count))
    return triplets_count,total_std,total_mean

def random_patch_move(row,col,num_rows,num_cols):
    # select negative from patch in the 4-neigborhood
    y_ext_val = random.choice([0, 1]) if row == 0 else random.choice([-1, 0])
    move_y = random.choice([-1, 0, 1]) if row not in [0, num_rows] else y_ext_val
    if move_y != 0:
        x_ext_val = random.choice([0, 1]) if col == 0 else random.choice([-1, 0])
        move_x = random.choice([-1, 0, 1]) if col not in [0, num_cols] else x_ext_val
    else:
        x_ext_val = 1 if col == 0 else -1
        move_x = random.choice([-1, 1]) if col not in [0, num_cols] else x_ext_val

    return move_y, move_x

def random_patch_move_extended(row,col,num_rows,num_cols):
    # select negative from patch in the 4-neigborhood
    y_ext_val = random.choice(range(0, 4)) if row == 0 else random.choice(range(-4, 0))
    move_y = random.choice(range(-4, 4)) if row not in [0, num_rows] else y_ext_val
    if move_y != 0:
        x_ext_val = random.choice(range(0, 4)) if col == 0 else random.choice(range(-4, 0))
        move_x = random.choice(range(-4, 4)) if col not in [0, num_cols] else x_ext_val
    else:
        x_ext_val = random.choice(range(1, 4)) if col == 0 else random.choice(range(-4, -1))
        move_x = random.choice([-4,-3,-2,-1,1,2,3,4]) if col not in [0, num_cols] else x_ext_val

    return move_y, move_x


def write_triplet_line(a_crop_box,n_crop_box,border_margin,a_img_name,p_img_name,triplets_file):
    """
    line format: <anchor_image>@<positive_image>@<anchor/positive_patch_box>@<negative_patch_box>
    patch_box format: start_y,start_x,end_y,end_x
    example: 0001.png@0005.png@212,20,340,148@244,52,372,180
    """
    DEL = '@'
    # save the coordinates of the crop according to the original image
    a_crop_box_org_string = ','.join(map(str, [x + border_margin for x in a_crop_box]))
    n_crop_box_org_string = ','.join(map(str, [x + border_margin for x in n_crop_box]))
    triplet_line = a_img_name + DEL + p_img_name + DEL + a_crop_box_org_string + DEL + n_crop_box_org_string
    triplets_file.writelines(triplet_line + "\n")

def debug_show(a,p=None,n=None, time=1):
    if p is None and n is None:
        plt.figure()
        plt.imshow(a)
        plt.show(block=False)
        plt.pause(time)
        plt.close()
    elif n is None:
        plt.figure()
        plt.subplot(121), plt.imshow(a)
        plt.subplot(122), plt.imshow(p)
        plt.show(block=False)
        plt.pause(time)
        plt.close()
    else:
        plt.figure()
        plt.subplot(131), plt.imshow(a)
        plt.subplot(132), plt.imshow(p)
        plt.subplot(133), plt.imshow(n)
        plt.show(block=False)
        plt.pause(time)
        plt.close()

