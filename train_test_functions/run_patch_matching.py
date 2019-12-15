import pandas as pd
import os
import cv2
import numpy as np
import random

def get_diff_scaled(img1,img2,scale):
    return ((np.clip((img1.astype('int32') - img2.astype('int32')) * scale, -255, 255) + np.ones_like(
        img1) * 255) / 2).astype('uint8')

def get_diff(img1,img2):
    return ((img1.astype('int32')-img2.astype('int32')+np.ones_like(img1)*255)/2).astype('uint8')

def run_patch_matching_3(pm_mode=0,args=None):
    # select rep mode
    if pm_mode == 0:
        representations = ['invrepnet', 'org']
    elif pm_mode == 1:
        representations = ['invrepnet_gray', 'org_gray']
    elif pm_mode == 2:
        representations = ['additional_rep']
    else:
        exit("error")

    # obtain image names
    images = next(os.walk(args.task_invrep_dirs))[-1]
    scenes_list = list(set([im.split('_im')[0] for im in images]))
    scenes_list.sort()
    num_scenes = float(len(scenes_list))
    test_image_dir = [args.task_image_dirs, args.task_invrep_dirs]

    #set iters (number of patches) in each image
    iters=100
    print("Iters = {}".format(iters))
    #set seed
    random.seed(2019)
    #set patch sizes
    patch_sizes = np.array([32, 64, 128])
    col_names = ['32', '64', '128']
    #choose matching method
    methods = ['cv2.TM_CCORR_NORMED']
    method_final=[]
    loglist = []
    #patchmathcing  main loop
    for m, meth in enumerate(methods):
        to_csv = np.copy(patch_sizes)
        for reptype in representations:
            if 'org' in reptype or 'additional' in reptype:
                rt =0
            elif 'invrepnet' in reptype:
                rt = 1
            else:
                exit('error')
            method_mean_iou_0 = np.zeros(len(patch_sizes))
            for sl, scene_name in enumerate(scenes_list):
                #read images
                if args.out_channels == 3 or args.out_channels == 1:
                    im0_rep = cv2.imread(os.path.join(test_image_dir[rt], scene_name + '_im0.png'))
                    im1_rep = cv2.imread(os.path.join(test_image_dir[rt], scene_name + '_im1.png'))
                else:
                    im0_rep=np.load(os.path.join(test_image_dir[rt], scene_name + '_im0.png.npy'))
                    im1_rep=np.load(os.path.join(test_image_dir[rt], scene_name + '_im1.png.npy'))
                #convert to gray if gray mode
                if 'gray' in reptype:
                    if 'invrepnet' in reptype:
                        if 'mean' in reptype:
                            im0_rep = im0_rep.mean(-1).astype('uint8')
                            im1_rep = im1_rep.mean(-1).astype('uint8')
                        else:
                            im0_rep = cv2.cvtColor(im0_rep, cv2.COLOR_BGR2GRAY)
                            im1_rep = cv2.cvtColor(im1_rep, cv2.COLOR_BGR2GRAY)
                    else:
                        im0_rep = cv2.cvtColor(im0_rep, cv2.COLOR_BGR2GRAY)
                        im1_rep = cv2.cvtColor(im1_rep, cv2.COLOR_BGR2GRAY)
                scene_mean_iou_0 = np.zeros(len(patch_sizes))
                #loop over patch sizes
                for jj, ps in enumerate(patch_sizes):
                    np.random.seed(2019)
                    max_x = im1_rep.shape[1] - ps
                    max_y = im1_rep.shape[0] - ps
                    x = np.random.randint(0, max_x, size=iters)
                    y = np.random.randint(0, max_y, size=iters)
                    patch_size_iou_sum = 0
                    #random patches iter loop
                    for ii in range(iters):
                        template, cr_location = get_random_crop(im1_rep, ps, ps, [x[ii], y[ii]])
                        # Apply template Matching
                        if 'gray' in reptype:
                            res = cv2.matchTemplate(im0_rep, template, eval(meth))
                        else:
                            res=cv2.matchTemplate(im0_rep[:,:,0], template[:,:,0], eval(meth))
                            for ch in range(1, args.out_channels):
                                res= res * cv2.matchTemplate(im0_rep[:,:,ch], template[:,:,ch], eval(meth))
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                        if eval(meth) in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            top_left = min_loc
                        else:
                            top_left = max_loc
                        bottom_right = (top_left[0] + ps, top_left[1] + ps)
                        box_detected = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
                        box_gt = [cr_location[0], cr_location[1], cr_location[0] + ps, cr_location[1] + ps]
                        #calculate accuracy IOU
                        patch_size_iou_sum = patch_size_iou_sum + bb_intersection_over_union(box_detected, box_gt)
                    scene_mean_iou_0[jj] = patch_size_iou_sum / float(iters)
                    printlog(loglist,'{}: method_0, scene: {}, patch_size= {}, mean_iou= {:4}'.format(reptype, scene_name, ps, scene_mean_iou_0[jj]))
                method_mean_iou_0 += scene_mean_iou_0 / num_scenes

            printlog(loglist,'final mean_iou for match method: {}'.format(meth))
            for kk, ps in enumerate(patch_sizes):
                printlog(loglist,'{}: method_0: patch_size={}, mean_iou={}'.format(reptype, ps, method_mean_iou_0[kk]))
            printlog(loglist,'\n*****************************************************************************************************\n')

            to_csv = np.vstack([to_csv, method_mean_iou_0])
            if reptype == 'invrepnet' or reptype == 'invrepnet_multi_all':
                method_final.append(method_mean_iou_0)


        #saving results to CSV
        I = pd.Index(representations, name="rows")
        C = pd.Index(col_names, name="columns")
        df = pd.DataFrame(data=to_csv[1:,:], index=I, columns=C)
        print(df)
        try:
            df.to_csv(os.path.join(args.output, "{}.csv".format(meth.split('.')[-1])))
        except:
            print("output directory not found, results were not saved in csv file")
    #saving log
    try:
        with open(os.path.join(args.output, 'patchmatching_record.txt'), 'w') as f:
            for line in loglist:
                f.write("%s\n" % line)
        f.close()
    except:
        print("output directory not found, record was not saved")

    return method_final, methods

def printlog(loglist,line):
    print(line)
    loglist.append(line)


def get_random_crop(image, crop_height, crop_width, xy=None):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    if xy == None:
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
    else:
        x = xy[0]
        y = xy[1]

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop, [x, y]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


if __name__ == '__main__':
    print('main')