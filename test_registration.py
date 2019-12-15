from train_test_functions.test_tasks_utils import get_psnr_rmse
from train_test_functions.test_tasks_utils import  affine_transform as aftf
import torch.utils.data
from train_test_functions.utils import tensor2array
import torch
import numpy as np
import torchvision.transforms.functional as TVF
import cv2
import torch.backends.cudnn as cudnn
import torch.utils.data
from train_test_functions import prepare_train
import os
import arguments
from train_test_functions.train_test_run_inference_ import run_reg_images as run_inf


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


'''
This script receives a test data directory and a pre_trained model.
It formats the images for test and runs the registartion test.

To run this from command line:
python test_registration.py /path/to/the/test/data/ --pretrained_model /path/to/the/pretrained/model/
'''


def run_registration_rep(invrep_net,args,im1,im2,angle,shear,translation,iters):
    with torch.no_grad():

        # size of original images
        org_sz = im1.shape
        # apply affine transform
        mask = (np.ones_like(im2)).astype('uint8')
        mask_transformed, M = aftf(mask, angle, shear, translation)

        # new size
        sz = mask_transformed.shape
        # get gray transformed
        pw = [0, 0]
        pw[0] = int(np.ceil((sz[0] - org_sz[0]) / 2))
        pw[1] = int(np.ceil((sz[1] - org_sz[1]) / 2))
        #update borders
        if pw[0] > 0 or pw[1] > 0:
            mask = cv2.copyMakeBorder(mask, pw[0], pw[0], pw[1], pw[1], cv2.BORDER_CONSTANT, value=0)
        #set mask
        mask_valid =  mask_transformed
        #image2 transformed
        im2_transformed,M = aftf(im2, angle, shear, translation, type_border=cv2.BORDER_REFLECT)
        #rep2 transformed
        rep2_transformed_tensor = invrep_net([TVF.to_tensor(cv2.cvtColor(im2_transformed, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)])[0]
        rep2_transformed= (255 * np.transpose(tensor2array(args, rep2_transformed_tensor[0], args.norm_mm_by_act),
                                               (1, 2, 0))).astype(np.uint8)*mask_valid
        rep2_gray_transformed = cv2.cvtColor(rep2_transformed, cv2.COLOR_RGB2GRAY) * mask_valid[:,:,0]

        if pw[0] > 0 or pw[1] > 0:
            # Padd images with zeros according to new size
            im1 = cv2.copyMakeBorder(im1, pw[0], pw[0], pw[1], pw[1], cv2.BORDER_REFLECT, value=0)
            im2 = cv2.copyMakeBorder(im2, pw[0], pw[0], pw[1], pw[1], cv2.BORDER_CONSTANT, value=0)

        #rep1 to gray
        rep1_tensor = invrep_net([TVF.to_tensor(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)])[0]
        rep1 = (255 * np.transpose(tensor2array(args, rep1_tensor[0], args.norm_mm_by_act), (1, 2, 0))).astype(np.uint8)
        rep1_gray = cv2.cvtColor(rep1, cv2.COLOR_RGB2GRAY)*mask[:,:,0]

        # Convert images to grayscale
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)*mask[:,:,0]
        im2_transformed_gray = cv2.cvtColor(im2_transformed, cv2.COLOR_BGR2GRAY)* mask_valid[:,:,0]

        # Define the motion model
        warp_mode = cv2.MOTION_AFFINE

        # Specify the threshold of the increment in the correlation coefficient between two iterations
        termination_eps = 1e-10
        #initial warp matrix
        warp_matrix_rep = np.eye(2, 3, dtype=np.float32)
        warp_matrix_img = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        number_of_iterations = iters
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix. If doesnt converge use EYE
        if iters > 0:
            try:
                (cc, warp_matrix_rep) = cv2.findTransformECC(rep1_gray, rep2_gray_transformed, warp_matrix_rep,
                                                             warp_mode, criteria, mask[:,:,0], 1)
            except:
                print('rep: didnt converge, using eye')
            try:
                (cc, warp_matrix_img) = cv2.findTransformECC(im1_gray, im2_transformed_gray, warp_matrix_img,
                                                             warp_mode, criteria, mask[:,:,0], 1)
            except:
                print('image: didnt converge, using eye')

        # GT results
        im2_transformed=im2_transformed*mask_valid
        im2_aligned_GT = cv2.warpAffine(im2_transformed, M, (sz[1], sz[0]), flags=cv2.WARP_INVERSE_MAP)
        psnr_rmse_final_GT = get_psnr_rmse(im2_aligned_GT, im2, mask)

        # Warping full image according to crop
        im2_aligned_img = cv2.warpAffine(im2_transformed, warp_matrix_img, (sz[1], sz[0]),
                                         flags=cv2.WARP_INVERSE_MAP)
        im2_aligned_rep = cv2.warpAffine(im2_transformed, warp_matrix_rep, (sz[1], sz[0]),
                                         flags=cv2.WARP_INVERSE_MAP)

        # Calc PSNR
        psnr_rmse_final_img = get_psnr_rmse(im2_aligned_img, im2, mask)
        psnr_rmse_final_rep = get_psnr_rmse(im2_aligned_rep, im2, mask)

        return psnr_rmse_final_img[0],psnr_rmse_final_rep[0],psnr_rmse_final_GT[0]


if __name__ == '__main__':
    with torch.no_grad():
        parser = arguments.get_parser()
        cudnn.benchmark = True
        #load args from parser
        args = prepare_train.initialize_test(parser)
        args = parser.parse_args()
        #save new parser args
        pretrained_model = args.pretrained_model
        use_cpu = args.use_cpu
        data_dir = args.data

        #load all other args from file
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
                        # print('arg.{}={}'.format(tokens[0], tokens[1]))
                        vars(args)[tokens[0]] = tokens[1]
        # eval args
        args.freeze_model = True
        args.batch_size = 1
        #set parser args
        args.pretrained_model = pretrained_model
        args.use_cpu = use_cpu
        args.data = data_dir

        args.output = args.pretrained_model.replace(".pth.tar", "registration_images")

        if not os.path.exists(args.output):
            os.makedirs(args.output)
        print('==> Format image pairs for registration\n')
        run_inf(args)
        print('==> Finished registration preparations \n')

        print('=> Starting Registration\n')

        #create model and set to eval
        invrep_net = prepare_train.create_model(args, device, inference_ref_padded=True)
        invrep_net.eval()

        # Read the images to be aligned
        main_dir = args.output
        images_all = os.listdir(main_dir)
        images=[]
        for l in images_all:
            if 'im0' in l:
                images.append(l.split('_im')[0])
        angles = np.array([2,10,18,26])
        iters = 1000

        #init containers
        num_images=len(images)
        psnr_rep_average = np.zeros([len(angles)])
        psnr_img_average = np.zeros([len(angles)])
        psnr_maddern_average = np.zeros([len(angles)])
        psnr_rep_average_diff = np.zeros([len(angles)])
        psnr_img_average_diff = np.zeros([len(angles)])
        psnr_maddern_average_diff = np.zeros([len(angles)])
        psnr_GT_average = np.zeros([len(angles)])

        #set shear parameter
        shear = 0.1

        #registartion loop
        for im_name in images:
            im1 = cv2.imread(os.path.join(main_dir,im_name+'_im0.png'))
            im2 = cv2.imread(os.path.join(main_dir,im_name+'_im1.png'))

            for ii,angle in enumerate(angles):
                print('image: {}, shear:{:1.2f}, angle: {}'.format(im_name, shear, angle))
                #calc registartion and psnr results
                psnr_rmse_final_img, psnr_rmse_final_rep, psnr_rmse_final_GT = run_registration_rep(
                    invrep_net,args, im1, im2, angle=angle, shear=shear, translation=0, iters=iters)
                #overall relative psnr
                psnr_img_average[ii] += psnr_rmse_final_img/num_images
                psnr_rep_average[ii] += psnr_rmse_final_rep/num_images
                psnr_GT_average[ii] += psnr_rmse_final_GT/num_images

                print('img average = {}, rep average = {}, GT average = {}'
                      .format(psnr_img_average[ii],psnr_rep_average[ii],psnr_GT_average[ii]))

        #print results
        print('Average PSNR')
        print(angles)
        print('img:')
        print(psnr_img_average)
        print('rep:')
        print(psnr_rep_average)
        print('GT:')
        print(psnr_GT_average)

        print('\n=> Finished Registration\n')
