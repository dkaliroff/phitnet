import torch.backends.cudnn as cudnn
import torch.utils.data
from train_test_functions import prepare_train
import os
from train_test_functions.train_test_run_inference_ import run_pm_images_rep as run_inf
from train_test_functions.run_patch_matching import run_patch_matching_3 as run_pm
import arguments
import argparse
'''
In order to test invrepnet representation:

This script receives a test data directory (formatted as the given test e.g. BT_TEST_100 and a pre_trained network model.
It formats the images for test and runs patch matching test.

To run this from command line:
python test_patchmatching.py /path/to/the/test/data/ --pretrained_model /path/to/the/pretrained/model/

***

In order to test with additional representations, call the script with the flag --additional_rep :
python test_patchmatching.py /path/to/the/test/data/ --additional_rep 

And format the test data directory as a directory containing only image paris named:
<name_1>_im0.<ext>, <name_1>_im1.<ext>
<name_2>_im0.<ext>, <name_3>_im1.<ext>
<name_3>_im0.<ext>, <name_3>_im1.<ext>
...

'''

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = arguments.get_parser()
    cudnn.benchmark = True
    args = prepare_train.initialize_test(parser)

    #regular invrepnet and original images mode
    if args.additional_rep==False:
        #LOADING ARGS FROM COMMAND LINE
        pretrained_model = args.pretrained_model
        use_cpu=args.use_cpu
        data_dir = args.data
        pm_mode = args.pm_mode

        #LOADING ARGS FROM FILE
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

        #EVAL ARGS
        args.freeze_model = True
        args.batch_size = 1

        # OVERRIDE ARGS FROM FILE WITH ARGS FROM COMMAND LINE
        args.pretrained_model = pretrained_model
        args.data = data_dir
        args.use_cpu=use_cpu
        args.pm_mode = pm_mode

        # inference
        args.output = args.pretrained_model.replace(".pth.tar","_patchmatching_images")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        print('\n=> Starting Inference\n')
        run_inf(args)
        print('=> Finished Inference\n')

        # tasks
        args.task_image_dirs = os.path.join(args.output, 'img')
        args.task_invrep_dirs = os.path.join(args.output, 'rep')

        #start patch matching eval after inference
        print('=> Starting Template Matching\n')
        run_pm(pm_mode=args.pm_mode, args=args)
        print('=> Finished Template Matching\n')
    else:
        args.task_invrep_dirs = args.data
        args.task_image_dirs = args.data
        args.pm_mode=2
        print('=> Starting --ADDITIONAL REPRESENTATION-- Patch Matching\n')
        run_pm(pm_mode=args.pm_mode, args=args)
        print('=> Finished --ADDITIONAL REPRESENTATION-- Patch Matching\n')

if __name__ == '__main__':
    main()