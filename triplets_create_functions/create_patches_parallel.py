import os
import time
import sys
import multiprocessing
from joblib import Parallel, delayed
import argparse
from triplets_create_functions.worker_create_patches_parallel import process_scene
parser = argparse.ArgumentParser(description='Create patches from scenes - Damian',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#data
parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
parser.add_argument('--single_anchor', action='store_true', default=False, help='create with single anchor')
parser.add_argument('--ap_neighbors', action='store_true', default=False, help='anchor and positive must be neighbors (by file name)')
parser.add_argument('--crop_len', default=64, type=int, metavar='N', help='square crop length')
parser.add_argument('--step_len', default=8, type=int, metavar='N', help='step ratio for patches')
parser.add_argument('--apdist_thresh', default=0, type=float, metavar='N', help='anchor - positive distance threshold')
parser.add_argument('--andist_thresh', default=0, type=float, metavar='N', help='anchor - negative distance threshold')
parser.add_argument('--BT_apdist', default=0, type=float, metavar='N', help='anchor positive distance for files containing "BT" (for multiple file types)')
parser.add_argument('--std_thresh', default=0.1, type=float, metavar='N', help='minimum std for patch')
parser.add_argument('--crop_len_random', action='store_true', default=False, help='random crop len enable')
parser.add_argument('--extended_random', action='store_true', default=False, help='extended random patch choice')
parser.add_argument('--no_sort', action='store_true', default=False, help='dont sort file names')
parser.add_argument('--num_jobs', default=6, type=int, metavar='N', help='nunmber of threads for workers')


if __name__ == "__main__":

    '''
    create data script :  create triplets according to arguments in input, 
    works in parallel by running worker in threads
    '''

    args = parser.parse_args()
    scenes_folders = next(os.walk(args.data_dir))[1]
    scenes_folders.remove('VAL')
    scenes_list = []
    num_images = 0
    num_scenes = len(scenes_folders)
    for l, name in enumerate(scenes_folders):
        curr_scene=[]
        file_names = os.listdir(os.path.join(args.data_dir, name, 'data'))
        for s in file_names:
            if 'mask' not in s:
                curr_scene.append(os.path.join(name, 'data',s))
        scenes_list.append(curr_scene)
        num_images = num_images + len(scenes_list[l])


    print("{} scenes found, with a total of {} images in all the scenes".format(num_scenes, num_images))

    if "PYCHARM_HOSTED" in os.environ:
        num_jobs=1
    else:
        num_jobs = min(num_scenes,multiprocessing.cpu_count()//2) #work with half the cores
        if args.num_jobs is not None:
            num_jobs=args.num_jobs

    print("Processing triplets in {} jobs".format(num_jobs))

    scenes_list = scenes_list
    t = time.time()
    num_triplets_list  = Parallel(n_jobs=num_jobs)(delayed(process_scene)(
        scene,args.data_dir,args.single_anchor,args.ap_neighbors, args.crop_len, args.step_len, args.andist_thresh, args.apdist_thresh,
                                args.std_thresh, args.extended_random, args.crop_len_random, args.no_sort, args.BT_apdist) for scene in scenes_list)
    triplets_count=0
    total_std=0
    total_mean=0
    num_scenes=len(num_triplets_list)
    for nt in num_triplets_list:
        if nt[0]>0:
            triplets_count = triplets_count+nt[0]
            total_std = total_std+nt[1]/(nt[0]*num_scenes)
            total_mean = total_mean+nt[2]/(nt[0]*num_scenes)
    print("total number of triplets = {}".format(triplets_count))
    print("total std=")
    print(total_std)
    print("total mean=")
    print(total_mean)
    # stats_file = open(os.path.join(args.data_dir,"stats.txt"), "w")
    # stats_file.writelines(str(total_std)+'\n')
    # stats_file.writelines(str(total_mean))
    # stats_file.close()
    print("saving args")
    with open(os.path.join(args.data_dir,'args_in_run.txt'), 'w') as f:
        f.write("command line in run:\n")
        f.write(' '.join(sys.argv[1:]))
        f.write("\n\nargs in run:\n")
        for a in vars(args):
            f.write('--{} {}\n'.format(a, getattr(args, a)))
    f.close()

    print("*** FINISHED: elapsed time = {:.2f} minutes".format((time.time() - t)/60))

