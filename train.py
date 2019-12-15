import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from train_test_functions import prepare_train
from train_test_functions import run_epoch
from train_test_functions import utils
import os
from train_test_functions.train_test_run_inference_ import run_pm_images_rep as run_inf
from train_test_functions.run_patch_matching import run_patch_matching_3 as run_pm
import arguments

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


'''
This script a training data directory and optional training arguments.
It trains a new invrepnet network according to the arguments (or default arguments of not given).

To run this from command line:
python train.py /path/to/the/training/data/ [--optional arguments]
'''

def main():
    #create parser
    parser = arguments.get_parser()
    #initialize global iteration counter
    n_iter = 0
    #cuda configuration
    cudnn.benchmark = True
    #initializing train from argumnents and creating args namespace
    training_writer, args = prepare_train.initialize_train(parser)
    #creating data loaders
    train_loader, val_loader, args = prepare_train.create_dataloaders(args)
    #create invrepnet model
    invrep_net = prepare_train.create_model(args, device)
    #create optimizer
    optimizer = prepare_train.create_optimizer(args, invrep_net)

    #train loop by epoch
    for epoch in range(args.epochs):
        # train for one epoch
        train_loss, n_iter = run_epoch.train_epoch(args, train_loader, invrep_net,
                                           optimizer, args.epoch_size, training_writer, epoch, n_iter)
        print(' * Epoch: {}, Avg Train Loss : {:.3f}'.format(epoch, train_loss))

        # loss evaluate on validation set
        if ((epoch+1) % args.validation_freq==0) and not args.skip_validation:
            val_loss, data_time, batch_time = run_epoch.validate_epoch(args, val_loader, invrep_net,
                                                args.epoch_size, training_writer, epoch)
            print('Val: Epoch={}, AlgTime={:.3f}, DataTime={:.3f}, AvgLoss={:.4f}'.format(epoch, batch_time, data_time, val_loss))

        # save checkpoint of model
        utils.save_checkpoint(args.save_path, {'epoch': epoch + 1, 'state_dict': invrep_net.module.state_dict()},
                              args.checkpoint_freq, epoch+1)

        #patch matching task validation
        if args.with_task_val:
            if (epoch+1) % args.checkpoint_freq == 0 or (epoch+1)==args.epochs:
                #inference
                print('==> Starting Inference\n')
                args.pretrained_model = os.path.join(args.save_path, 'invrep_checkpoint.pth.tar')
                args.output = os.path.join(args.save_path, 'inference_epoch_{}'.format(epoch+1))
                run_inf(args)
                print('==> Finished Inference\n')
                #directories for image saving
                args.task_image_dirs = os.path.join(args.output, 'img')
                args.task_invrep_dirs = os.path.join(args.output, 'rep')

                print('==> Starting Template Matching\n')
                pmres, methods=run_pm(pm_mode=args.pm_mode,args=args)
                for rr, m_res in enumerate(pmres):
                    training_writer.add_scalar('val-task/pm_{}_32'.format(methods[rr]), np.around(m_res[0],decimals=4), epoch+1)
                    training_writer.add_scalar('val-task/pm_{}_64'.format(methods[rr]), np.around(m_res[1],decimals=4), epoch+1)
                    training_writer.add_scalar('val-task/pm_{}_128'.format(methods[rr]), np.around(m_res[2],decimals=4), epoch+1)
                print('==> Finished Template Matching\n')

if __name__ == '__main__':
    main()