import time
import torch.utils.data
from train_test_functions.utils import tensor2array
from train_test_functions import loss_functions
from train_test_functions.logger import AverageMeter
import numpy as np
import torch
import torch.nn.functional as F
from train_test_functions.utils import square_center_crop
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_epoch(args, train_loader, invrep_net, optimizer, epoch_size, training_writer, epoch, n_iter):
    # meters for time and loss tracking
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=5)

    invrep_net.train()

    # start data time
    end = time.time()

    # loop over training images for 1 epoch
    for i, imgs in enumerate(train_loader):
        imgs = [im.to(device) for im in imgs]
        #patch triplet
        anchor_img = imgs[0]
        positive_img = imgs[1]
        negative_img = imgs[2]
        #crop border if enabled (border_len_ignore>0, default: border_len_ignore=0)
        valid_len = anchor_img.shape[3] - args.border_len_ignore*2

        # measure data loading time
        data_time.update(time.time() - end)
        #obtain triplet rep
        anchor_rep_full_size, positive_rep_full_size, negative_rep_full_size = invrep_net([anchor_img, positive_img, negative_img])
        #crop if needed
        anchor_rep=square_center_crop(anchor_rep_full_size,valid_len)
        positive_rep=square_center_crop(positive_rep_full_size,valid_len)
        negative_rep=square_center_crop(negative_rep_full_size,valid_len)
        anchor_img_valid = square_center_crop(anchor_img, valid_len)
        #rotation loss if enabled
        if args.rot_consistency_weight > 0:
            rot_vec=np.random.randint(3,size=(args.batch_size))+1
            assert(len(rot_vec)==anchor_img.shape[0])
            anchor_rot_img=torch.zeros_like(anchor_img)
            for b in range(args.batch_size):
                anchor_rot_img[b] = anchor_img[b].rot90(int(rot_vec[b]),(1,2))
            anchor_rot_rep = invrep_net([anchor_rot_img])[0]
        #scale loss if enabled
        if args.scale_consistency_weight > 0:
            assert(args.scale_consistency_max<=2 and args.scale_consistency_max>1)
            scale = 1 + torch.rand(1) * (args.scale_consistency_max-1)
            anchor_up_img_interped = F.interpolate(anchor_img, mode='bilinear', scale_factor=scale.item(), align_corners=False)
            anchor_up_img = square_center_crop(anchor_up_img_interped, anchor_img.shape[-1])
            anchor_up_rep = square_center_crop(invrep_net([anchor_up_img])[0],valid_len)

        # main loss
        loss, inter_loss, intra_loss, intra_secondary_loss, \
        dist_positive, dist_negative, anchor_positive_corr_dist, anchor_negative_corr_dist = \
            loss_functions.triplet_loss(anchor_rep, positive_rep, negative_rep, args)
        # regularization losses
        if args.scale_consistency_weight > 0:
            scale_loss = args.scale_consistency_weight * loss_functions.scale_consistency(anchor_rep, anchor_up_rep, scale, args)
            loss += scale_loss

        if args.multi_channel_corr_weight > 0:
            multi_channel_corr_loss = args.multi_channel_corr_weight * loss_functions.multi_channel_corr_loss([anchor_img_valid], [anchor_rep])
            loss += multi_channel_corr_loss

        if args.rot_consistency_weight > 0:
            rot_loss = args.rot_consistency_weight * loss_functions.rot_consistency(
                anchor_rep, anchor_rot_rep,rot_vec, args)
            loss += rot_loss

        losses.update(loss.item(), args.batch_size)
        # compute gradient and do Adam step
        optimizer.zero_grad()
        # loss.register_hook(lambda grad: print(anchor_rep))
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with torch.no_grad():
            # SCALARS logging
            if epoch % args.print_freq_epoch == 0 and i % args.print_freq == 0 and n_iter > 0:
                print('Train: Epoch={}, Iter={:}, N_iter={:}. AlgTime={:.3f}, DataTime={:.3f}, AvgLoss={:.4f}'.format(
                    epoch, i, n_iter, batch_time.avg[0], data_time.avg[0], losses.avg[0]))
                # main losses
                training_writer.add_scalar('1-loss/1-total_loss', loss.item(), n_iter)
                training_writer.add_scalar('1-loss/2-triplet_loss', inter_loss.item(), n_iter)
                training_writer.add_scalar('1-loss/3-intra_loss', intra_loss.item(), n_iter)
                if args.intra_secondary_weight > 0:
                    training_writer.add_scalar('1-loss/3-intra_secondary_loss', intra_secondary_loss.item(), n_iter)

                if args.scale_consistency_weight > 0:
                    training_writer.add_scalar('1-loss/2-scale_const-loss', scale_loss, n_iter)
                if args.multi_channel_corr_weight > 0:
                    training_writer.add_scalar('1-loss/2-multi_channel_corr-loss', multi_channel_corr_loss, n_iter)
                if args.rot_consistency_weight > 0:
                    training_writer.add_scalar('1-loss/2-rot_loss', rot_loss.item(), n_iter)

                # distances
                training_writer.add_scalar('2-dist_rep/1-dist_positive', dist_positive.mean(), n_iter)
                training_writer.add_scalar('2-dist_rep/2-dist_negative', dist_negative.mean(), n_iter)
                training_writer.add_scalar('2-dist_rep/3-diff-dist_positive-dist_negative',
                                           (dist_positive.mean() - dist_negative.mean()), n_iter)
                training_writer.add_scalar('2-dist_rep/4-corrdist_positive', anchor_positive_corr_dist.mean(), n_iter)
                training_writer.add_scalar('2-dist_rep/5-corrdist_negative', anchor_negative_corr_dist.mean(), n_iter)
                training_writer.add_scalar('2-dist_rep/6-diff-corrdist_positive-corrdist_negative',
                                           (anchor_positive_corr_dist.mean() - anchor_negative_corr_dist.mean()), n_iter)

            # IMAGES logging
            if epoch % args.print_freq_epoch == 0 and i % args.print_freq == 0 and n_iter > 0:
                training_writer.add_image('1-input_images/1-anchor_img', tensor2array(args, anchor_img[0], norm_mm=[0]),n_iter)
                training_writer.add_image('1-input_images/2-positive_img', tensor2array(args, positive_img[0], norm_mm=[0]),n_iter)
                training_writer.add_image('1-input_images/3-negative_img', tensor2array(args, negative_img[0], norm_mm=[0]),n_iter)
                dfunc = loss_functions.get_dfunc(args)
                training_writer.add_image('1-input_images/4-dist_anchor_positive_range_0-1',
                                          tensor2array(args, dfunc(anchor_img[0],positive_img[0]).mean(0), norm_mm=[0,1]), n_iter)
                training_writer.add_image('1-input_images/5-dist_anchor_negative_range_0-1',
                                          tensor2array(args, dfunc(anchor_img[0],negative_img[0]).mean(0), norm_mm=[0,1]), n_iter)

                if anchor_rep.shape[1] == 3:
                    training_writer.add_image('1-output_rgb/1-anchor_img', tensor2array(args, anchor_rep[0], norm_mm=args.norm_mm_by_act),n_iter)
                    training_writer.add_image('1-output_rgb/2-positive_img',tensor2array(args, positive_rep[0], norm_mm=args.norm_mm_by_act), n_iter)
                    training_writer.add_image('1-output_rgb/3-negative_img',tensor2array(args, negative_rep[0], norm_mm=args.norm_mm_by_act), n_iter)
                for ch in range(anchor_rep.shape[1]):
                    training_writer.add_image('2-output_ch{}/1-anchor'.format(ch),
                                              tensor2array(args, anchor_rep[0, ch], norm_mm=args.norm_mm_by_act), n_iter)
                    training_writer.add_image('2-output_ch{}/2-positive'.format(ch),
                                              tensor2array(args, positive_rep[0, ch], norm_mm=args.norm_mm_by_act), n_iter)
                    training_writer.add_image('2-output_ch{}/3-negative'.format(ch),
                                              tensor2array(args, negative_rep[0, ch], norm_mm=args.norm_mm_by_act), n_iter)
                    training_writer.add_image('2-output_ch{}/4-dist_anchor_positive_range_0-1'.format(ch),
                                              tensor2array(args, dist_positive[0, ch], norm_mm=[0, 1]), n_iter)
                    training_writer.add_image('2-output_ch{}/5-dist_anchor_negative_range_0-1'.format(ch),
                                              tensor2array(args, dist_negative[0, ch], norm_mm=[0, 1]), n_iter)
                    max95ap = 0.95 * torch.max(dist_positive[0,ch]).item()
                    training_writer.add_image('2-output_ch{}/6-dist_anchor_positive_range_0-max95ap'.format(ch),
                                              tensor2array(args, dist_positive[0, ch], norm_mm=[0, max95ap]), n_iter)
                    training_writer.add_image('2-output_ch{}/7-dist_anchor_negative_range_0-max95ap'.format(ch),
                                              tensor2array(args, dist_negative[0, ch], norm_mm=[0, max95ap]), n_iter)

        if i >= epoch_size - 1:
            break
        n_iter += 1

    return losses.avg[0], n_iter


def validate_epoch(args, val_loader, invrep_net, epoch_size, validation_writer, epoch):
    # meters for time and loss tracking
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=5)

    invrep_net.eval()

    # start data time
    end = time.time()
    with torch.no_grad():
        # loop over training images for 1 epoch
        for i, imgs in enumerate(val_loader):
            if i%2!=0:
                continue
            anchor_img = imgs[0].to(device).detach()
            positive_img = imgs[1].to(device).detach()

            # measure data loading time
            data_time.update(time.time() - end)

            anchor_rep = invrep_net([anchor_img])[0]
            positive_rep = invrep_net([positive_img])[0]
            #get distance function
            dfunc = loss_functions.get_dfunc(args)
            # get rep for logging
            anchor_rep_norm = torch.tensor(tensor2array(args, anchor_rep, norm_mm=args.norm_mm_by_act))
            positive_rep_norm = torch.tensor(tensor2array(args, positive_rep, norm_mm=args.norm_mm_by_act))
            #get distances for logging
            dist_rep_ap = dfunc(anchor_rep_norm, positive_rep_norm)
            corr_rep_ap = loss_functions.corr_dist_batch(anchor_rep_norm, positive_rep_norm)

            losses.update(dist_rep_ap.mean().item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # IMAGES logging
            if i==0 or i==4:
                if epoch == 0:
                    if args.out_channels == 1:
                        anchor_img=(anchor_img[:,0]*0.3+anchor_img[:,1]*0.59+anchor_img[:,2]*0.11).unsqueeze(1)
                        positive_img=(positive_img[:,0]*0.3+positive_img[:,1]*0.59+positive_img[:,2]*0.11).unsqueeze(1)
                    dist_img_ap = dfunc(anchor_img, positive_img)
                    validation_writer.add_image('val-{}-input_images/1-anchor_img'.format(i),
                                                tensor2array(args, anchor_img[0], norm_mm=[0]), epoch)
                    validation_writer.add_image('val-{}-input_images/2-positive_img'.format(i),
                                                tensor2array(args, positive_img[0], norm_mm=[0]), epoch)
                    validation_writer.add_image('val-{}-input_images/3-dist_anchor_positive_range_0-1'.format(i),
                                                tensor2array(args, dist_img_ap[0].mean(dim=0), norm_mm=[0,1]), epoch)
                    max95 = 0.95 * torch.max(dist_img_ap[0].mean(dim=0))
                    validation_writer.add_image('val-{}-input_images/4-dist_anchor_positive_norm95'.format(i),
                                                tensor2array(args, dist_img_ap[0].mean(dim=0), norm_mm=[0, max95]), epoch)
                    validation_writer.add_histogram('val_hist_0-rgb_idx{}/anchor_img'.format(i), anchor_img[0], epoch)
                    validation_writer.add_histogram('val_hist_0-rgb_idx{}/positive_img'.format(i), positive_img[0],epoch)
                if anchor_rep.shape[1] == 3:
                    validation_writer.add_image('val-{}-output_rgb/1-anchor_rep'.format(i),
                                                tensor2array(args, anchor_rep[0], norm_mm=args.norm_mm_by_act),epoch)
                    validation_writer.add_image('val-{}-output_rgb/2-positive_rep'.format(i),
                                                tensor2array(args, positive_rep[0], norm_mm=args.norm_mm_by_act), epoch)
                    validation_writer.add_image('val-{}-output_rgb/3-dist_anchor_positive_range_0-1'.format(i),
                                                tensor2array(args, dist_rep_ap[0].mean(dim=0), norm_mm=[0,1]), epoch)
                    max95 = 0.95 * torch.max(dist_rep_ap[0])
                    validation_writer.add_image('val-{}-output_rgb/4-dist_anchor_positive_norm95'.format(i),
                                                tensor2array(args, dist_rep_ap[0], norm_mm=[0, max95]), epoch)
                    validation_writer.add_histogram('val_hist_channels{}/anchor_rep'.format(i), anchor_rep[0], epoch)
                    validation_writer.add_histogram('val_hist_0-rgb_idx{}/anchor_rep'.format(i), anchor_rep[0], epoch)
                    validation_writer.add_histogram('val_hist_0-rgb_idx{}/positive_rep'.format(i), positive_rep[0], epoch)
                for ch in range(anchor_rep.shape[1]):
                    validation_writer.add_image('val-{}-output_ch_{}/1-anchor'.format(i,ch),
                                                tensor2array(args, anchor_rep[0, ch], norm_mm=args.norm_mm_by_act), epoch)
                    validation_writer.add_image('val-{}-output_ch_{}/2-positive'.format(i,ch),
                                                tensor2array(args, positive_rep[0, ch], norm_mm=args.norm_mm_by_act), epoch)
                    validation_writer.add_image('val-{}-output_ch_{}/3-dist_anchor_positive_range_0-1'.format(i,ch),
                                                tensor2array(args, dist_rep_ap[0, ch], norm_mm=[0, 1]), epoch)
                    max95 = 0.95 * torch.max(dist_rep_ap[0, ch])
                    validation_writer.add_image('val-{}-output_ch_{}/4-dist_anchor_positive_norm95'.format(i,ch),
                                                tensor2array(args, dist_rep_ap[0, ch], norm_mm=[0, max95]), epoch)
                    validation_writer.add_histogram('val_hist_channels{}/anchor_rep'.format(i), anchor_rep[0], epoch)
                    validation_writer.add_histogram('val_hist_channels{}/positive_rep'.format(i), positive_rep[0], epoch)
    # Final results - SCALARS logging
    validation_writer.add_scalar('val-loss/1-dist_rep_ap', dist_rep_ap.mean(), epoch)
    validation_writer.add_scalar('val-loss/2-corr_rep_ap', corr_rep_ap.mean(), epoch)

    return losses.avg[0], data_time.avg[0], batch_time.avg[0],
