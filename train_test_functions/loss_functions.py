from __future__ import division
import torch
import torch.nn.functional as F
import torch.nn as NN
from train_test_functions.utils import square_center_crop

def corr_dist_batch(x1, x2):
    return 1 - ((x1 * x2).sum(-1).sum(-1) * torch.rsqrt((x1 ** 2).sum(-1).sum(-1) * (x2 ** 2).sum(-1).sum(-1)))

def get_dfunc(args):
    def l1_dist(x1, x2):
        return torch.abs(x1 - x2)
    def l2_dist(x1, x2):
        return torch.pow(x1-x2,2)
    if args.dfunc == 'l2':
            return l2_dist
    if args.dfunc == 'l1':
        return l1_dist
    else:
        exit()

def triplet_loss(anchor, positive, negative, args):
    inter_margin=args.inter_margin
    intra_margin_ratio=args.intra_margin_ratio
    loss_ver=args.loss_ver
    dfunc = get_dfunc(args)

    #calulate elemtwise distances: l1 or l2^2
    anchor_positive_dist = dfunc(anchor, positive)
    anchor_negative_dist = dfunc(anchor, negative)
    anchor_positive_corr_dist = corr_dist_batch(anchor, positive)
    anchor_negative_corr_dist = corr_dist_batch(anchor, negative)
    #loss between anchor and positive
    intra_margin=inter_margin/intra_margin_ratio if intra_margin_ratio>0 else 0

    if loss_ver ==0:
        intra_loss= F.relu(anchor_positive_dist - intra_margin).mean()
        triplet_loss = (F.relu(anchor_positive_dist - anchor_negative_dist + inter_margin)).mean()
        intra_secondary_loss = (F.relu(anchor_positive_corr_dist - intra_margin)).mean()
    elif loss_ver ==1:
        intra_loss = (F.relu(anchor_positive_dist.mean(-1).mean(-1) - intra_margin)).mean()
        triplet_loss = (F.relu(anchor_positive_dist.mean(-1).mean(-1) - anchor_negative_dist.mean(-1).mean(-1) + inter_margin)).mean()
        intra_secondary_loss = (F.relu(anchor_positive_corr_dist.mean(-1).mean(-1) - intra_margin)).mean()
    elif loss_ver ==2:
        intra_loss = (F.relu(anchor_positive_dist - intra_margin)).mean()
        anchor_negative_dist_avg_per_batch=\
            anchor_negative_dist.mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,anchor_positive_dist.shape[2],anchor_positive_dist.shape[3])
        triplet_loss = (F.relu(anchor_positive_dist - anchor_negative_dist_avg_per_batch + inter_margin)).mean()
        intra_secondary_loss = (F.relu(anchor_positive_corr_dist - intra_margin)).mean()
    elif loss_ver ==3:
        intra_loss = (F.relu(anchor_positive_corr_dist - intra_margin)).mean()
        triplet_loss = (F.relu(anchor_positive_corr_dist - anchor_negative_corr_dist + inter_margin)).mean()
        intra_secondary_loss = (F.relu(anchor_positive_dist.mean(-1).mean(-1) - intra_margin)).mean()
    #get final mean triplet loss over the positive valid triplets
    weighted_triplet_loss=args.inter_weight * triplet_loss
    weighted_intra_loss=args.intra_weight * intra_loss
    weighted_intra_secondary_loss = args.intra_secondary_weight * intra_secondary_loss
    loss = weighted_triplet_loss + weighted_intra_loss + weighted_intra_secondary_loss

    return loss, weighted_triplet_loss, weighted_intra_loss, weighted_intra_secondary_loss,\
           anchor_positive_dist, anchor_negative_dist, anchor_positive_corr_dist, anchor_negative_corr_dist

def multi_channel_corr_loss(imgs,reps):
    cos_sim = NN.CosineSimilarity(dim=0, eps=1e-6)
    def cos_sq(x,y):
        return torch.pow(cos_sim(torch.flatten(x),torch.flatten(y)),2)
    oo_channel_corr_loss=0
    for ii in range(len(imgs)):
        re = reps[ii]
        for ch1 in range(re.shape[1]):
            for ch2 in range(ch1+1, re.shape[1]):
                oo_channel_corr_loss += cos_sq(re[:, ch1], re[:, ch2])
    return oo_channel_corr_loss

def rot_consistency(x, x_rot, rot_vec, args):
    x_rot_back = torch.zeros_like(x_rot)
    for b in range(args.batch_size):
        x_rot_back[b] = x_rot[b].rot90(int(4-rot_vec[b]), (1, 2))
    if args.rot_consistency_dfunc =='corr':
        loss=corr_dist_batch(x, x_rot_back).mean()
    elif args.rot_consistency_dfunc =='l12':
        dfunc = get_dfunc(args)
        loss = dfunc(x, x_rot_back).mean()
    assert (loss==loss)
    return loss

def scale_consistency(rep, rep_up_from_img_up, scale, args):
    dfunc = get_dfunc(args)
    rep_up_from_rep_interped=F.interpolate(rep, mode='bilinear', scale_factor=scale.item(), align_corners=False)
    rep_up_from_rep=square_center_crop(rep_up_from_rep_interped,rep_up_from_img_up.shape[-1])
    if args.scale_consistency_dfunc =='corr':
        return corr_dist_batch(rep_up_from_rep, rep_up_from_img_up).mean()
    elif args.scale_consistency_dfunc =='l12':
        return dfunc(rep_up_from_rep, rep_up_from_img_up).mean()
    elif args.scale_consistency_dfunc =='l12normed':
        p=2 if args.dfunc=='l2' else 1
        return dfunc(rep_up_from_rep/rep_up_from_rep.norm(p=p), rep_up_from_img_up/rep_up_from_img_up.norm(p=p)).mean()

