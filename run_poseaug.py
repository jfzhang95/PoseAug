from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from function_baseline.model_pos_preparation import model_pos_preparation
from function_poseaug.config import get_parse_args
from function_poseaug.data_preparation import data_preparation
from function_poseaug.dataloader_update import dataloader_update
from function_poseaug.model_gan_preparation import get_poseaug_model
from function_poseaug.model_gan_train import train_gan
from function_poseaug.model_pos_eval import evaluate_posenet
from function_poseaug.model_pos_train import train_posenet
from utils.gan_utils import Sample_from_Pool
from utils.log import Logger
from utils.utils import save_ckpt, Summary, get_scheduler

'''
This code is used to train PoseAug model 
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
'''


def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_preparation(args)

    print("==> Creating PoseNet model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    model_pos_eval = model_pos_preparation(args, data_dict['dataset'], device)  # used for evaluation only
    # prepare optimizer for posenet
    posenet_optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr_p)
    posenet_lr_scheduler = get_scheduler(posenet_optimizer, policy='lambda', nepoch_fix=0,
                                         nepoch=args.epochs)

    print("==> Creating PoseAug model...")
    poseaug_dict = get_poseaug_model(args, data_dict['dataset'])

    # loss function
    criterion = nn.MSELoss(reduction='mean').to(device)

    # GAN trick: data buffer for fake data
    fake_3d_sample = Sample_from_Pool()
    fake_2d_sample = Sample_from_Pool()

    args.checkpoint = path.join(args.checkpoint, args.posenet_name, args.keypoints,
                              datetime.datetime.now().isoformat() + '_' + args.note)
    os.makedirs(args.checkpoint, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(args.checkpoint))

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), args)
    logger.record_args(str(model_pos))
    logger.set_names(['epoch', 'lr', 'error_h36m_p1', 'error_h36m_p2', 'error_3dhp_p1', 'error_3dhp_p2'])

    # Init monitor for net work training
    #########################################################
    summary = Summary(args.checkpoint)
    writer = summary.create_summary()

    ##########################################################
    # start training
    ##########################################################
    start_epoch = 0
    dhpp1_best = None
    s911p1_best = None

    for _ in range(start_epoch, args.epochs):

        if summary.epoch == 0:
            # evaluate the pre-train model for epoch 0.
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                  summary, writer, tag='_fake')
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                  summary, writer, tag='_real')
            summary.summary_epoch_update()

        # update train loader
        dataloader_update(args=args, data_dict=data_dict, device=device)

        # Train for one epoch
        train_gan(args, poseaug_dict, data_dict, model_pos, criterion, fake_3d_sample, fake_2d_sample, summary, writer)

        if summary.epoch > args.warmup:
            train_posenet(model_pos, data_dict['train_fake2d3d_loader'], posenet_optimizer, criterion, device)
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                  summary, writer, tag='_fake')

            train_posenet(model_pos, data_dict['train_det2d3d_loader'], posenet_optimizer, criterion, device)
            h36m_p1, h36m_p2, dhp_p1, dhp_p2 = evaluate_posenet(args, data_dict, model_pos, model_pos_eval, device,
                                                                  summary, writer, tag='_real')
        # Update learning rates
        ########################
        poseaug_dict['scheduler_G'].step()
        poseaug_dict['scheduler_d3d'].step()
        poseaug_dict['scheduler_d2d'].step()
        posenet_lr_scheduler.step()
        lr_now = posenet_optimizer.param_groups[0]['lr']
        print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))

        # Update log file
        logger.append([summary.epoch, lr_now, h36m_p1, h36m_p2, dhp_p1, dhp_p2])

        # Update checkpoint
        if dhpp1_best is None or dhpp1_best > dhp_p1:
            dhpp1_best = dhp_p1
            logger.record_args("==> Saving checkpoint at epoch '{}', with dhp_p1 {}".format(summary.epoch, dhpp1_best))
            save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_dhp_p1')

        if s911p1_best is None or s911p1_best > h36m_p1:
            s911p1_best = h36m_p1
            logger.record_args("==> Saving checkpoint at epoch '{}', with s911p1 {}".format(summary.epoch, s911p1_best))
            save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_h36m_p1')

        summary.summary_epoch_update()

    writer.close()
    logger.close()


if __name__ == '__main__':
    args = get_parse_args()

    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    main(args)
