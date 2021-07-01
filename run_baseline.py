from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from function_baseline.config import get_parse_args
from function_baseline.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation
from function_baseline.model_pos_train import train
from function_poseaug.model_pos_eval import evaluate
from utils.log import Logger, savefig
from utils.utils import save_ckpt

"""
this code is used to pretrain the baseline model
1. Simple Baseline
2. VideoPose
3. SemGCN
4. ST-GCN
code are modified from https://github.com/garyzhao/SemGCN
"""


def main(args):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_preparation(args)

    print("==> Creating PoseNet model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    print("==> Prepare optimizer...")
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

    ckpt_dir_path = path.join(args.checkpoint, args.posenet_name, args.keypoints,
                                   datetime.datetime.now().strftime('%m%d%H%M%S') + '_' + args.note)
    os.makedirs(ckpt_dir_path, exist_ok=True)
    print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

    logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'), args)
    logger.set_names(['epoch', 'lr', 'loss_train', 'error_h36m_p1', 'error_h36m_p2', 'error_3dhp_p1', 'error_3dhp_p2'])

    #################################################
    # ########## start training here
    #################################################
    start_epoch = 0
    error_best = None
    glob_step = 0
    lr_now = args.lr

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(data_dict['train_loader'], model_pos, criterion, optimizer, device, args.lr, lr_now,
                                              glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)

        # Evaluate
        error_h36m_p1, error_h36m_p2 = evaluate(data_dict['H36M_test'], model_pos, device)
        error_3dhp_p1, error_3dhp_p2 = evaluate(data_dict['3DHP_test'], model_pos, device, flipaug='_flip')

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_h36m_p1, error_h36m_p2, error_3dhp_p1, error_3dhp_p2])

        # Update checkpoint
        if error_best is None or error_best > error_h36m_p1:
            error_best = error_h36m_p1
            save_ckpt({'state_dict': model_pos.state_dict(), 'epoch': epoch + 1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'state_dict': model_pos.state_dict(), 'epoch': epoch + 1}, ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_h36m_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))
    return



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
    cudnn.benchmark = True

    main(args)
