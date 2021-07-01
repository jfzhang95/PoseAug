from __future__ import print_function, absolute_import, division

import glob

import torch

from models_baseline.gcn.graph_utils import adj_mx_from_skeleton
from models_baseline.gcn.sem_gcn import SemGCN
from models_baseline.mlp.linear_model import LinearModel, init_weights
from models_baseline.models_st_gcn.st_gcn_single_frame_test import WrapSTGCN
from models_baseline.videopose.model_VideoPose3D import TemporalModelOptimized1f


def model_pos_preparation(args, dataset, device):
    """
    return a posenet Model: with Bx16x2 --> posenet --> Bx16x3
    """
    # Create model
    num_joints = dataset.skeleton().num_joints()   # num_joints = 16 fix
    print('create model: {}'.format(args.posenet_name))

    if args.posenet_name == 'gcn':
        adj = adj_mx_from_skeleton(dataset.skeleton())
        model_pos = SemGCN(adj, 128, num_layers=args.stages, p_dropout=args.dropout, nodes_group=None).to(device)

    elif args.posenet_name == 'stgcn':
        model_pos = WrapSTGCN(p_dropout=args.dropout).to(device)

    elif args.posenet_name == 'mlp':
        model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3, num_stage=args.stages, p_dropout=args.dropout)

    elif args.posenet_name == 'videopose':
        filter_widths = [1]
        for stage_id in range(args.stages):
            filter_widths.append(1)  # filter_widths = [1, 1, 1, 1, 1]
        model_pos = TemporalModelOptimized1f(16, 2, 15, filter_widths=filter_widths, causal=False,
                                             dropout=0.25, channels=1024)
    else:
        assert False, 'posenet_name invalid'

    model_pos = model_pos.to(device)
    print("==> Total parameters for model {}: {:.2f}M"
          .format(args.posenet_name, sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    if args.pretrain:
        # pretrain path will be saved at ./checkpoint/pretrain_baseline/{}/{}/*/ckpt_best.pth.tar by default
        tmp_path = './checkpoint/pretrain_baseline/{}/{}/*/ckpt_best.pth.tar'.format(args.posenet_name, args.keypoints)
        posenet_pretrain_path = glob.glob(tmp_path)
        assert len(posenet_pretrain_path) == 1, 'suppose only 1 pretrain path for each model setting, ' \
                                                'please delete the redundant file'
        tmp_ckpt = torch.load(posenet_pretrain_path[0])
        model_pos.load_state_dict(tmp_ckpt['state_dict'])
        print('==> Pretrained posenet loaded')
    else:
        model_pos.apply(init_weights)

    return model_pos
