from __future__ import print_function, absolute_import, division

import torch

from models_baseline.mlp.linear_model import init_weights
from models_poseaug.PosDiscriminator import Pos2dDiscriminator, Pos3dDiscriminator
from models_poseaug.gan_generator import PoseGenerator
from utils.utils import get_scheduler


def get_poseaug_model(args, dataset):
    """
    return PoseAug augmentor and discriminator
    and corresponding optimizer and scheduler
    """
    # Create model: G and D
    print("==> Creating model...")
    device = torch.device("cuda")
    num_joints = dataset.skeleton().num_joints()

    # generator for PoseAug
    model_G = PoseGenerator(args, num_joints * 3).to(device)
    model_G.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_G.parameters()) / 1000000.0))

    # discriminator for 3D
    model_d3d = Pos3dDiscriminator(num_joints).to(device)
    model_d3d.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d3d.parameters()) / 1000000.0))

    # discriminator for 2D
    model_d2d = Pos2dDiscriminator(num_joints).to(device)
    model_d2d.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d2d.parameters()) / 1000000.0))

    # prepare optimizer
    g_optimizer = torch.optim.Adam(model_G.parameters(), lr=args.lr_g)
    d3d_optimizer = torch.optim.Adam(model_d3d.parameters(), lr=args.lr_d)
    d2d_optimizer = torch.optim.Adam(model_d2d.parameters(), lr=args.lr_d)

    # prepare scheduler
    g_lr_scheduler = get_scheduler(g_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d3d_lr_scheduler = get_scheduler(d3d_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)
    d2d_lr_scheduler = get_scheduler(d2d_optimizer, policy='lambda', nepoch_fix=0, nepoch=args.epochs)

    return {
        'model_G': model_G,
        'model_d3d': model_d3d,
        'model_d2d': model_d2d,
        'optimizer_G': g_optimizer,
        'optimizer_d3d': d3d_optimizer,
        'optimizer_d2d': d2d_optimizer,
        'scheduler_G': g_lr_scheduler,
        'scheduler_d3d': d3d_lr_scheduler,
        'scheduler_d2d': d2d_lr_scheduler,
    }
