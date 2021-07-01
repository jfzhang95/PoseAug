from __future__ import absolute_import, division

import os

import numpy as np
import torch
from tensorboardX import SummaryWriter


# self define tools
class Summary(object):
    def __init__(self, directory):
        self.directory = directory
        self.epoch = 0
        self.writer = None
        self.phase = 0
        self.train_iter_num = 0
        self.train_realpose_iter_num = 0
        self.train_fakepose_iter_num = 0
        self.test_iter_num = 0
        self.test_MPI3D_iter_num = 0

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return self.writer

    def summary_train_iter_num_update(self):
        self.train_iter_num = self.train_iter_num + 1

    def summary_train_realpose_iter_num_update(self):
        self.train_realpose_iter_num = self.train_realpose_iter_num + 1

    def summary_train_fakepose_iter_num_update(self):
        self.train_fakepose_iter_num = self.train_fakepose_iter_num + 1

    def summary_test_iter_num_update(self):
        self.test_iter_num = self.test_iter_num + 1

    def summary_test_MPI3D_iter_num_update(self):
        self.test_MPI3D_iter_num = self.test_MPI3D_iter_num + 1

    def summary_epoch_update(self):
        self.epoch = self.epoch + 1

    def summary_phase_update(self):
        self.phase = self.phase + 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])

    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)


def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


from torch.optim import lr_scheduler
def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler
