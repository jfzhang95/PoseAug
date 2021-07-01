from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce


#####################################
# data loader with four output
#####################################
class PoseDataSet(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, cams):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)
        self._cams = np.concatenate(cams)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        assert self._poses_3d.shape[0] == self._cams.shape[0]
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]
        out_cam = self._cams[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action, out_cam

    def __len__(self):
        return len(self._actions)


#####################################
# data loader with two output
#####################################
class PoseBuffer(Dataset):
    def __init__(self, poses_3d, poses_2d, score=None):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        print('Generating {} poses...'.format(self._poses_3d.shape[0]))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d

    def __len__(self):
        return len(self._poses_2d)


#############################################################
# data loader for GAN
#############################################################
class PoseTarget(Dataset):
    def __init__(self, poses):
        assert poses is not None
        self._poses = np.concatenate(poses)
        print('Generating {} poses...'.format(self._poses.shape[0]))

    def __getitem__(self, index):
        out_pose = self._poses[index]
        out_pose = torch.from_numpy(out_pose).float()
        return out_pose

    def __len__(self):
        return len(self._poses)


class PoseTarget3D(Dataset):
    def __init__(self, poses_3d):
        assert poses_3d is not None
        self._poses_3d = np.concatenate(poses_3d)
        print('Generating {} poses...'.format(self._poses_3d.shape[0]))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        return out_pose_3d

    def __len__(self):
        return len(self._poses_3d)


class PoseTarget2D(Dataset):
    def __init__(self, poses_2d):
        assert poses_2d is not None
        poses_2d = np.concatenate(poses_2d)
        tmp_mask = np.ones((poses_2d.shape[0], poses_2d.shape[1], 1), dtype='float32')
        self._poses_2d = np.concatenate((poses_2d, tmp_mask), axis=2)
        print('Generating {} poses...'.format(self._poses_2d.shape[0]))

    def __getitem__(self, index):
        out_pose_2d = self._poses_2d[index]
        out_pose_2d = torch.from_numpy(out_pose_2d).float()
        return out_pose_2d[:, :-1], out_pose_2d[:, -1:]

    def __len__(self):
        return len(self._poses_2d)

