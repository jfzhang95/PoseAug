from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torchgeometry as tgm

from utils.gan_utils import get_bone_lengthbypose3d, get_bone_unit_vecbypose3d, \
    get_pose3dbyBoneVec, blaugment9to15


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(inplace=True)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)

        return y


######################################################
###################  START  ##########################
######################################################
class PoseGenerator(nn.Module):
    def __init__(self, args, input_size=16 * 3):
        super(PoseGenerator, self).__init__()
        self.BAprocess = BAGenerator(input_size=input_size)
        self.BLprocess = BLGenerator(input_size=input_size, blr_tanhlimit=args.blr_tanhlimit)
        self.RTprocess = RTGenerator(input_size=input_size)

    def forward(self, inputs_3d):
        '''
        input: 3D pose
        :param inputs_3d: nx16x3, with hip root
        :return: nx16x3
        '''
        pose_ba, ba_diff = self.BAprocess(inputs_3d)  # diff may be used for div loss
        pose_bl, blr = self.BLprocess(inputs_3d, pose_ba)  # blr used for debug
        pose_rt, rt = self.RTprocess(inputs_3d, pose_bl)  # rt=(r,t) used for debug

        return {'pose_ba': pose_ba,
                'ba_diff': ba_diff,
                'pose_bl': pose_bl,
                'blr': blr,
                'pose_rt': pose_rt,
                'rt': rt}


######################################################
###################  END  ############################
######################################################

class BAGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=48, linear_size=256, num_stage=2, p_dropout=0.5):
        super(BAGenerator, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size + self.noise_channle, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.input_size - 3)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d):
        '''
        :param inputs_3d: nx16x3.
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]  # x: root relative

        # extract length, unit bone vec
        bones_unit = get_bone_unit_vecbypose3d(x)
        bones_length = get_bone_lengthbypose3d(x)

        # pre-processing
        x = x.view(x.size(0), -1)
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)

        y = self.w1(torch.cat((x, noise), dim=1))
        y = self.batch_norm1(y)
        y = self.relu(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)
        y = y.view(x.size(0), -1, 3)

        # modify the bone angle with length unchanged.
        modifyed = bones_unit + y
        modifyed_unit = modifyed / torch.norm(modifyed, dim=2, keepdim=True)

        # fix bone segment from pelvis to thorax to avoid pure rotation of whole body without ba changes.
        tmp_mask = torch.ones_like(bones_unit)
        tmp_mask[:, [6, 7], :] = 0.
        modifyed_unit = modifyed_unit * tmp_mask + bones_unit * (1 - tmp_mask)

        cos_angle = torch.sum(modifyed_unit * bones_unit, dim=2)
        ba_diff = 1 - cos_angle

        modifyed_bone = modifyed_unit * bones_length

        # convert bone vec back to 3D pose
        out = get_pose3dbyBoneVec(modifyed_bone) + root_origin

        return out, ba_diff


class RTGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=48, linear_size=256, num_stage=2, p_dropout=0.5):
        super(RTGenerator, self).__init__()
        '''
        :param input_size: n x 16 x 3
        :param output_size: R T 3 3 -> get new pose for pose 3d projection.
        '''
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle

        # 3d joints
        self.input_size = input_size  # 16 * 3

        # process input to linear size -> for R
        self.w1_R = nn.Linear(self.input_size + self.noise_channle, self.linear_size)
        self.batch_norm_R = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_R = []
        for l in range(num_stage):
            self.linear_stages_R.append(Linear(self.linear_size))
        self.linear_stages_R = nn.ModuleList(self.linear_stages_R)

        # process input to linear size -> for T
        self.w1_T = nn.Linear(self.input_size + self.noise_channle, self.linear_size)
        self.batch_norm_T = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_T = []
        for l in range(num_stage):
            self.linear_stages_T.append(Linear(self.linear_size))
        self.linear_stages_T = nn.ModuleList(self.linear_stages_T)

        # post processing
        self.w2_R = nn.Linear(self.linear_size, 3)
        self.w2_T = nn.Linear(self.linear_size, 3)

        self.relu = nn.LeakyReLU(inplace=True)
        # self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, inputs_3d, augx):
        '''
        :param inputs_3d: nx16x3
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]  # x: root relative

        # pre-processing
        x = x.view(x.size(0), -1)

        # caculate R
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        r = self.w1_R(torch.cat((x, noise), dim=1))
        r = self.batch_norm_R(r)
        r = self.relu(r)
        # r = self.dropout(r)
        for i in range(self.num_stage):
            r = self.linear_stages_R[i](r)

        r = self.w2_R(r)
        r = nn.Tanh()(r) * 3.1415
        r = r.view(x.size(0), 3)
        rM = tgm.angle_axis_to_rotation_matrix(r)[..., :3, :3]  # Nx4x4->Nx3x3 rotation matrix

        # caculate T
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        t = self.w1_T(torch.cat((x, noise), dim=1))
        t = self.batch_norm_T(t)
        t = self.relu(t)
        for i in range(self.num_stage):
            t = self.linear_stages_T[i](t)

        t = self.w2_T(t)
        t[:, 2] = t[:, 2].clone() * t[:, 2].clone()
        t = t.view(x.size(0), 1, 3)  # Nx1x3 translation t

        # operat RT on original data - augx
        augx = augx - augx[:, :1, :]  # x: root relative
        augx = augx.permute(0, 2, 1).contiguous()
        augx_r = torch.matmul(rM, augx)
        augx_r = augx_r.permute(0, 2, 1).contiguous()
        augx_rt = augx_r + t

        return augx_rt, (r, t)  # return r t for debug


class BLGenerator(nn.Module):
    def __init__(self, input_size, noise_channle=48, linear_size=256, num_stage=2, p_dropout=0.5, blr_tanhlimit=0.2):
        super(BLGenerator, self).__init__()
        '''
        :param input_size: n x 16 x 3
        :param output_size: R T 3 3 -> get new pose for pose 3d projection.
        '''
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_channle = noise_channle
        self.blr_tanhlimit = blr_tanhlimit

        # 3d joints
        self.input_size = input_size + 15  # 16 * 3 + bl

        # process input to linear size -> for R
        self.w1_BL = nn.Linear(self.input_size + self.noise_channle, self.linear_size)
        self.batch_norm_BL = nn.BatchNorm1d(self.linear_size)

        self.linear_stages_BL = []
        for l in range(num_stage):
            self.linear_stages_BL.append(Linear(self.linear_size))
        self.linear_stages_BL = nn.ModuleList(self.linear_stages_BL)

        # post processing
        self.w2_BL = nn.Linear(self.linear_size, 9)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d, augx):
        '''
        :param inputs_3d: nx16x3
        :return: nx16x3
        '''
        # convert 3d pose to root relative
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]  # x: root relative

        # pre-processing
        x = x.view(x.size(0), -1)

        # caculate blr
        bones_length_x = get_bone_lengthbypose3d(x.view(x.size(0), -1, 3)).squeeze(2)  # 0907
        noise = torch.randn(x.shape[0], self.noise_channle, device=x.device)
        blr = self.w1_BL(torch.cat((x, bones_length_x, noise), dim=1))
        blr = self.batch_norm_BL(blr)
        blr = self.relu(blr)
        for i in range(self.num_stage):
            blr = self.linear_stages_BL[i](blr)

        blr = self.w2_BL(blr)

        # create a mask to filter out 8th blr to avoid ambiguity (tall person at far may have same 2D with short person at close point).
        tmp_mask = torch.from_numpy(np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]]).astype('float32')).to(blr.device)
        blr = blr * tmp_mask
        # operate BL modification on original data
        blr = nn.Tanh()(blr) * self.blr_tanhlimit  # allow +-20% length change.

        bones_length = get_bone_lengthbypose3d(augx)
        augx_bl = blaugment9to15(augx, bones_length, blr.unsqueeze(2))
        return augx_bl, blr  # return blr for debug


def random_bl_aug(x):
    '''
    :param x: nx16x3
    :return: nx16x3
    '''
    bl_15segs_templates_mdifyed = np.load('./data_extra/bone_length_npy/hm36s15678_bl_templates.npy')

    # convert 3d pose to root relative
    root = x[:, :1, :] * 1.0
    x = x - x[:, :1, :]

    # extract length, unit bone vec
    bones_unit = get_bone_unit_vecbypose3d(x)

    # prepare a bone length list for augmentation.
    tmp_idx = np.random.choice(bl_15segs_templates_mdifyed.shape[0], x.shape[0])
    bones_length = torch.from_numpy(bl_15segs_templates_mdifyed[tmp_idx].astype('float32')).unsqueeze(2)

    modifyed_bone = bones_unit * bones_length.to(x.device)

    # convert bone vec back to pose3d
    out = get_pose3dbyBoneVec(modifyed_bone)

    return out + root  # return the pose with position information.


if __name__ == '__main__':
    # test = Project_cam3d_to_cam2d()
    random_bl_aug(None)
    print('done')
