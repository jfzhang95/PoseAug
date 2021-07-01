import torch
import numpy as np

def blaugment9to15(x, bl, blr, num_bone=15):
    '''
    this function convert 9 blr to 15 blr, and apply to bl
    bl: b x joints-1 x 1
    blr: b x 9 x 1
    out: pose3d b x joints x 3
    '''
    blr9to15 = torch.Tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 7
        [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 9
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 12
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 13
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 14
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 15
    ]).transpose(1, 0)  # 9 x 15 matrix

    blr9to15 = blr9to15.to(blr.device)
    blr9to15 = blr9to15.repeat([blr.size(0), 1, 1]).view(blr.size(0), 9, 15)
    blr_T = blr.permute(0, 2, 1).contiguous()
    blr_15_T = torch.matmul(blr_T, blr9to15)
    blr_15 = blr_15_T.permute(0, 2, 1).contiguous()  # back to N x 15 x 1

    # convert 3d pose to root relative
    root = x[:, :1, :] * 1.0
    x = x - x[:, :1, :]

    # extract length, unit bone vec
    bones_unit = get_bone_unit_vecbypose3d(x)

    # prepare a bone length list for augmentation.
    bones_length = torch.mul(bl, blr_15) + bl  # res
    modifyed_bone = bones_unit * bones_length

    # convert bone vec back to pose3d
    out = get_pose3dbyBoneVec(modifyed_bone)

    return out + root  # return the pose with position information.


def get_pose3dbyBoneVec(bones, num_joints=16):
    '''
    convert bone vect to pose3d， inverse function of get_bone_vector
    :param bones:
    :return:
    '''
    Ctinverse = torch.Tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 basement
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0],  # 8 10
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0],  # 10 11
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0],  # 8 13
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0],  # 13 14
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1],  # 14 15
    ]).transpose(1, 0)

    Ctinverse = Ctinverse.to(bones.device)
    C = Ctinverse.repeat([bones.size(0), 1, 1]).view(-1, num_joints - 1, num_joints)
    bonesT = bones.permute(0, 2, 1).contiguous()
    pose3d = torch.matmul(bonesT, C)
    pose3d = pose3d.permute(0, 2, 1).contiguous()  # back to N x 16 x 3
    return pose3d


def get_BoneVecbypose3d(x, num_joints=16):
    '''
    convert 3D point to bone vector
    :param x: N x number of joint x 3
    :return: N x number of bone x 3  number of bone = number of joint - 1
    '''
    Ct = torch.Tensor([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],  # 8 10
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # 10 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],  # 8 13
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # 13 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # 14 15
    ]).transpose(1, 0)

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), 1, 1]).view(-1, num_joints, num_joints - 1)
    pose3 = x.permute(0, 2, 1).contiguous()  # 这里16x3变成3x16的话 应该用permute吧
    B = torch.matmul(pose3, C)
    B = B.permute(0, 2, 1)  # back to N x 15 x 3
    return B


def get_bone_lengthbypose3d(x, bone_dim=2):
    '''
    :param bone_dim: dim=2
    :return:
    '''
    bonevec = get_BoneVecbypose3d(x)
    bones_length = torch.norm(bonevec, dim=2, keepdim=True)
    return bones_length


def get_bone_unit_vecbypose3d(x, num_joints=16, bone_dim=2):
    bonevec = get_BoneVecbypose3d(x)
    bonelength = get_bone_lengthbypose3d(x)
    bone_unitvec = bonevec / bonelength
    return bone_unitvec


def get_discriminator_accuracy(prediction, label):
    '''
    this is to get discriminator accuracy for tensorboard
    input is tensor -> convert to numpy
    :param tensor_in: Bs x Score :: where score > 0.5 mean True.
    :return:
    '''
    # get numpy from tensor
    prediction = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    rlt = np.abs(prediction - label)
    rlt = np.where(rlt > 0.5, 0, 1)
    num_of_correct = np.sum(rlt)
    accuracy = num_of_correct / label.shape[0]
    return accuracy


import copy
# To store data in a pool and sample from it when it is full
# Shrivastava et al’s strategy
class Sample_from_Pool(object):
    def __init__(self, max_elements=4096):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items
