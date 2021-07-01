import os

import matplotlib.pyplot as plt

from common.camera import project_to_2d
from common.viz import show3Dpose, show3DposePair, show2Dpose


def plot_poseaug(inputs_3d, inputs_2d, g_rlt, cam_param, epoch, iter, args):
    outputs_3d_ba = g_rlt['pose_ba']
    outputs_3d_bl = g_rlt['pose_bl']
    outputs_3d_rt = g_rlt['pose_rt']

    outputs_2d_ba = project_to_2d(outputs_3d_ba, cam_param)  # fake 2d data
    outputs_2d_bl = project_to_2d(outputs_3d_bl, cam_param)  # fake 2d data
    outputs_2d_rt = project_to_2d(outputs_3d_rt, cam_param)  # fake 2d data

    # plot the augmented pose from origin -> ba -> bl -> rt
    _plot_poseaug(
        inputs_3d.cpu().detach().numpy()[0], inputs_2d.cpu().detach().numpy()[0],
        outputs_3d_ba.cpu().detach().numpy()[0], outputs_2d_ba.cpu().detach().numpy()[0],
        outputs_3d_bl.cpu().detach().numpy()[0], outputs_2d_bl.cpu().detach().numpy()[0],
        outputs_3d_rt.cpu().detach().numpy()[0], outputs_2d_rt.cpu().detach().numpy()[0],
        epoch, iter, args
    )



def _plot_poseaug(
        tmp_inputs_3d, tmp_inputs_2d,
        tmp_outputs_3d_ba, tmp_outputs_2d_ba,
        tmp_outputs_3d_bl, tmp_outputs_2d_bl,
        tmp_outputs_3d_rt, tmp_outputs_2d_rt,
        epoch, iter, args
):
    # plot all the rlt
    fig3d = plt.figure(figsize=(16, 8))

    # input 3d
    ax3din = fig3d.add_subplot(2, 4, 1, projection='3d')
    ax3din.set_title('input 3D')
    show3Dpose(tmp_inputs_3d, ax3din, gt=False)

    # show source 2d
    ax2din = fig3d.add_subplot(2, 4, 5)
    ax2din.set_title('input 2d')
    show2Dpose(tmp_inputs_2d, ax2din)

    # input 3d to modify 3d
    ax3dba = fig3d.add_subplot(2, 4, 2, projection='3d')
    ax3dba.set_title('input/ba 3d')
    show3DposePair(tmp_inputs_3d, tmp_outputs_3d_ba, ax3dba)

    # show source 2d
    ax2dba = fig3d.add_subplot(2, 4, 6)
    ax2dba.set_title('ba 2d')
    show2Dpose(tmp_outputs_2d_ba, ax2dba)

    # input 3d to modify 3d
    ax3dbl = fig3d.add_subplot(2, 4, 3, projection='3d')
    ax3dbl.set_title('ba/bl 3d')
    show3DposePair(tmp_outputs_3d_ba, tmp_outputs_3d_bl, ax3dbl)

    # show source 2d
    ax2dbl = fig3d.add_subplot(2, 4, 7)
    ax2dbl.set_title('bl 2d')
    show2Dpose(tmp_outputs_2d_bl, ax2dbl)

    # modify 3d to rotated 3d
    ax3drt = fig3d.add_subplot(2, 4, 4, projection='3d')
    ax3drt.set_title('modify 3d - rt')
    show3Dpose(tmp_outputs_3d_rt, ax3drt, gt=False)

    # rt 3d to 2d
    ax2d = fig3d.add_subplot(2, 4, 8)
    ax2d.set_title('rt 2d')
    show2Dpose(tmp_outputs_2d_rt, ax2d)

    os.makedirs('{}/poseaug_viz'.format(args.checkpoint), exist_ok=True)
    image_name = '{}/poseaug_viz/epoch_{:0>4d}_iter_{:0>4d}.png'.format(args.checkpoint, epoch, iter)
    plt.savefig(image_name)
    plt.close('all')
