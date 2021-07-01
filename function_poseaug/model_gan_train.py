from __future__ import print_function, absolute_import, division

import time

import numpy as np
import torch
import torch.nn as nn
# add model for generator and discriminator
from torch.autograd import Variable
from torch.utils.data import DataLoader

from common.camera import project_to_2d
from common.data_loader import PoseDataSet
from function_poseaug.poseaug_viz import plot_poseaug
from progress.bar import Bar
from utils.gan_utils import get_discriminator_accuracy
from utils.loss import diff_range_loss, rectifiedL2loss
from utils.utils import AverageMeter, set_grad


def get_adv_loss(model_dis, data_real, data_fake, criterion, summary, writer, writer_name):
    device = torch.device("cuda")
    # Adversarial losses
    real_3d = model_dis(data_real)
    fake_3d = model_dis(data_fake)

    real_label_3d = Variable(torch.ones(real_3d.size())).to(device)
    fake_label_3d = Variable(torch.zeros(fake_3d.size())).to(device)

    # adv loss
    # adv_3d_loss = criterion(real_3d, fake_3d)    # choice either one

    adv_3d_real_loss = criterion(real_3d, fake_label_3d)
    adv_3d_fake_loss = criterion(fake_3d, real_label_3d)
    # Total discriminators losses
    adv_3d_loss = (adv_3d_real_loss + adv_3d_fake_loss) * 0.5

    # monitor training process
    ###################################################
    real_acc = get_discriminator_accuracy(real_3d.reshape(-1), real_label_3d.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_3d.reshape(-1), fake_label_3d.reshape(-1))
    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_adv_loss'.format(writer_name), adv_3d_loss.item(),
                      summary.train_iter_num)
    return adv_3d_loss


def train_dis(model_dis, data_real, data_fake, criterion, summary, writer, writer_name, fake_data_pool, optimizer):
    device = torch.device("cuda")
    optimizer.zero_grad()

    data_real = data_real.clone().detach().to(device)
    data_fake = data_fake.clone().detach().to(device)
    # store the fake buffer for discriminator training.
    data_fake = Variable(torch.Tensor(fake_data_pool(data_fake.cpu().detach().data.numpy()))).to(device)

    # predicte the label
    real_pre = model_dis(data_real)
    fake_pre = model_dis(data_fake)

    real_label = Variable(torch.ones(real_pre.size())).to(device)
    fake_label = Variable(torch.zeros(fake_pre.size())).to(device)
    dis_real_loss = criterion(real_pre, real_label)
    dis_fake_loss = criterion(fake_pre, fake_label)

    # Total discriminators losses
    dis_loss = (dis_real_loss + dis_fake_loss) * 0.5

    # record acc
    real_acc = get_discriminator_accuracy(real_pre.reshape(-1), real_label.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_pre.reshape(-1), fake_label.reshape(-1))

    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_dis_loss'.format(writer_name), dis_loss.item(), summary.train_iter_num)

    # Update generators
    ###################################################
    dis_loss.backward()
    nn.utils.clip_grad_norm_(model_dis.parameters(), max_norm=1)
    optimizer.step()
    return real_acc, fake_acc


def get_diff_loss(args, bart_rlt_dict, summary, writer):
    '''
    control the modification range
    '''
    diff_loss_dict = {}
    diff_log_dict = {}

    # regulation loss for bart to avoid gan collapse
    angle_diff = bart_rlt_dict['ba_diff']  # 'ba_diff' bx15;
    angle_diff_loss = diff_range_loss(torch.mean(angle_diff, dim=-1), args.ba_range_m, args.ba_range_w)

    diff_loss_dict['loss_diff_angle'] = angle_diff_loss.mean()
    diff_log_dict['log_angle_diff'] = angle_diff.detach().mean()  # record in cos_angle
    # record each bone angle
    for i in range(bart_rlt_dict['ba_diff'].shape[1]):
        diff_log_dict['log_angle@bone_{:0>2d}'.format(i)] = \
            torch.acos(torch.clamp((1 - angle_diff.detach())[:, i], -1, 1)).mean() * 57.29  # record in angle degree

    blr = bart_rlt_dict['blr']

    blr_loss = rectifiedL2loss(blr, args.blr_limit)  # blr_limit

    diff_loss_dict['loss_diff_blr'] = blr_loss.mean()
    diff_log_dict['log_diff_blr'] = blr.detach().mean()

    for key in diff_log_dict:
        writer.add_scalar('train_G_iter_diff_log/' + key, diff_log_dict[key].item(), summary.train_iter_num)

    loss = 0
    for key in diff_loss_dict:
        loss = loss + diff_loss_dict[key]
        writer.add_scalar('train_G_iter_diff_loss/' + key, diff_loss_dict[key].item(), summary.train_iter_num)
    return loss


def get_feedback_loss(args, model_pos, criterion, summary, writer,
                      inputs_2d, inputs_3d, outputs_2d_ba, outputs_3d_ba, outputs_2d_rt, outputs_3d_rt):
    def get_posenet_loss(input_pose_2d, target_pose_3d):
        predict_pose_3d = model_pos(input_pose_2d.view(num_poses, -1)).view(num_poses, -1, 3)
        target_pose_3d_rooted = target_pose_3d[:, :, :] - target_pose_3d[:, :1, :]  # ignore the 0 joint
        posenet_loss = torch.norm(predict_pose_3d - target_pose_3d_rooted, dim=-1)  # return b x j loss
        weights = torch.Tensor([1, 5, 2, 1, 5, 2, 1, 1, 1, 1, 5, 2, 1, 5, 2, 1]).to(target_pose_3d.device).unsqueeze(0)
        posenet_loss = posenet_loss * weights

        posenet_loss = torch.cat([
            torch.mean(posenet_loss[:, [1, 4, 10, 13]], dim=-1, keepdim=True),
            torch.mean(posenet_loss[:, [2, 5, 9, 11, 14]], dim=-1, keepdim=True),
            torch.mean(posenet_loss[:, [3, 6, 12, 15]], dim=-1, keepdim=True)
        ], dim=-1)
        return posenet_loss

    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    def fix_hardratio(target_std, taget_mean, harder, easier, gloss_factordiv, gloss_factorfeedback, tag=''):
        harder_value = harder / easier

        hard_std = torch.std(harder_value)
        hard_mean = torch.mean(harder_value)

        hard_div_loss = torch.mean((hard_std - target_std) ** 2)
        hard_mean_loss = diff_range_loss(harder_value, taget_mean, target_std)

        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_std'.format(tag), hard_std.mean().item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_mean'.format(tag), hard_mean.mean().item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_sample'.format(tag), harder_value[0].mean().item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_mean_loss'.format(tag), hard_mean_loss.item(),
                          summary.train_iter_num)
        writer.add_scalar('train_G_iter_posenet_feedback/{}_hard_std_loss'.format(tag), hard_div_loss.item(),
                          summary.train_iter_num)
        return hard_div_loss * gloss_factordiv + hard_mean_loss * gloss_factorfeedback

    # posenet loss: to generate harder case.
    # the flow: original pose --> pose BA --> pose RT
    ###################################################
    device = torch.device("cuda")
    num_poses = inputs_2d.shape[0]

    # outputs_2d_origin -> posenet -> outputs_3d_origin
    fake_pos_pair_loss_origin = get_posenet_loss(inputs_2d, inputs_3d)
    # outputs_2d_ba -> posenet -> outputs_3d_ba
    fake_pos_pair_loss_ba = get_posenet_loss(outputs_2d_ba, outputs_3d_ba)
    # # outputs_2d_rt -> posenet -> outputs_3d_rt
    fake_pos_pair_loss_rt = get_posenet_loss(outputs_2d_rt, outputs_3d_rt)

    # pair up posenet loss
    ##########################################
    hardratio_ba = update_hardratio(args.hardratio_ba_s, args.hardratio_ba, summary.epoch, args.epochs)
    hardratio_rt = update_hardratio(args.hardratio_rt_s, args.hardratio_rt, summary.epoch, args.epochs)

    # get feedback loss
    pos_pair_loss_baToorigin = fix_hardratio(args.hardratio_std_ba, hardratio_ba,
                                             fake_pos_pair_loss_ba, fake_pos_pair_loss_origin,
                                             args.gloss_factordiv_ba, args.gloss_factorfeedback_ba, tag='ba')
    pos_pair_loss_rtToorigin = fix_hardratio(args.hardratio_std_rt, hardratio_rt,
                                             fake_pos_pair_loss_rt, fake_pos_pair_loss_origin,
                                             args.gloss_factordiv_rt, args.gloss_factorfeedback_rt, tag='rt')

    feedback_loss = pos_pair_loss_baToorigin + pos_pair_loss_rtToorigin

    writer.add_scalar('train_G_iter_posenet_feedback/1) pos_pair_loss_origin', fake_pos_pair_loss_origin.mean().item(),
                      summary.train_iter_num)
    writer.add_scalar('train_G_iter_posenet_feedback/2) pos_pair_loss_ba', fake_pos_pair_loss_ba.mean().item(),
                      summary.train_iter_num)
    writer.add_scalar('train_G_iter_posenet_feedback/3) pos_pair_loss_rt', fake_pos_pair_loss_rt.mean().item(),
                      summary.train_iter_num)

    return feedback_loss


def train_gan(args, poseaug_dict, data_dict, model_pos, criterion, fake_3d_sample, fake_2d_sample, summary, writer):
    device = torch.device("cuda")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # extract necessary module for training.
    model_G = poseaug_dict['model_G']
    model_d3d = poseaug_dict['model_d3d']
    model_d2d = poseaug_dict['model_d2d']

    g_optimizer = poseaug_dict['optimizer_G']
    d3d_optimizer = poseaug_dict['optimizer_d3d']
    d2d_optimizer = poseaug_dict['optimizer_d2d']

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_G.train()
    model_d3d.train()
    model_d2d.train()
    model_pos.train()
    end = time.time()

    # prepare buffer list for update
    tmp_3d_pose_buffer_list = []
    tmp_2d_pose_buffer_list = []
    tmp_camparam_buffer_list = []

    bar = Bar('Train pose gan', max=len(data_dict['train_gt2d3d_loader']))
    for i, ((inputs_3d, _, _, cam_param), target_d2d, target_d3d) in enumerate(
            zip(data_dict['train_gt2d3d_loader'], data_dict['target_2d_loader'], data_dict['target_3d_loader'])):
        lr_now = g_optimizer.param_groups[0]['lr']

        ##################################################
        #######      Train Generator     #################
        ##################################################
        set_grad([model_d3d], False)
        set_grad([model_d2d], False)
        set_grad([model_G], True)
        set_grad([model_pos], False)
        g_optimizer.zero_grad()

        # Measure data loading time
        data_time.update(time.time() - end)

        inputs_3d, cam_param = inputs_3d.to(device), cam_param.to(device)
        inputs_2d = project_to_2d(inputs_3d, cam_param)

        # poseaug: BA BL RT
        g_rlt = model_G(inputs_3d)
        # extract the generator result
        outputs_3d_ba = g_rlt['pose_ba']
        outputs_3d_rt = g_rlt['pose_rt']

        outputs_2d_ba = project_to_2d(outputs_3d_ba, cam_param)  # fake 2d data
        outputs_2d_rt = project_to_2d(outputs_3d_rt, cam_param)  # fake 2d data

        # adv loss
        adv_3d_loss = get_adv_loss(model_d3d, inputs_3d, outputs_3d_ba, criterion, summary, writer, writer_name='g3d')
        adv_2d_loss = get_adv_loss(model_d2d, inputs_2d, outputs_2d_rt, criterion, summary, writer, writer_name='g2d')

        # diff loss. encourage diversity.
        ###################################################
        diff_loss = get_diff_loss(args, g_rlt, summary, writer)

        # posenet loss: to generate harder case.
        ###################################################
        feedback_loss = get_feedback_loss(args, model_pos, criterion, summary, writer,
                                          inputs_2d, inputs_3d, outputs_2d_ba, outputs_3d_ba, outputs_2d_rt,
                                          outputs_3d_rt)

        if summary.epoch > args.warmup:
            gen_loss = adv_2d_loss * args.gloss_factord2d + \
                       adv_3d_loss * args.gloss_factord3d + \
                       feedback_loss + \
                       diff_loss * args.gloss_factordiff
        else:
            gen_loss = adv_2d_loss * args.gloss_factord2d + \
                       adv_3d_loss * args.gloss_factord3d + \
                       diff_loss * args.gloss_factordiff

        writer.add_scalar('train_G_iter/gen_loss', gen_loss.item(), summary.train_iter_num)
        writer.add_scalar('train_G_iter/lr_now', lr_now, summary.train_iter_num)

        # Update generators
        ###################################################
        gen_loss.backward()
        nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1)
        g_optimizer.step()

        ##################################################
        #######      Train Discriminator     #############
        ##################################################
        if summary.train_iter_num % args.df == 0:
            set_grad([model_d3d], True)
            set_grad([model_d2d], True)
            set_grad([model_G], False)
            set_grad([model_pos], False)

            # d3d training
            train_dis(model_d3d, target_d3d, outputs_3d_ba, criterion, summary, writer, writer_name='d3d',
                      fake_data_pool=fake_3d_sample, optimizer=d3d_optimizer)
            # d2d training
            train_dis(model_d2d, target_d2d, outputs_2d_rt, criterion, summary, writer, writer_name='d2d',
                      fake_data_pool=fake_2d_sample, optimizer=d2d_optimizer)

        ##############################################
        # save fake data buffer for posenet training #
        ##############################################
        # here add a check so that outputs_2d_rt that out of box will be remove.
        valid_rt_idx = torch.sum(outputs_2d_rt > 1, dim=(1, 2)) < 1
        tmp_3d_pose_buffer_list.append(outputs_3d_rt.detach()[valid_rt_idx].cpu().numpy())
        tmp_2d_pose_buffer_list.append(outputs_2d_rt.detach()[valid_rt_idx].cpu().numpy())
        tmp_camparam_buffer_list.append(cam_param.detach()[valid_rt_idx].cpu().numpy())

        # update writer iter num
        summary.summary_train_iter_num_update()

        # plot a image for visualization
        if i % 400 == 0:
            plot_poseaug(inputs_3d, inputs_2d, g_rlt, cam_param, summary.epoch, i, args)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
            .format(batch=i + 1, size=len(data_dict['train_gt2d3d_loader']), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    bar.finish()
    ###################################
    # re-define the buffer dataloader #
    ###################################
    # buffer loader will be used to save fake pose pair
    print('\nprepare buffer loader for train on fake pose')
    train_fake2d3d_loader = DataLoader(PoseDataSet(tmp_3d_pose_buffer_list, tmp_2d_pose_buffer_list,
                                                   [['none'] * len(np.concatenate(tmp_camparam_buffer_list))],
                                                   tmp_camparam_buffer_list),
                                       batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, pin_memory=True)

    data_dict['train_fake2d3d_loader'] = train_fake2d3d_loader

    return
