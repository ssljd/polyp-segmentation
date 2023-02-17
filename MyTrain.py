import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.unet_model import UNet
from lib.PraNet_Res2Net import *
from lib.nest_unet import *
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from utils.metrics import *
import torch.nn.functional as F

def train(train_loader, model, optimizer, epoch):

    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        dice = 0.0
        iou = 0.0
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            # lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            lateral_map_2 = model(images)
            # ---- loss function ----
            # loss5 = structure_loss(lateral_map_5, gts)
            # loss4 = structure_loss(lateral_map_4, gts)
            # loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2
            # loss = loss2 + loss3 + loss4 + loss5    # TODO: try different weights for loss

            # ---- dice function ----
            dice = dice_coef(lateral_map_2, gts)

            # ---- iou function ----
            iou = iou_score(lateral_map_2, gts)

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                # loss_record3.update(loss3.data, opt.batchsize)
                # loss_record4.update(loss4.data, opt.batchsize)
                # loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
            #       '[dice: {:.4f}, dice: {:.4f}, lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
            #       format(datetime.now(), epoch, opt.epoch, i, total_step, dice, iou,
            #              loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))

            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[dice: {:.4f}, iou: {:.4f}, lateral-2: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, dice, iou,
                         loss_record2.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        # torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        # torch.save(model.state_dict(), save_path + 'PraNet-best.pth')
        torch.save(model.state_dict(), save_path + 'Res2UNet-%d.pth' % epoch)
        torch.save(model.state_dict(), save_path + 'Res2UNet-best.pth')
        # print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth' % epoch)
        # print('[Saving Snapshot:]', save_path + 'PraNet-best.pth')
        print('[Saving Snapshot:]', save_path + 'Res2UNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'Res2UNet-best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='/home/labrobo/extend_disk/sljd/Polyp-Dataset/TrainDataset', help='path to train dataset')
    # parser.add_argument('--train_save', type=str,
    #                     default='snapshots/PraNet_Res2Net')
    parser.add_argument('--train_save', type=str,
                        default='Res2UNet')
    opt = parser.parse_args()

    # ---- build models ----
    model = Res2UNet().cuda()
    # model.load_state_dict(torch.load('snapshots/snapshots/Unet/Unet-best.pth'))
    # model = PraNet().cuda()
    # model.load_state_dict(torch.load('snapshots/PraNet_Res2Net/PraNet-best.pth'))

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    for epoch in range(0, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
