import torch
import xlwt
import torch.nn.functional as F
import numpy
import numpy as np
import os, argparse
import cv2
from lib.PraNet_Res2Net import *
from lib.unet_model import UNet
from lib.nest_unet import *
from utils.dataloader import test_dataset

# torch.cuda.current_device()
# torch.cuda._initialized = True


# dice
def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

# IOU
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

# sensitivity
def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)

def precision(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    fp = numpy.count_nonzero(predict & ~target)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision

def recall(predict, target): #Sensitivity, Recall, true positive rate都一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = numpy.atleast_1d(predict.astype(numpy.bool))
    target = numpy.atleast_1d(target.astype(numpy.bool))

    tp = numpy.count_nonzero(predict & target)
    fn = numpy.count_nonzero(~predict & target)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall

def build_excel(dataset_name, dice, IOU, P, R):
    workbook = xlwt.Workbook(encoding='utf-8')  # 创建一个workbook 设置编码。Workbook（）是构造函数，返回一个工作薄的对象。
    # 初始化
    sheet_test = workbook.add_sheet('sheet_test', cell_overwrite_ok=True)  # 用cell_overwrite_ok=True实现对单元格的重复写
    sheet_test.write(0, 0, 'dataset')  #
    sheet_test.write(0, 1, 'mean dice')  #
    sheet_test.write(0, 2, 'mean IOU')  #
    sheet_test.write(0, 3, 'P')  #
    sheet_test.write(0, 4, 'R')  #

    # 添加数据
    for i in range(len(dataset_name)):
        sheet_test.write(i+1, 0, dataset_name[i])  #
        sheet_test.write(i+1, 1, dice[i])  #
        sheet_test.write(i+1, 2, IOU[i])  #
        sheet_test.write(i+1, 3, P[i])  #
        sheet_test.write(i+1, 4, R[i])  #

    workbook.save('./results/Res2UNet/metrics.xls')

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/Res2UNet/Res2UNet-best.pth')
dataset_name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
dices = []
IOUs = []
R = []
P = []

for _data_name in dataset_name:
    data_path = '/home/labrobo/extend_disk/sljd/Polyp-Dataset/TestDataset/{}'.format(_data_name)
    save_path = './results/Res2UNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Res2UNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    dice = 0
    IOU = 0
    r = 0
    p = 0
    print(_data_name)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # res5, res4, res3, res2 = model(image)
        res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 计算指标
        dice += dice_coef(res, gt)
        IOU += iou_score(res, gt)
        p += precision(res, gt)
        r += recall(res, gt)

        cv2.imwrite(save_path+name, res*255)

    dices.append(dice / test_loader.size)
    IOUs.append(IOU / test_loader.size)
    P.append(p / test_loader.size)
    R.append(r / test_loader.size)

build_excel(dataset_name, dices, IOUs, P, R)
