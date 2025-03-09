import os
from net import Mnet
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
import pytorch_iou
from torch import nn
import pytorch_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
IOU = pytorch_iou.IOU(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
bce_loss = nn.BCELoss(reduction='mean')

L1 = nn.L1Loss()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def muti_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = IOU(pred, target)

    loss = (bce_out + iou_out).mean()

    return loss



if __name__ == '__main__':

    # dataset
    img_root = './VDT-2048 dataset/Train/'
    save_path = './model'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001
    batch_size = 4
    epoch = 400
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)
    net = Mnet().cuda()
    net.load_pretrained_model()
    params = net.parameters()
    optimizer = torch.optim.Adam(params, 0.0001, betas=(0.5, 0.999))

    for p in net.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    iter_num = len(loader)
    net.train()

    for epochi in tqdm(range(1, epoch + 1)):
        prefetcher = DataPrefetcher(loader)
        rgb, t, d, eg, label = prefetcher.next()
        r_sal_loss = 0
        epoch_ave_loss = 0
        i = 0
        while rgb is not None:
            i += 1
            x1e_pred, x1e_pred_t, x1e_pred_d, x_pred, x_pred_l, x_pred_h = net(rgb, t, d)

            loss1 = structure_loss(x1e_pred, label)

            loss2 = structure_loss(x1e_pred_t, label)

            loss3 = structure_loss(x1e_pred_d, label)

            loss4 = structure_loss(x_pred[0], label)
            loss5 = structure_loss(x_pred[1], label)
            loss6 = structure_loss(x_pred[2], label)
            loss7 = structure_loss(x_pred[3], label)

            loss_x4eg = muti_loss(torch.sigmoid(x_pred_h[0]), eg)
            loss_x3eg = muti_loss(torch.sigmoid(x_pred_h[1]), eg)
            loss_x2eg = muti_loss(torch.sigmoid(x_pred_h[2]), eg)
            loss_x1eg = muti_loss(torch.sigmoid(x_pred_h[3]), eg)

            loss_x4body = muti_loss(torch.sigmoid(x_pred_l[0]), label)
            loss_x3body = muti_loss(torch.sigmoid(x_pred_l[1]), label)
            loss_x2body = muti_loss(torch.sigmoid(x_pred_l[2]), label)
            loss_x1body = muti_loss(torch.sigmoid(x_pred_l[3]), label)

            sal_loss = loss1 + loss2 + loss3 + loss4*2 + loss5 + loss6 + loss7 + loss_x4eg + loss_x3eg + loss_x2eg + loss_x1eg + loss_x4body + loss_x3body + loss_x2body + loss_x1body

            r_sal_loss += sal_loss.data
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 100, lr,))
                epoch_ave_loss += (r_sal_loss / 100)
                r_sal_loss = 0
            rgb, t, d, eg, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        if epochi % 10 == 0:
            model_path = '%s/epoch_%d.pth' % (save_path, epochi)
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))

    torch.save(net.state_dict(), '%s/final.pth' % (save_path))