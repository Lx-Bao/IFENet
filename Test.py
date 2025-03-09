import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
import numpy as np
from net import Mnet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    model_path = './model/epoch_200.pth'

    out_path = './output/'


    data = Data(root='./VDT-2048 dataset/Test/', mode='test')
    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = Mnet().cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)

    img_num = len(loader)
    net.eval()
    with torch.no_grad():
        for rgb, t, d,eg, mask, (H, W), name in loader:
            x1e_pred, x1e_pred_t, x1e_pred_d, x_pred, x_pred_l, x_pred_h= net(rgb.cuda().float(), t.cuda().float(), d.cuda().float())
            score1 = F.interpolate(x_pred[0], size=(H, W), mode='bilinear', align_corners=True)
            pred = np.squeeze(torch.sigmoid(score1).cpu().data.numpy())
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), 255 * pred)





