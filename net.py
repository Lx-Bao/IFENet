import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from lib.swin_transformer import SwinTransformer
from models.OctaveConv2 import FirstOctaveCBR, LastOCtaveCBR, OctaveCBR
from models.DCTlayer import MultiSpectralAttentionLayer
from models.exchange_modules import *
import math
from models.GCN import *

class Gate(nn.Module):
    def __init__(self, in_plane):
        super(Gate, self).__init__()
        self.gate1 = nn.Conv3d(in_plane,in_plane, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.gate2 = nn.Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, fea):
        gate = torch.sigmoid(self.gate1(fea))
        gate_fea = fea * gate + fea
        gate_fea = gate_fea.permute(0, 2, 1, 3, 4)
        gate_fea = torch.squeeze(self.gate2(gate_fea), dim=1)

        return gate_fea


class Mnet(nn.Module):
    def __init__(self,pretrained=False):
        super(Mnet, self).__init__()
        self.swin1 = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.gate4 = Gate(512)
        self.gate3 = Gate(256)
        self.gate2 = Gate(128)
        self.gate1 = Gate(64)

        self.CBAM1 = CBAMLayer(64)
        self.CBAM2 = CBAMLayer(128)
        self.CBAM3 = CBAMLayer(256)
        self.CBAM4 = CBAMLayer(512)

        self.de_conv4 = Conv(512, 256, 3, bn=False, relu=False)
        self.de_conv3 = Conv(256, 128, 3, bn=False, relu=False)
        self.de_conv2 = Conv(128, 64, 3, bn=False, relu=False)
        self.de_conv1 = Conv(64, 32, 3, bn=False, relu=False)

        self.interaction4 =Interaction(in_channels=1024, out_channels=512)
        self.interaction3 = Interaction(in_channels=512, out_channels=256)
        self.interaction2 = Interaction(in_channels=256, out_channels=128)
        self.interaction1 =Interaction(in_channels=128, out_channels=64)

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_d = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_t = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.conv4_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=512, out_channels=512, kernel_size=(1,1), padding=0, bias=True),
        )
        self.conv3_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=256, out_channels=256, kernel_size=(1,1), padding=0, bias=True),
        )
        self.conv2_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=128, out_channels=128, kernel_size=(1,1), padding=0, bias=True),
        )
        self.conv1_vdt = nn.Sequential(
            FirstOctaveCBR(in_channels=64, out_channels=64, kernel_size=(1,1), padding=0, bias=True),
        )


        self.conv4_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=512, out_channels=256, kernel_size=(1,1), padding=0, bias=True),

        )
        self.conv3_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=256, out_channels=128, kernel_size=(1,1), padding=0, bias=True),

        )
        self.conv2_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=128, out_channels=64, kernel_size=(1,1), padding=0, bias=True),

        )
        self.conv1_vdt2 = nn.Sequential(
            OctaveCBR(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=True),
            LastOCtaveCBR(in_channels=64, out_channels=32, kernel_size=(1,1), padding=0, bias=True),
        )

        self.LF_conv4 = Conv(256, 256, 3, bn=True, relu=True)
        self.LF_conv3 = Conv(128, 128, 3, bn=True, relu=True)
        self.LF_conv2 = Conv(64, 64, 3, bn=True, relu=True)
        self.LF_conv1 = Conv(32, 32, 3, bn=True, relu=True)
        self.DCTatt_L4 = MultiSpectralAttentionLayer(256, 6, 6,  reduction=8, freq_sel_method = 'low4')
        self.DCTatt_L3 = MultiSpectralAttentionLayer(128, 12, 12,  reduction=8, freq_sel_method = 'low4')
        self.DCTatt_L2 = MultiSpectralAttentionLayer(64, 24, 24,  reduction=8, freq_sel_method = 'low4')
        self.DCTatt_L1 = MultiSpectralAttentionLayer(32, 48, 48,  reduction=8, freq_sel_method = 'low4')

        self.HF_conv4 = Conv(256, 256, 3, bn=True, relu=True)
        self.HF_conv3 = Conv(128, 128, 3, bn=True, relu=True)
        self.HF_conv2 = Conv(64, 64, 3, bn=True, relu=True)
        self.HF_conv1 = Conv(32, 32, 3, bn=True, relu=True)
        self.DCTatt_H4 = MultiSpectralAttentionLayer(256, 12, 12,  reduction=8, freq_sel_method = 'hig4')
        self.DCTatt_H3 = MultiSpectralAttentionLayer(128, 24, 24,  reduction=8, freq_sel_method = 'hig4')
        self.DCTatt_H2 = MultiSpectralAttentionLayer(64, 48, 48,  reduction=8, freq_sel_method = 'hig4')
        self.DCTatt_H1 = MultiSpectralAttentionLayer(32, 96, 96,  reduction=8, freq_sel_method = 'hig4')

        self.final_4_vdt_h = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_4_vdt_l = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_4_vdt = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_3_vdt_h = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_3_vdt_l = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_3_vdt = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_2_vdt_h = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_2_vdt_l = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_2_vdt = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )
        self.final_1_vdt_h = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )
        self.final_1_vdt_l = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )
        self.final_1_vdt = nn.Sequential(
            Conv(32, 32, 3, bn=True, relu=True),
            Conv(32, 1, 1, bn=False, relu=False)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, rgb, t, d):
        score_list_t, score_PE = self.swin1(t)
        score_list_rgb, score_PE = self.swin1(rgb)
        score_list_d, score_PE = self.swin1(d)

        x1 = [score_list_rgb[0],score_list_t[0],score_list_d[0]]
        x2 = [score_list_rgb[1],score_list_t[1],score_list_d[1]]
        x3 = [score_list_rgb[2],score_list_t[2],score_list_d[2]]
        x4 = [score_list_rgb[3],score_list_t[3],score_list_d[3]]

##########################Inter-modal and Intra-modal Graph-based Interaction Module########################################
        x4e = self.interaction4(x4)
        x4e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x4_) for x4_ in x4e]


        x3e = [x3[i]+x4e[i] for i in range(3)]
        x3e = self.interaction3(x3e)
        x3e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x3e]


        x2e = [x2[i] + x3e[i] for i in range(3)]
        x2e = self.interaction2(x2e)
        x2e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x2e]


        x1e = [x1[i] + x2e[i] for i in range(3)]
        x1e = self.interaction1(x1e)
        x1e = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xe_) for xe_ in x1e]

        x4e_v = x4e[0]
        x3e_v = x3e[0]
        x2e_v = x2e[0]
        x1e_v = x1e[0]
        x1e_pred = self.final_1(x1e_v)
        x1e_pred = self.up2(x1e_pred)

        x4e_t = x4e[1]
        x3e_t = x3e[1]
        x2e_t = x2e[1]
        x1e_t = x1e[1]
        x1e_pred_t = self.final_1_t(x1e_t)
        x1e_pred_t = self.up2(x1e_pred_t)

        x4e_d = x4e[2]
        x3e_d = x3e[2]
        x2e_d = x2e[2]
        x1e_d = x1e[2]
        x1e_pred_d = self.final_1_d(x1e_d)
        x1e_pred_d = self.up2(x1e_pred_d)

#################################Gated Attention-based Fusion Module#####################################
        x1e_vdt = torch.stack([x1e_t, x1e_v, x1e_d], dim=2)
        x1_vdt = self.gate1(x1e_vdt)

        x2e_vdt = torch.stack([x2e_t, x2e_v, x2e_d], dim=2)
        x2_vdt = self.gate2(x2e_vdt)

        x3e_vdt = torch.stack([x3e_t, x3e_v, x3e_d], dim=2)
        x3_vdt = self.gate3(x3e_vdt)

        x4e_vdt = torch.stack([x4e_t, x4e_v, x4e_d], dim=2)
        x4_vdt = self.gate4(x4e_vdt)

################################Frequency Split-based Enhancement Module#####################################
        x4e_vdt_h,x4e_vdt_l = self.conv4_vdt(x4_vdt)
        x4e_vdt_h = self.DCTatt_H4(x4e_vdt_h)
        x4e_vdt_l = self.DCTatt_L4(x4e_vdt_l)
        x4e_vdt_h = self.HF_conv4(x4e_vdt_h)
        x4e_vdt_l = self.LF_conv4(x4e_vdt_l)
        x4e_pred_vdt_h = self.final_4_vdt_h(x4e_vdt_h)
        x4e_pred_vdt_l = self.final_4_vdt_l(x4e_vdt_l)
        x4e_vdt = self.conv4_vdt2((x4e_vdt_h,x4e_vdt_l))
        x4s_vdt = self.de_conv4(self.CBAM4(x4_vdt))
        x4e_vdt = x4e_vdt + x4s_vdt
        x4e_vdt = self.up2(x4e_vdt)
        x4e_pred_vdt = self.final_4_vdt(x4e_vdt)

        x3i_vdt = x4e_vdt + x3_vdt
        x3e_vdt_h,x3e_vdt_l = self.conv3_vdt(x3i_vdt)
        x3e_vdt_h = self.DCTatt_H3(x3e_vdt_h)
        x3e_vdt_l = self.DCTatt_L3(x3e_vdt_l)
        x3e_vdt_h = self.HF_conv3(x3e_vdt_h)
        x3e_vdt_l = self.LF_conv3(x3e_vdt_l)
        x3e_pred_vdt_h = self.final_3_vdt_h(x3e_vdt_h)
        x3e_pred_vdt_l = self.final_3_vdt_l(x3e_vdt_l)
        x3e_vdt = self.conv3_vdt2((x3e_vdt_h, x3e_vdt_l))
        x3s_vdt = self.de_conv3(self.CBAM3(x3i_vdt))
        x3e_vdt = x3e_vdt + x3s_vdt
        x3e_vdt = self.up2(x3e_vdt)
        x3e_pred_vdt = self.final_3_vdt(x3e_vdt)

        x2i_vdt = x3e_vdt + x2_vdt
        x2e_vdt_h,x2e_vdt_l = self.conv2_vdt(x2i_vdt)
        x2e_vdt_h = self.DCTatt_H2(x2e_vdt_h)
        x2e_vdt_l = self.DCTatt_L2(x2e_vdt_l)
        x2e_vdt_h = self.HF_conv2(x2e_vdt_h)
        x2e_vdt_l = self.LF_conv2(x2e_vdt_l)
        x2e_pred_vdt_h = self.final_2_vdt_h(x2e_vdt_h)
        x2e_pred_vdt_l = self.final_2_vdt_l(x2e_vdt_l)
        x2e_vdt = self.conv2_vdt2((x2e_vdt_h, x2e_vdt_l))
        x2s_vdt = self.de_conv2(self.CBAM2(x2i_vdt))
        x2e_vdt = x2e_vdt + x2s_vdt
        x2e_vdt = self.up2(x2e_vdt)
        x2e_pred_vdt = self.final_2_vdt(x2e_vdt)

        x1i_vdt = x2e_vdt + x1_vdt
        x1e_vdt_h,x1e_vdt_l = self.conv1_vdt(x1i_vdt)
        x1e_vdt_h = self.DCTatt_H1(x1e_vdt_h)
        x1e_vdt_l = self.DCTatt_L1(x1e_vdt_l)
        x1e_vdt_h = self.HF_conv1(x1e_vdt_h)
        x1e_vdt_l = self.LF_conv1(x1e_vdt_l)
        x1e_pred_vdt_h = self.final_1_vdt_h(x1e_vdt_h)
        x1e_pred_vdt_l = self.final_1_vdt_l(x1e_vdt_l)
        x1e_vdt = self.conv1_vdt2((x1e_vdt_h, x1e_vdt_l))
        x1s_vdt = self.de_conv1(self.CBAM1(x1i_vdt))
        x1e_vdt = x1e_vdt + x1s_vdt
        x1e_vdt = self.up2(x1e_vdt)
        x1e_pred_vdt = self.final_1_vdt(x1e_vdt)

        x2e_pred_vdt = self.up2(x2e_pred_vdt)
        x3e_pred_vdt = self.up4(x3e_pred_vdt)
        x4e_pred_vdt = self.up8(x4e_pred_vdt)
        x_pred = [x1e_pred_vdt, x2e_pred_vdt, x3e_pred_vdt, x4e_pred_vdt]

        x1e_pred_vdt_l = self.up4(x1e_pred_vdt_l)
        x2e_pred_vdt_l = self.up8(x2e_pred_vdt_l)
        x3e_pred_vdt_l = self.up16(x3e_pred_vdt_l)
        x4e_pred_vdt_l = self.up32(x4e_pred_vdt_l)
        x_pred_l = [x1e_pred_vdt_l, x2e_pred_vdt_l, x3e_pred_vdt_l, x4e_pred_vdt_l]

        x1e_pred_vdt_h = self.up2(x1e_pred_vdt_h)
        x2e_pred_vdt_h = self.up4(x2e_pred_vdt_h)
        x3e_pred_vdt_h = self.up8(x3e_pred_vdt_h)
        x4e_pred_vdt_h = self.up16(x4e_pred_vdt_h)
        x_pred_h = [x1e_pred_vdt_h, x2e_pred_vdt_h, x3e_pred_vdt_h, x4e_pred_vdt_h]

        return x1e_pred, x1e_pred_t, x1e_pred_d, x_pred, x_pred_l, x_pred_h




    def load_pretrained_model(self):
        self.swin1.load_state_dict(torch.load('/home/baoliuxin/VDT/swin_base_patch4_window12_384_22k.pth')['model'],strict=False)
        print('loading pretrained model success!')



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class Interaction(nn.Module):
    def __init__(self, in_channels, out_channels, num_parallel=3, bn_threshold=2e-2):
        super(Interaction, self).__init__()
        self.inp_dim = in_channels
        self.num_parallel = num_parallel
        self.bn_threshold = bn_threshold


        self.conv1 = ModuleParallel(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv2 = ModuleParallel(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv3 = ModuleParallel(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True))

        self.relu = ModuleParallel(nn.ReLU(inplace=False))

        self.exchange = Exchange(in_channels)

        self.bn1 = BatchNorm2dParallel(in_channels)
        self.bn2 = BatchNorm2dParallel(in_channels)
        self.bn3 = BatchNorm2dParallel(out_channels)

        self.intra_GCA1 = intra_graph(out_channels)
        self.intra_GCA2 = intra_graph(out_channels)
        self.intra_GCA3 = intra_graph(out_channels)
        self.inter_GCA = inter_gcn(out_channels)

        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)

        Fe_v,Fe_d,Fe_t = self.inter_GCA(out[0], out[1], out[2])


        intra_out1 = self.intra_GCA1(Fe_v)
        intra_out2 = self.intra_GCA2(Fe_d)
        intra_out3 = self.intra_GCA3(Fe_t)

        out = [intra_out1, intra_out2, intra_out3]


        return out


class inter_gcn(nn.Module):
    def __init__(self, in_channel, GNN=False):
        super(inter_gcn, self).__init__()
        self.in_channel = in_channel
        self.gcn = Channel_gcn(in_channel, 48)

    def forward(self, F_v,F_d,F_t):
        Fe_v,Fe_d,Fe_t = self.gcn(F_v,F_d,F_t)
        Fe_v = F_v + Fe_v
        Fe_d = F_d + Fe_d
        Fe_t = F_t + Fe_t

        return Fe_v,Fe_d,Fe_t


class Channel_gcn(nn.Module):
    def __init__(self, dim_a, num):  # 64 64 64 64
        super(Channel_gcn, self).__init__()
        dim = dim_a * 3  # 64*6

        self.gcn = CascadeGCNet(dim//num, loop=2)  # 邻接矩阵size 输入节点维度 输出节点维度 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.fc_1 = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()
        )

        self.num = num
        self.dim_a = dim_a

    def forward(self, F_v,F_d,F_t):
        batch, channel, _, _ = F_v.size()
        combined = torch.cat([F_v, F_d, F_t], dim=1)
        combined_fc = self.avg_pool(combined).view(batch, self.num, -1).permute(0, 2, 1)

        feat_cat = self.gcn(combined_fc)
        feat_cat = feat_cat.view(batch, -1)

        excitation1 = self.fc_1(feat_cat[:,:self.dim_a]).view(batch, channel, 1, 1)
        excitation2 = self.fc_2(feat_cat[:,self.dim_a:self.dim_a*2]).view(batch, channel, 1, 1)
        excitation3 = self.fc_3(feat_cat[:,self.dim_a*2:]).view(batch, channel, 1, 1)

        return excitation1 * F_v, excitation2 * F_d, excitation3 * F_t



class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x