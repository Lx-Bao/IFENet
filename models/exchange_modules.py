import torch.nn as nn
import torch



class Exchange(nn.Module):
    def __init__(self,in_channels):
        super(Exchange, self).__init__()
        self.in_channel = in_channels

    def forward(self, x, bn, bn_threshold):
        bn1, bn2, bn3 = bn[0].weight.abs(), bn[1].weight.abs(), bn[2].weight.abs()
        x1, x2, x3 = torch.zeros_like(x[0]), torch.zeros_like(x[1]), torch.zeros_like(x[2])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = (x[1][:, bn1 < bn_threshold] + x[2][:, bn1 < bn_threshold])/2
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = (x[0][:, bn2 < bn_threshold] + x[2][:, bn2 < bn_threshold])/2
        x3[:, bn3 >= bn_threshold] = x[2][:, bn3 >= bn_threshold]
        x3[:, bn3 < bn_threshold] = (x[0][:, bn3 < bn_threshold] + x[1][:, bn3 < bn_threshold])/2

        return [x1, x2, x3]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel=3):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
