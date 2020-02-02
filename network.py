import torch
from torch import nn
import torchvision.models as models
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
                                  nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                 )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

    def forward(self, x):
        x = self.double_conv(x)
        x, indices = self.maxpool(x)
        return x, indices



class QuadruConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(QuadruConv, self).__init__()
        self.quadru_conv = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                        nn.ReLU(inplace=True),
                                        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

    def forward(self, x):
        x = self.quadru_conv(x)
        x, indices = self.maxpool(x)
        return x, indices



class Deconv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Deconv, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.deconv = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)),
                                    nn.BatchNorm2d(out_channels)
                                   )

    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = self.deconv(x)
        return x



class EncoderDecoder19(nn.Module):

    def __init__(self):
        super(EncoderDecoder19, self).__init__()
        self.Conv1 = DoubleConv(4, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = QuadruConv(128, 256)
        self.Conv4 = QuadruConv(256, 512)
        self.Conv5 = QuadruConv(512, 512)
        self.Deconv5 = Deconv(512, 512)
        self.Deconv4 = Deconv(512, 256)
        self.Deconv3 = Deconv(256, 128)
        self.Deconv2 = Deconv(128, 64)
        self.Deconv1 = Deconv(64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, x):
        x, indices1 = self.Conv1(x)
        x, indices2 = self.Conv2(x)
        x, indices3 = self.Conv3(x)
        x, indices4 = self.Conv4(x)
        x, indices5 = self.Conv5(x)
        x = self.Deconv5(x, indices5)
        x = self.Deconv4(x, indices4)
        x = self.Deconv3(x, indices3)
        x = self.Deconv2(x, indices2)
        x = self.Deconv1(x, indices1)
        x = self.final_conv(x)
        return x
    
