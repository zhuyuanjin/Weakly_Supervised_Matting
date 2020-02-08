import torch 
from torchvision.models import vgg11_bn
from torch import nn
from collections import OrderedDict






class DeconvSingle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvSingle, self).__init__()
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv = nn.Sequential(
                                    nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),

                                   )
    def forward(self, x ):
        x = self.unpool(x)
        x = self.deconv(x)
        return x
        
       

class DeconvDouble(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(DeconvDouble, self).__init__()
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)

        self.deconv = nn.Sequential(
                                    nn.Conv2d(in_channels, in_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                   )
    def forward(self, x ):
        x = self.unpool(x)
        x = self.deconv(x)
        return x



class DeconvTriple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvTriple, self).__init__()
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.deconv = nn.Sequential(
                                    nn.Conv2d(in_channels, in_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, in_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
                                    nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                   )
    def forward(self, x ):
        x = self.unpool(x)
        x = self.deconv(x)
        return x










class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg11 = vgg11_bn(pretrained=True)
        self.features = vgg11.features
    def forward(self, x):
        x = self.features(x)
        return x




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Conv =  nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.Deconv5 = DeconvDouble(512, 512)
        self.Deconv4 = DeconvDouble(512, 256)
        self.Deconv3 = DeconvDouble(256, 128)
        self.Deconv2 = DeconvSingle(128, 64)
        self.Deconv1 = DeconvSingle(64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

    def forward(self, x):
        x = self.Conv(x)
        x = self.Deconv5(x)
        x = self.Deconv4(x)
        x = self.Deconv3(x)
        x = self.Deconv2(x)
        x = self.Deconv1(x)
        x = self.final_conv(x)
        return x
