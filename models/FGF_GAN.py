# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")
from utils import box_blur, CBAM, FastGuidedFilter


def upsample(x, h, w):
    return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)
    
class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, padding=1)
        self.relu  = nn.ReLU(True)
        
    def forward(self, x):
        x = x+self.conv2(self.relu(self.conv1(x)))
        return x
    
class GPNN(nn.Module):
    def __init__(self, 
                 ms_channels,
                 pan_channels,
                 n_feat,
                 n_layer):
        super(GPNN, self).__init__()
        self.n_layer = n_layer
        
        relu = nn.ReLU()
        # feat_extractor ms
        feat_extractor_ms = [nn.Conv2d(ms_channels, n_feat, 3, padding=1), relu]
        for i in range(n_layer-1):
            feat_extractor_ms.append(ResBlock(n_feat, n_feat))
        self.feat_extractor_ms = nn.ModuleList(feat_extractor_ms)
        
        # feat_extractor pan
        feat_extractor_pan = [nn.Conv2d(pan_channels, n_feat, 3, padding=1), relu]
        for i in range(n_layer-1):
            feat_extractor_pan.append(ResBlock(n_feat, n_feat))
        self.feat_extractor_pan = nn.ModuleList(feat_extractor_pan)
        
        # Attention
        self.attention = nn.ModuleList([CBAM(n_feat, 2) for i in range(n_layer)])
        # guided fusion unit
        self.guided_filter = FastGuidedFilter(1, 1e-4)
        
        # reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(n_feat*n_layer, n_feat, 1),
            relu,
            nn.Conv2d(n_feat, ms_channels, 3, padding=1)
            )
    
    def get_high_freq(self, x):
        return x-box_blur(x, kernel_size=[5,5])
        
    def forward(self, ms, pan=None):
        # ms  - low-resolution multi-spectral image [N,C,h,w] 
        # pan - high-resolution panchromatic image [N,1,H,W] 
        if type(pan)==torch.Tensor:
            pass
        elif pan==None:
            raise Exception('User does not provide pan image!')
        # 0. up-sample ms
        N,C,h,w = ms.shape
        _,_,H,W = pan.shape
        ms0 = upsample(ms, H, W)
        guided_ms = []
        
        # 1. backbone
        # ms = self.get_high_freq(ms)
        # pan = self.get_high_freq(pan)
        for i in range(self.n_layer):
            ms = self.feat_extractor_ms[i](ms)
            pan = self.feat_extractor_pan[i](pan)
            lr_pan = upsample(pan, h, w)
            guided_ms.append(self.attention[i](self.guided_filter(lr_pan,ms,pan)))
        # 2. reconstruction
        guided_ms = torch.cat(guided_ms, dim=1)
        ms = self.recon(guided_ms)
        ms = ms0+ms
        return ms

class Discriminator(nn.Module):
    def __init__(self, 
                 in_channel,
                 base_channel=32):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(base_channel,   2*base_channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(2*base_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2*base_channel, 4*base_channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(4*base_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4*base_channel, 8*base_channel, 3, padding=1),
            nn.BatchNorm2d(8*base_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8*base_channel, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
        self.model.apply(weights_init_normal)
    
    def forward(self, output_img, input_img, input_pan):
        _,_,H,W = input_pan.shape
        input_img = upsample(input_img, H,W)
        img = torch.cat((output_img, input_img, input_pan), dim=1)
        validity = self.model(img)
        return validity

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
# test
from torchsummary import summary
summary(GPNN(10,1,32,5).cuda(), [(10,32,32),(1,64,64)]) 
