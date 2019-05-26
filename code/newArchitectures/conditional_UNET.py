# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Emb') != -1:
        init.normal(m.weight, mean=0, std=0.01)

class _netZ(nn.Module):
    def __init__(self, nz, n):
        super(_netZ, self).__init__()
        self.n = n
        self.emb = nn.Embedding(self.n, nz)
        self.nz = nz

    def get_norm(self):
        wn = self.emb.weight.norm(2, 1).data.unsqueeze(1)
        self.emb.weight.data = \
            self.emb.weight.data.div(wn.expand_as(self.emb.weight.data))

    def forward(self, idx):
        z = self.emb(idx).squeeze()
        return z

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dim):
        super(UNet, self).__init__()
        self.dim = dim

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def main(self, x):
        if (self.dim == 64):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        elif (self.dim == 32):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            # x5 = self.down4(x4)

            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            # x = self.up4(x, x1)

        else:
            print('The images dimension is wrong')
            exit()

        x = self.outc(x)
        x = F.sigmoid(x)
        return x

class _netT64(nn.Module):
    def __init__(self, ngf, in_nc, out_nc):
        super(_netT64, self).__init__()
        print('net T64')
        n_channels = 3
        n_classes = 3
        self.unet = UNet(n_channels, n_classes, 64)

    def forward(self, Is):
        return self.unet.main(Is)

class _netT32(nn.Module):
    def __init__(self, ngf, in_nc, out_nc):
        super(_netT32, self).__init__()
        print('net T32')
        n_channels = 3
        n_classes = 3
        self.unet = UNet(n_channels, n_classes, 32)

    def forward(self, Is):
        return self.unet.main(Is)
