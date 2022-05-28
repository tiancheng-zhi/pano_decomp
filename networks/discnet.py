import torch
import torch.nn as nn
import torch.nn.functional as F

from .sngan.snlayers.snconv2d import SNConv2d

class DiscNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, nf=64, pad='pano'):
        super().__init__()
        self.inc = iconv(in_c, nf, norm='bn', act='leaky', pad=pad)
        self.down1 = down(nf, nf * 2, norm='bn', act='leaky', pad=pad)
        self.down2 = down(nf * 2, nf * 4, norm='bn', act='leaky', pad=pad)
        self.down3 = down(nf * 4, nf * 4, norm='bn', act='leaky', pad=pad)
        self.outc = oconv(nf * 4, out_c)

    def forward(self, im):
        inc = self.inc(im)
        down1 = self.down1(inc)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        outc = self.outc(down3)
        return outc


class conv1(nn.Module):
    def __init__(self, in_ch, out_ch, norm='in', act='relu', pad='pano'):
        super().__init__()
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_ch)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv = SNConv2d(in_ch, out_ch, 3, bias=False)
        self.pad = pad

    def forward(self, x):
        if self.pad == 'pano':
            x = F.pad(x, (1, 1, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, 1, 1), mode='replicate')
        else:
            x = F.pad(x, (1, 1, 1, 1), mode=self.pad)
        x = self.act(self.norm(self.conv(x)))
        return x


class conv2(nn.Module):
    def __init__(self, in_ch, out_ch, norm='in', act='relu', pad='pano'):
        super().__init__()
        self.conv = nn.Sequential(conv1(in_ch, out_ch, norm, act, pad), conv1(out_ch, out_ch, norm, act, pad))

    def forward(self, x):
        return self.conv(x)


class iconv(nn.Module):
    def __init__(self, in_ch, out_ch, conv='conv2', norm='in', act='relu', pad='pano'):
        super().__init__()
        if conv == 'conv2':
            self.conv = conv2(in_ch, out_ch, norm, act, pad)
        elif conv == 'conv1':
            self.conv = conv1(in_ch, out_ch, norm, act, pad)
        else:
            assert(False)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, conv='conv2', norm='in', act='relu', pad='pano'):
        super().__init__()
        if conv == 'conv2':
            self.conv = conv2(in_ch, out_ch, norm, act, pad)
        elif conv == 'conv1':
            self.conv = conv1(in_ch, out_ch, norm, act, pad)
        else:
            assert(False)

        self.conv = nn.Sequential(nn.MaxPool2d(2), self.conv)

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv='conv2', norm='in', act='relu', pad='pano'):
        super().__init__()
        if conv == 'conv2':
            self.conv = conv2(in_ch, out_ch, norm, act, pad)
        elif conv == 'conv1':
            self.conv = conv1(in_ch, out_ch, norm, act, pad)
        self.pad = pad

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        dy = x2.size()[2] - x1.size()[2]
        dx = x2.size()[3] - x1.size()[3]
        if self.pad == 'pano':
            x1 = F.pad(x1, (dx//2, dx - dx//2, 0, 0), mode='circular')
            x1 = F.pad(x1, (0, 0, dy//2, dy - dy//2), mode='replicate')
        else:
            x1 = F.pad(x1, (dx//2, dx - dx//2, dy//2, dy - dy//2), mode=self.pad)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class oconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


