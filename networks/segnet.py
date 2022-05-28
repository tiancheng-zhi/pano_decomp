import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import torch
import torch.nn as nn

from mit_semseg.models import ModelBuilder
from utils.misc import convert_prob


class SegNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.im_mean = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None]
        self.im_std = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None]
       
    def forward(self, im):
        im = (im - self.im_mean.to(im.device)) / self.im_std.to(im.device)
        im = self.encoder(im)
        im = self.decoder(im)
        prob = torch.softmax(im, 1)
        return prob
