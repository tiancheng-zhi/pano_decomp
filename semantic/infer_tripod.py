import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from mit_semseg.models import ModelBuilder

from datasets.pano_dataset import PanoDataset
from utils.misc import *
from utils.ops import *
from utils.transform import *
from networks.segnet import SegNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dsets', type=str, default='zind')
    parser.add_argument('--test-lists', type=str, default='../lists/zind_panos_test.txt')
    parser.add_argument('--log', type=str, default='result_tripod')
    parser.add_argument('--ckpt', type=str, default='../models/tripod.pth')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:0')
    opt = parser.parse_args()

    opt.test_dsets = opt.test_dsets.split(',')
    opt.test_lists = opt.test_lists.split(',')
    opt.log = Path(opt.log)
    opt.ckpt = Path(opt.ckpt)
    checkpoint = torch.load(opt.ckpt)
    remkdir(opt.log)
    return opt, checkpoint


def run(stage, opt, epoch, data_loader, segnet, segnet_optim=None):
    np.random.seed(0)
    segnet = segnet.eval()
    torch.autograd.set_grad_enabled(False)
    for iteration, batch in enumerate(data_loader):
        index, dset_id, record_id, rgb = batch
        rgb = rgb.to(opt.device)
        h = rgb.shape[2] // 5
        rgb_h = rgb[:, :, -h:]
        pred_h = segnet(rgb_h)
        pred_h = F.interpolate(pred_h, (rgb_h.shape[2], rgb_h.shape[3]), mode='bilinear')
        pred_h = (F.softmax(pred_h, 1)[:,1:] >= 0.5).float()

        pred = torch.zeros_like(rgb[:,:1])
        pred[:,:,-h:] = pred_h
        pred = F.interpolate(pred, (256, 512), mode='bilinear')
        pred = (pred >= 0.5).float()
        pred = morpho_open_pano(pred)
        pred_data = [(pred, 'im', 'tripod')]
        save_vis(opt.log, data=pred_data, pref=record_id[0].split(' ')[1], overwrite=True)
        print(iteration, '/', len(data_loader))


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt, checkpoint = parse_args()
    print_log(opt, opt.log / 'opt.txt')
    net_encoder = ModelBuilder.build_encoder(arch='hrnetv2', fc_dim=720, weights='')
    net_decoder = ModelBuilder.build_decoder(arch='c1', fc_dim=720, num_class=2, weights='')
    segnet = SegNet(net_encoder, net_decoder).to(opt.device)

    test_set = PanoDataset(dsets=opt.test_dsets, record_lists=opt.test_lists, height=opt.height, modals=['rgb'], rotate=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1, shuffle=False)
    segnet.load_state_dict(checkpoint['segnet'])

    run('test', opt, 0, test_loader, segnet)
