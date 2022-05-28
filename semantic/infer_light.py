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
from networks.regnet import RegNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dsets', type=str, default='zind')
    parser.add_argument('--test-lists', type=str, default='../lists/zind_panos_all.txt')
    parser.add_argument('--log', type=str, default='result')
    parser.add_argument('--ckpt', type=str, default='../models/light.pth')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    opt = parser.parse_args()

    opt.test_dsets = opt.test_dsets.split(',')
    opt.test_lists = opt.test_lists.split(',')
    opt.log = Path(opt.log)
    opt.ckpt = Path(opt.ckpt)
    checkpoint = torch.load(opt.ckpt)
    remkdir(opt.log)
    return opt, checkpoint


def run(stage, opt, epoch, data_loader, itmnet, itmnet_optim=None):
    np.random.seed(0)
    itmnet = itmnet.eval()
    torch.autograd.set_grad_enabled(False)
    for iteration, batch in enumerate(data_loader):
        index, dset_id, record_id, rgb = batch
        rgb = rgb.to(opt.device)
        ldr = rgb ** 2.2
        pred_logres = itmnet(ldr)
        pred_hdr = torch.expm1(pred_logres) + ldr
        pred_mask = F.relu(torch.tanh(pred_hdr.mean(1, keepdim=True) - 2))
        pred_mask1 = F.relu(torch.tanh(pred_hdr.mean(1, keepdim=True) - 1))
        pred_hdr = pred_mask1 * pred_hdr + (1 - pred_mask1) * ldr
        pred_data = [(pred_mask, 'im', 'light'), (pred_hdr, 'hdr', 'hdr')]
        save_vis(opt.log, data=pred_data, pref=record_id[0].split(' ')[1], overwrite=True)
        print(iteration, '/', len(data_loader))


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt, checkpoint = parse_args()
    print_log(opt, opt.log / 'opt.txt')
    itmnet = RegNet().to(opt.device)

    test_set = PanoDataset(dsets=opt.test_dsets, record_lists=opt.test_lists, height=opt.height, modals=['rgb'], rotate=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1, shuffle=False)

    ckpt_net = checkpoint['itmnet']
    itmnet.load_state_dict(ckpt_net)

    run('test', opt, 0, test_loader, itmnet)
