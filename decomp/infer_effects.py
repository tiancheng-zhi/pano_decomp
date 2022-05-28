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
from torch.utils.data import DataLoader

from datasets.pano_dataset import PanoDataset
from utils.misc import *
from utils.transform import *
from utils.ops import *
from networks.regnet import RegNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dsets', type=str, default='zind')
    parser.add_argument('--test-lists', type=str, default='../lists/zind_panos_test.txt')
    parser.add_argument('--log', type=str, default='result_eff')
    parser.add_argument('--ckpt', type=str, default='../models/eff.pth')
    parser.add_argument('--rgb-modal', default='lowres')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    opt = parser.parse_args()

    opt.test_dsets = opt.test_dsets.split(',')
    opt.test_lists = opt.test_lists.split(',')
    opt.log = Path(opt.log)
    if opt.ckpt is not None:
        opt.ckpt = Path(opt.ckpt)
        checkpoint = torch.load(opt.ckpt)
    else:
        checkpoint = None
    remkdir(opt.log)
    return opt, checkpoint


def test(opt, data_loader, effnet):
    np.random.seed(0)
    torch.autograd.set_grad_enabled(False)
    data_len = len(data_loader)
    start_time = time.time()
    for iteration, batch in enumerate(data_loader):
        index, dset_id, record_id, rgb = batch
        rgb = rgb.to(opt.device)
        eff = effnet(rgb)
        spec, sunl = eff[:, :3], eff[:, 3:]
        pred_data = [(spec, 'im', 'spec'), (sunl, 'im', 'sunl')]
        save_vis(opt.log, data=pred_data, pref=record_id[0].split(' ')[1], overwrite=True)
        print(f'eff test ({iteration}/{data_len}) {round(time.time()-start_time,2)}s')
        start_time = time.time()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt, checkpoint = parse_args()
    print_log(opt, opt.log / 'opt.txt')

    effnet = RegNet(out_c=6, nf=checkpoint['opt'].effnet_nf).to(opt.device)

    test_set = PanoDataset(dsets=opt.test_dsets, record_lists=opt.test_lists, height=opt.height, modals=[opt.rgb_modal], rotate=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1, shuffle=False)

    if checkpoint is not None:
        effnet.load_state_dict(checkpoint['effnet'])

    test(opt, test_loader, effnet)

