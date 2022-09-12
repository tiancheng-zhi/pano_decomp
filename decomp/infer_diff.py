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
    parser.add_argument('--log', type=str, default='result_diff')
    parser.add_argument('--ckpt', type=str, default='../models/diff.pth')
    parser.add_argument('--rgb-modal', type=str, default='lowres')
    parser.add_argument('--semantic-modal', type=str, default='semantic')
    parser.add_argument('--no-diff-minimum', action='store_true')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--mask-dilate', type=int, default=5)
    parser.add_argument('--mask-thres', type=float, default=0.05)
    parser.add_argument('--alpha-dilate', type=int, default=21)
    parser.add_argument('--alpha-thres', type=float, default=0.03)
    parser.add_argument('--device', type=str, default='cuda:0')
    opt = parser.parse_args()

    opt.test_dsets = opt.test_dsets.split(',')
    opt.test_lists = opt.test_lists.split(',')
    opt.log = Path(opt.log)
    if opt.ckpt is not None:
        opt.ckpt = Path(opt.ckpt)
        checkpoint = torch.load(opt.ckpt, map_location=opt.device)
    else:
        checkpoint = None
    remkdir(opt.log)
    return opt, checkpoint


def test(opt, data_loader, diffnet):
    np.random.seed(0)
    diffnet = diffnet.eval()
    torch.autograd.set_grad_enabled(False)
    data_len = len(data_loader)
    start_time = time.time()
    for iteration, batch in enumerate(data_loader):
        index, dset_id, record_id, rgb, sem, spec = batch
        rgb, sem, spec = to_device([rgb, sem, spec], opt.device)
        start_time = time.time()

        sem_floor = (sem == SEM_FLOOR)[:,None].float()
        diff = diffnet(rgb)

        mask = (spec.mean(1, keepdim=True) > opt.mask_thres).float()
        mask = morpho_dilate_pano(mask, opt.mask_dilate) * sem_floor
        if (1 - mask).sum() >= 10:
            diff, _ = color_match(diff, rgb, (1 - mask))
        if not opt.no_diff_minimum: # optional
            diff = torch.min(diff, rgb - spec)
        alpha = ((diff - rgb).abs().mean(1, keepdim=True) > opt.alpha_thres).float()
        alpha = morpho_dilate_pano(alpha, opt.alpha_dilate) * sem_floor
        alpha = kornia.filters.gaussian_blur2d(alpha, (5, 5), (3, 3), 'circular')
        blend = diff * alpha + rgb * (1 - alpha)

        pred_data = [(diff, 'im', 'pred'), (blend, 'im', 'diff'), (alpha, 'im', 'alpha'), (rgb - spec, 'im', 'res')]
        save_vis(opt.log, data=pred_data, pref=record_id[0].split(' ')[1], overwrite=True)
        print(f'diff test ({iteration}/{data_len}) {round(time.time()-start_time,2)}s')
        start_time = time.time()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt, checkpoint = parse_args()
    print_log(opt, opt.log / 'opt.txt')

    diffnet = RegNet(nf=checkpoint['opt'].diffnet_nf).to(opt.device)

    test_set = PanoDataset(dsets=opt.test_dsets, record_lists=opt.test_lists, height=opt.height, modals=[opt.rgb_modal, opt.semantic_modal, 'specular'], rotate=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1, shuffle=False)

    if checkpoint is not None:
        diffnet.load_state_dict(checkpoint['diffnet'])

    test(opt, test_loader, diffnet)

