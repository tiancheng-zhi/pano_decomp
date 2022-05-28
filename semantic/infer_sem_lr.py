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
from mit_semseg.models import ModelBuilder

from datasets.pano_dataset import PanoDataset
from utils.misc import *
from utils.transform import *
from networks.segnet import SegNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dsets', type=str, default='zind')
    parser.add_argument('--test-lists', type=str, default='../lists/zind_panos_all.txt')
    parser.add_argument('--log', type=str, default='result_semantic')
    parser.add_argument('--ckpt', type=str, default='../models/semantic.pth')
    parser.add_argument('--height', type=int, default=512)
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


def top2_connected_component(mask):
    """
        mask: 1 x 1 x H x W
    """
    _, _, H, W = mask.shape
    mask = torch.cat((mask, mask), 3)
    maskcv = (th2cv(mask[0,0]) * 255).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(maskcv)
    counts = []
    for i in range(num_labels):
        counts.append((labels_im == i).sum())
    maskcv = ((labels_im == np.argsort(counts)[-1]) | (labels_im == np.argsort(counts)[-2])).astype(np.float32)
    mask = torch.from_numpy(maskcv).to(mask.device)[None,None]
    mask = torch.cat((mask[:,:,:,W:W+W//2], mask[:,:,:,W//2:W]), 3)
    return mask


def test(opt, data_loader, segnet):
    np.random.seed(0)
    segnet = segnet.eval()
    torch.autograd.set_grad_enabled(False)
    data_len = len(data_loader)
    start_time = time.time()
    for iteration, batch in enumerate(data_loader):
        index, dset_id, record_id, rgb, light, tripod = batch
        B = len(index)
        rgb, light, tripod = to_device([rgb, light, tripod], opt.device)
        out = segnet(rgb)
        out = F.interpolate(out, (256, 512), mode='bilinear')
        pred_label = out[:1].argmax(dim=1)
        sem = convert_label(pred_label, 'ade2room')

        light = F.interpolate(light, (256, 512), mode='bilinear')
        tripod = F.interpolate(tripod, (256, 512), mode='bilinear')


        sem[(sem == SEM_PROP)] = SEM_STRUCT

        geo = torch.zeros_like(sem)
        geo[:,-5:,:] = 1
        GEO_TOP = 0
        GEO_BOTTOM = 1
        sem[(geo == GEO_BOTTOM) & (sem != SEM_FLOOR)] = SEM_NONE

        geo = torch.zeros_like(sem)
        geo[:,geo.shape[1]//2:,:] = 1
        GEO_TOP = 0
        GEO_BOTTOM = 1

        sem[(geo == GEO_TOP) & (sem == SEM_FLOOR)] = SEM_WALL

        mask = top2_connected_component(((sem == SEM_FLOOR) | (sem == SEM_NONE))[None].float())
        sem[(mask[0] < 0.5) & (sem == SEM_FLOOR)] = SEM_STRUCT
        sem[(geo == GEO_BOTTOM) & (sem == SEM_CEILING)] = SEM_WALL

        geo = torch.zeros_like(sem)
        geo[:,-5:,:] = 1
        GEO_TOP = 0
        GEO_BOTTOM = 1
        sem[(geo == GEO_BOTTOM) & (sem != SEM_FLOOR)] = SEM_NONE

        light = light[:,0] > 0
        tripod = tripod[:,0] >= 0.5
        light[:,light.shape[1]//2:] = 0
        light = light & ((sem == SEM_STRUCT) | (sem == SEM_WALL) | (sem == SEM_CEILING))
        sem[light] = SEM_LIGHT
        sem[tripod] = SEM_NONE

        pred_color = label2color(sem, 'room')
        pred_data = [(pred_color, 'im', 'semantic')]
        save_vis(opt.log, data=pred_data, pref=record_id[0].split(' ')[1], overwrite=True)
        print(f'seg test ({iteration}/{data_len}) {round(time.time()-start_time,2)}s')
        start_time = time.time()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt, checkpoint = parse_args()
    print_log(opt, opt.log / 'opt.txt')

    net_encoder = ModelBuilder.build_encoder(arch='hrnetv2', fc_dim=720, weights='pretrained/ade20k-hrnetv2-c1/encoder_epoch_30.pth')
    net_decoder = ModelBuilder.build_decoder(arch='c1', fc_dim=720, num_class=150, weights='pretrained/ade20k-hrnetv2-c1/decoder_epoch_30.pth')
    segnet = SegNet(net_encoder, net_decoder).to(opt.device)

    test_set = PanoDataset(dsets=opt.test_dsets, record_lists=opt.test_lists, height=opt.height, modals=['rgb', 'light', 'tripod'], rotate=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1, shuffle=False)

    if checkpoint is not None:
        segnet.load_state_dict(checkpoint['segnet'])

    test(opt, test_loader, segnet)

