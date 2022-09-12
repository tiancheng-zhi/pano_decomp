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
from torch.utils.data import DataLoader
import kornia

from datasets.pano_dataset import PanoDataset
from utils.misc import *
from utils.transform import *
from utils.ops import *
from networks.regnet import RegNet
from networks.discnet import DiscNet
from networks.vgg import VGG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dsets', type=str, default='zind')
    parser.add_argument('--train-lists', type=str, default='../lists/zind_panos_train.txt')
    parser.add_argument('--val-dsets', type=str, default='zind')
    parser.add_argument('--val-lists', type=str, default='../lists/zind_panos_val.txt')
    parser.add_argument('--log', type=str, default='log_diff')
    parser.add_argument('--ckpt', type=str, default='ckpt_diff')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint filename')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--rgb-modal', type=str, default='lowres')
    parser.add_argument('--semantic-modal', type=str, default='semantic')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--diffnet-nf', type=int, default=32)
    parser.add_argument('--disc-nf', type=int, default=32)
    parser.add_argument('--diffnet-lr', type=float, default=1e-4)
    parser.add_argument('--diffnet-wd', type=float, default=1e-5)
    parser.add_argument('--disc-lr', type=float, default=1e-4)
    parser.add_argument('--disc-wd', type=float, default=1e-5)
    parser.add_argument('--w-recon', type=float, default=1)
    parser.add_argument('--w-excl', type=float, default=1)
    parser.add_argument('--w-adv', type=float, default=10)
    parser.add_argument('--w-reg', type=float, default=0.1)
    parser.add_argument('--w-d', type=float, default=1)
    parser.add_argument('--mask-thres', type=float, default=0.05)
    parser.add_argument('--mask-dilate', type=int, default=5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--vis-step', type=int, default=10)
    parser.add_argument('--val-step', type=int, default=1)
    parser.add_argument('--save-step', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    opt = parser.parse_args()

    opt.train_dsets = opt.train_dsets.split(',')
    opt.train_lists = opt.train_lists.split(',')
    opt.val_dsets = opt.val_dsets.split(',')
    opt.val_lists = opt.val_lists.split(',')
    opt.log = Path(opt.log)
    opt.ckpt = Path(opt.ckpt)

    if opt.resume:
        checkpoint = torch.load(opt.resume)
        if not opt.finetune:
            opt = checkpoint['opt']
    else:
        checkpoint = None
        remkdir(opt.log)
        remkdir(opt.ckpt)
    return opt, checkpoint


def run(stage, opt, epoch, data_loader, diffnet, disc, vgg, diffnet_optim=None, disc_optim=None):
    if stage == 'train':
        np.random.seed()
        diffnet = diffnet.train()
        torch.autograd.set_grad_enabled(True)
    else:
        np.random.seed(0)
        diffnet = diffnet.eval()
        torch.autograd.set_grad_enabled(False)

    logger = Logger(opt.log, 'diff', stage, epoch, len(data_loader), ['loss', 'recon', 'excl', 'adv', 'reg', 'd'])
    for iteration, batch in enumerate(data_loader):
        index, dset_id, record_id, rgb, sem, spec, sunl = batch

        rgb, sem, spec, sunl = to_device([rgb, sem, spec, sunl], opt.device)
        sem_floor = (sem == SEM_FLOOR)[:,None].float()

        spec = spec * sem_floor

        mask = (spec.mean(1, keepdim=True) > opt.mask_thres).float() * \
               (spec.mean(1, keepdim=True) > sunl.mean(1, keepdim=True))
        mask = morpho_dilate_pano(mask, opt.mask_dilate) * sem_floor

        diff = diffnet(rgb)

        feats_pred = vgg(diff + spec)
        with torch.no_grad():
            feats_rgb = vgg(rgb)

        losses = {}
        losses['recon'] = opt.w_recon * (F.l1_loss(diff + spec, rgb) + perceptual_loss(feats_pred, feats_rgb))
        losses['excl'] = opt.w_excl * masked_excl_pano(diff, spec, None, 2)

        din = diff * sem_floor
        dout = disc.eval()(din)
        dout = F.interpolate(dout, (mask.shape[2], mask.shape[3]), mode='bilinear')
        losses['adv'] = opt.w_adv * masked_bce_logits(dout, 1 - mask, mask)

        pred = diffnet(diff)
        feats_pred = vgg(pred)
        with torch.no_grad():
            feats_diff = vgg(diff)
        losses['reg'] = opt.w_reg * (F.l1_loss(pred, diff) + perceptual_loss(feats_pred, feats_diff))

        losses['loss'] = losses['recon'] + losses['excl'] + losses['adv'] + losses['reg']

        if stage == 'train':
            diffnet.zero_grad(set_to_none=True)
            losses['loss'].backward()
            diffnet_optim.step()
            dout = disc.train()(din.detach())
            dout = F.interpolate(dout, (mask.shape[2], mask.shape[3]), mode='bilinear')
            losses['d'] = opt.w_d * masked_bce_logits(dout, mask, sem_floor)
            disc.zero_grad(set_to_none=True)
            losses['d'].backward()
            disc_optim.step()

        if stage == 'val' and iteration % opt.vis_step == 0:
            gt_data = [(rgb, 'im', 'im'), (spec, 'im', 'spec'), (mask * sem_floor, 'im', 'mask')]
            pred_data = [(diff, 'im', 'diff'), (torch.sigmoid(dout) * sem_floor, 'im', 'dout')]
            save_vis(opt.log / 'gt', data=gt_data, pref=iteration, overwrite=False)
            save_vis(opt.log / str(epoch), data=pred_data, pref=iteration, overwrite=True)

        logger.log_step(iteration, len(index), losses)
    logger.log_epoch()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt, checkpoint = parse_args()
    print_log(opt, opt.log / 'opt.txt')

    diffnet = RegNet(nf=opt.diffnet_nf).to(opt.device)
    diffnet_optim = optim.Adam(diffnet.parameters(), lr=opt.diffnet_lr, weight_decay=opt.diffnet_wd)

    disc = DiscNet(nf=opt.disc_nf).to(opt.device)
    disc_optim = optim.Adam(disc.parameters(), lr=opt.disc_lr, weight_decay=opt.disc_wd)

    vgg = VGG().to(opt.device).eval()

    train_set = PanoDataset(dsets=opt.train_dsets, record_lists=opt.train_lists, height=opt.height, modals=[opt.rgb_modal, opt.semantic_modal, 'specular', 'sunlight'], rotate=True)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.bs, num_workers=opt.threads, shuffle=True, pin_memory=True)
    val_set = PanoDataset(dsets=opt.val_dsets, record_lists=opt.val_lists, height=opt.height, modals=[opt.rgb_modal, opt.semantic_modal, 'specular', 'sunlight'], rotate=False)
    val_loader = DataLoader(dataset=val_set, batch_size=1, num_workers=opt.threads, shuffle=False, pin_memory=True)

    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        diffnet.load_state_dict(checkpoint['diffnet'])
        disc.load_state_dict(checkpoint['disc'])
        if not opt.finetune:
           diffnet_optim.load_state_dict(checkpoint['diffnet_optim'])
           disc_optim.load_state_dict(checkpoint['disc_optim'])
    else:
        start_epoch = 0

    diffnet = nn.DataParallel(diffnet)
    disc = nn.DataParallel(disc)

    for epoch in range(start_epoch, opt.n_epochs):
        print()
        run('train', opt, epoch, train_loader, diffnet, disc, vgg, diffnet_optim, disc_optim)
        if epoch % opt.save_step == opt.save_step - 1:
            torch.save({'opt':opt, 'epoch':epoch, 'diffnet':diffnet.module.state_dict(), 'diffnet_optim':diffnet_optim.state_dict(), 'disc':disc.module.state_dict(), 'disc_optim':disc_optim.state_dict()}, f'{opt.ckpt}/{epoch}.pth')

        if epoch % opt.val_step == opt.val_step - 1:
            run('val', opt, epoch, val_loader, diffnet, disc, vgg)
