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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dsets', type=str, default='zind')
    parser.add_argument('--train-lists', type=str, default='../lists/zind_panos_train.txt')
    parser.add_argument('--val-dsets', type=str, default='zind')
    parser.add_argument('--val-lists', type=str, default='../lists/zind_panos_val.txt')
    parser.add_argument('--log', type=str, default='log_eff')
    parser.add_argument('--ckpt', type=str, default='ckpt_eff')
    parser.add_argument('--resume', type=str, default=None, help='resume checkpoint filename')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--rgb-modal', type=str, default='lowres')
    parser.add_argument('--semantic-modal', type=str, default='semantic')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--effnet-nf', type=int, default=32)
    parser.add_argument('--disc-nf', type=int, default=32)
    parser.add_argument('--effnet-lr', type=float, default=1e-4)
    parser.add_argument('--effnet-wd', type=float, default=1e-5)
    parser.add_argument('--disc-spec-lr', type=float, default=1e-4)
    parser.add_argument('--disc-spec-wd', type=float, default=1e-5)
    parser.add_argument('--disc-sunl-lr', type=float, default=1e-4)
    parser.add_argument('--disc-sunl-wd', type=float, default=1e-5)
    parser.add_argument('--disc-eff-lr', type=float, default=1e-4)
    parser.add_argument('--disc-eff-wd', type=float, default=1e-5)
    parser.add_argument('--window-dilate', type=int, default=21)
    parser.add_argument('--spec-dilate', type=int, default=11)
    parser.add_argument('--sunl-dilate', type=int, default=7)
    parser.add_argument('--w-excl', type=float, default=0.1)
    parser.add_argument('--w-spars-spec', type=float, default=1)
    parser.add_argument('--w-spars-nspec', type=float, default=3)
    parser.add_argument('--w-spars-ivspec', type=float, default=10)
    parser.add_argument('--w-spars-sunl', type=float, default=1)
    parser.add_argument('--w-spars-nsunl', type=float, default=1)
    parser.add_argument('--w-spars-ivsunl', type=float, default=5)
    parser.add_argument('--w-adv-spec', type=float, default=1)
    parser.add_argument('--w-adv-sunl', type=float, default=1)
    parser.add_argument('--w-adv-eff', type=float, default=1)
    parser.add_argument('--w-d-spec', type=float, default=1)
    parser.add_argument('--w-d-sunl', type=float, default=1)
    parser.add_argument('--w-d-eff', type=float, default=1)
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


def run(stage, opt, epoch, data_loader, effnet, disc_spec, disc_sunl, disc_eff, effnet_optim=None, disc_spec_optim=None, disc_sunl_optim=None, disc_eff_optim=None):
    if stage == 'train':
        np.random.seed()
        effnet = effnet.train()
        torch.autograd.set_grad_enabled(True)
    else:
        np.random.seed(0)
        effnet = effnet.eval()
        torch.autograd.set_grad_enabled(False)

    logger = Logger(opt.log, 'eff', stage, epoch, len(data_loader), ['loss', 'spars_spec', 'spars_sunl', 'adv_spec', 'adv_sunl', 'adv_eff', 'd_spec', 'd_sunl', 'd_eff'])
    for iteration, batch in enumerate(data_loader):
        index, dset_id, record_id, rgb, sem, csunl = batch

        rgb, sem, csunl = to_device([rgb, sem, csunl], opt.device)

        sem_window = (sem == SEM_WINDOW)[:,None].float()
        sem_door = (sem == SEM_DOOR)[:,None].float()
        sem_light = (sem == SEM_LIGHT)[:,None].float()
        sem_floor = (sem == SEM_FLOOR)[:,None].float()
        sem_wall = (sem == SEM_WALL)[:,None].float()
        sem_door = (sem == SEM_DOOR)[:,None].float()
        sem_struct = (sem == SEM_STRUCT)[:,None].float()

        valid_spec = sem_floor
        cspec = morpho_dilate_pano(torch.maximum(sem_window.max(2, keepdim=True)[0], sem_light.max(2, keepdim=True)[0]), opt.spec_dilate)
        mask_spec = cspec * valid_spec

        valid_sunl = sem_floor + (sem_wall + sem_door + sem_struct) * (1 - morpho_dilate_pano(sem_window.max(2, keepdim=True)[0], opt.window_dilate))
        csunl = morpho_dilate_pano(csunl, opt.sunl_dilate)
        mask_sunl = csunl * valid_sunl

        valid_eff = torch.minimum(valid_spec, valid_sunl)
        mask_eff = torch.minimum(mask_spec, mask_sunl)

        eff = effnet(rgb)
        spec, sunl = eff[:, :3], eff[:, 3:]
        nspec = rgb - spec
        nsunl = rgb - sunl
        neff = rgb - spec - sunl

        losses = {}

        losses['spars_spec'] = opt.w_spars_spec * (spec * mask_spec).abs().mean() + \
                               opt.w_spars_nspec * (spec * valid_spec * (1 - mask_spec)).abs().mean() + \
                               opt.w_spars_ivspec * (spec * (1 - valid_spec)).abs().mean()
        losses['spars_sunl'] = opt.w_spars_sunl * (sunl * mask_sunl).abs().mean() + \
                               opt.w_spars_nsunl * (sunl * valid_sunl * (1 - mask_sunl)).abs().mean() + \
                               opt.w_spars_ivsunl * (sunl * (1 - valid_sunl)).abs().mean()

        din_spec = nspec * valid_spec
        dout_spec = disc_spec.eval()(din_spec)
        dout_spec = F.interpolate(dout_spec, (mask_spec.shape[2], mask_spec.shape[3]), mode='bilinear')
        losses['adv_spec'] = opt.w_adv_spec * masked_bce_logits(dout_spec, 1 - mask_spec, mask_spec)

        din_sunl = nsunl * valid_sunl
        dout_sunl = disc_sunl.eval()(din_sunl)
        dout_sunl = F.interpolate(dout_sunl, (mask_sunl.shape[2], mask_sunl.shape[3]), mode='bilinear')
        losses['adv_sunl'] = opt.w_adv_sunl * masked_bce_logits(dout_sunl, 1 - mask_sunl, mask_sunl)

        din_eff = neff * valid_eff
        dout_eff = disc_eff.eval()(din_eff)
        dout_eff = F.interpolate(dout_eff, (mask_eff.shape[2], mask_eff.shape[3]), mode='bilinear')
        losses['adv_eff'] = opt.w_adv_eff * masked_bce_logits(dout_eff, 1 - mask_eff, mask_eff)

        losses['loss'] = losses['spars_spec'] + losses['spars_sunl'] + losses['adv_spec'] + losses['adv_sunl'] + losses['adv_eff']

        if stage == 'train':
            effnet.zero_grad(set_to_none=True)
            losses['loss'].backward()
            effnet_optim.step()

            dout_spec = disc_spec.train()(din_spec.detach())
            dout_spec = F.interpolate(dout_spec, (mask_spec.shape[2], mask_spec.shape[3]), mode='bilinear')
            losses['d_spec'] = opt.w_d_spec * masked_bce_logits(dout_spec, mask_spec, valid_spec)
            disc_spec.zero_grad(set_to_none=True)
            losses['d_spec'].backward()
            disc_spec_optim.step()

            dout_sunl = disc_sunl.train()(din_sunl.detach())
            dout_sunl = F.interpolate(dout_sunl, (mask_sunl.shape[2], mask_sunl.shape[3]), mode='bilinear')
            losses['d_sunl'] = opt.w_d_sunl * masked_bce_logits(dout_sunl, mask_sunl, valid_sunl)
            disc_sunl.zero_grad(set_to_none=True)
            losses['d_sunl'].backward()
            disc_sunl_optim.step()

            dout_eff = disc_eff.train()(din_eff.detach())
            dout_eff = F.interpolate(dout_eff, (mask_eff.shape[2], mask_eff.shape[3]), mode='bilinear')
            losses['d_eff'] = opt.w_d_eff * masked_bce_logits(dout_eff, mask_eff, valid_eff)
            disc_eff.zero_grad(set_to_none=True)
            losses['d_eff'].backward()
            disc_eff_optim.step()

        if stage == 'val' and iteration % opt.vis_step == 0:
            gt_data = [(rgb, 'im', 'im'), (mask_spec, 'im', 'mask_spec'), (mask_sunl, 'im', 'mask_sunl')]
            pred_data = [(nspec, 'im', 'nspec'), (spec, 'im', 'spec'), (torch.sigmoid(dout_spec) * valid_spec, 'im', 'dout_spec'),
                         (nsunl, 'im', 'nsunl'), (sunl, 'im', 'sunl'), (torch.sigmoid(dout_sunl) * valid_sunl, 'im', 'dout_sunl'),
                         (neff, 'im', 'neff'), (spec + sunl, 'im', 'eff'), (torch.sigmoid(dout_eff) * valid_eff, 'im', 'dout_eff')]
            save_vis(opt.log / 'gt', data=gt_data, pref=iteration, overwrite=False)
            save_vis(opt.log / str(epoch), data=pred_data, pref=iteration, overwrite=True)

        logger.log_step(iteration, len(index), losses)
    logger.log_epoch()


if __name__ == '__main__':
    cv2.setNumThreads(0)
    opt, checkpoint = parse_args()
    print_log(opt, opt.log / 'opt.txt')

    effnet = RegNet(out_c=6, nf=opt.effnet_nf).to(opt.device)
    effnet_optim = optim.Adam(effnet.parameters(), lr=opt.effnet_lr, weight_decay=opt.effnet_wd)

    disc_spec = DiscNet(nf=opt.disc_nf).to(opt.device)
    disc_spec_optim = optim.Adam(disc_spec.parameters(), lr=opt.disc_spec_lr, weight_decay=opt.disc_spec_wd)
    disc_sunl = DiscNet(nf=opt.disc_nf).to(opt.device)
    disc_sunl_optim = optim.Adam(disc_sunl.parameters(), lr=opt.disc_sunl_lr, weight_decay=opt.disc_sunl_wd)
    disc_eff = DiscNet(nf=opt.disc_nf).to(opt.device)
    disc_eff_optim = optim.Adam(disc_eff.parameters(), lr=opt.disc_eff_lr, weight_decay=opt.disc_eff_wd)

    train_set = PanoDataset(dsets=opt.train_dsets, record_lists=opt.train_lists, height=opt.height, modals=[opt.rgb_modal, opt.semantic_modal, 'coarsesunlight'], rotate=True)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.bs, num_workers=opt.threads, shuffle=True, pin_memory=True)
    val_set = PanoDataset(dsets=opt.val_dsets, record_lists=opt.val_lists, height=opt.height, modals=[opt.rgb_modal, opt.semantic_modal, 'coarsesunlight'], rotate=False)
    val_loader = DataLoader(dataset=val_set, batch_size=1, num_workers=opt.threads, shuffle=False, pin_memory=True)

    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        effnet.load_state_dict(checkpoint['effnet'])
        disc_spec.load_state_dict(checkpoint['disc_spec'])
        disc_sunl.load_state_dict(checkpoint['disc_sunl'])
        disc_eff.load_state_dict(checkpoint['disc_eff'])
        if not opt.finetune:
           effnet_optim.load_state_dict(checkpoint['effnet_optim'])
           disc_spec_optim.load_state_dict(checkpoint['disc_spec_optim'])
           disc_eff_optim.load_state_dict(checkpoint['disc_eff_optim'])
    else:
        start_epoch = 0

    effnet = nn.DataParallel(effnet)
    disc_spec = nn.DataParallel(disc_spec)
    disc_sunl = nn.DataParallel(disc_sunl)
    disc_eff = nn.DataParallel(disc_eff)

    for epoch in range(start_epoch, opt.n_epochs):
        print()
        run('train', opt, epoch, train_loader, effnet, disc_spec, disc_sunl, disc_eff, effnet_optim, disc_spec_optim, disc_sunl_optim, disc_eff_optim)
        if epoch % opt.save_step == opt.save_step - 1:
            torch.save({'opt':opt, 'epoch':epoch, 'effnet':effnet.module.state_dict(), 'effnet_optim':effnet_optim.state_dict(),
                        'disc_spec':disc_spec.module.state_dict(), 'disc_spec_optim':disc_spec_optim.state_dict(),
                        'disc_sunl':disc_sunl.module.state_dict(), 'disc_sunl_optim':disc_sunl_optim.state_dict(),
                        'disc_eff':disc_eff.module.state_dict(), 'disc_eff_optim':disc_eff_optim.state_dict()}, f'{opt.ckpt}/{epoch}.pth')

        if epoch % opt.val_step == opt.val_step - 1:
            run('val', opt, epoch, val_loader, effnet, disc_spec, disc_sunl, disc_eff)
