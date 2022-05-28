import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import argparse
import torch.nn.functional as F
import kornia
from datasets.pano_reader import get_reader
from utils.misc import *
from utils.transform import *
from utils.ops import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default='../lists/data/zind_panos_train.txt')
    parser.add_argument('--record', type=str, default=None)
    parser.add_argument('--out', type=str, default='csunl_out')
    parser.add_argument('--topk', type=int, default=5)
    opt = parser.parse_args()
    remkdir(opt.out)
    return opt


def get_sun_range(azimuth_low, azimuth_high, azimuth_res, elevation_low, elevation_high, elevation_res):
    azimuth = (torch.arange(azimuth_low, azimuth_high, azimuth_res) + 360) % 360 * math.pi / 180
    elevation = torch.arange(elevation_low, elevation_high, elevation_res) * math.pi / 180
    sun_xy = torch.stack((torch.cos(azimuth), torch.sin(azimuth)), 1)[:, None].expand(azimuth.shape[0], elevation.shape[0], 2)
    sun_z = -torch.tan(elevation)[None, :, None].expand(azimuth.shape[0], elevation.shape[0], 1)
    sun = torch.cat((sun_xy, sun_z), 2).reshape(-1, 3).cuda()
    return sun, azimuth, elevation


def project_f2w(suns, pos, rgb, mask=None):
    suns, pos, rgb = suns.cuda(), pos.cuda(), rgb.cuda()
    pix = sun_floor2wall_grid(suns, pos)
    prgb = grid_sample(rgb, pix)
    if mask is not None:
        mask = mask.cuda()
        pmask = grid_sample(mask, pix)
        prgb, pmask = prgb.cpu(), pmask.cpu()
    else:
        return prgb.cpu()
    return prgb, pmask

def project_w2w(suns, pos, rgb, mask=None):
    suns, pos, rgb = suns.cuda(), pos.cuda(), rgb.cuda()
    pix, valid = sun_wall2wall_grid(suns, pos)
    prgb = grid_sample(rgb, pix) * valid
    if mask is not None:
        mask = mask.cuda()
        pmask = grid_sample(mask, pix) * valid
        prgb, pmask = prgb.cpu(), pmask.cpu()
    else:
        return prgb.cpu()
    return prgb, pmask

def project_w2f(suns, pos, rgb, mask=None):
    suns, pos, rgb = suns.cuda(), pos.cuda(), rgb.cuda()
    pix = sun_wall2floor_grid(suns, pos)
    prgb = grid_sample(rgb, pix)
    if mask is not None:
        mask = mask.cuda()
        pmask = grid_sample(mask, pix)
        prgb, pmask = prgb.cpu(), pmask.cpu()
    else:
        return prgb.cpu()
    return prgb, pmask


def search_sun(suns, rgb, pos, arch_floor, arch_wall, sem_floor, sem_window):
    """
        suns: N x 3[xyz]
        rgb: B x 3[rgb] x H x W
        pos: B x 3[xyz] x H x W
        sem_floor: B x 1 x H x W
        sem_window: B x 1 x H x W
    """
    # backward projection
    prgbs, pmasks = [], []
    for i in range(rgb.shape[0]):
        prgb, pmask = project_f2w(suns, pos[i:i+1], rgb[i:i+1], sem_floor[i:i+1])
        prgbs.append(prgb)
        pmasks.append(pmask)
    prgbs, pmasks = torch.stack(prgbs, 1), torch.stack(pmasks, 1) # N x B x C x H x W
    N, B, C, H, W = prgbs.shape
    grays = prgbs.mean(2, keepdim=True)
    gray = grays.reshape(N, B, H, W)
    mask = pmasks.reshape(N, B, H, W)

    # get template mask, B x H x W
    inner = sem_window.reshape(1, B, H, W)
    outter = (kornia.filters.box_blur(sem_window, (15, 15), 'circular') > 1e-10).float() * arch_wall
    outter = outter.reshape(1, B, H, W) * (1 - inner)

    inner_mean = weighted_mean(gray * mask, inner * mask, dim=(1,2,3))
    outter_mean = weighted_mean(gray, outter * mask, dim=(1,2,3))
    score = inner_mean - outter_mean

    return score

def sem2geo(sem):
    B, H, W = sem.shape
    h = H // 2
    sem_h = sem[:,h:]
    sem_floor_h = (sem_h == SEM_FLOOR)[:,None].float()
    sem_wall_h = (sem_h == SEM_WALL)[:,None].float()
    fline_wall_h = h - sem_wall_h.flip(2).argmax(dim=2)
    fline_floor_h = sem_floor_h.argmax(dim=2)
    has_wall_h = sem_wall_h.max(dim=2)[0]
    has_floor_h = sem_floor_h.max(dim=2)[0]
    fline_h = ((fline_wall_h + fline_floor_h) / 2 * has_wall_h * has_floor_h + \
              fline_wall_h * has_wall_h * (1 - has_floor_h) + \
              fline_floor_h * (1 - has_wall_h) * has_floor_h)[:,:,None]
    fline = h + fline_h #B11W
    fline_lower = fline.floor().long() #B11W
    fline_upper = fline.ceil().long() # B11W
    fline_alpha = fline - fline_lower.float()
    coor = torch.linspace(0, H-1, H)[None,None,:,None].expand(B, 1, H, W)
    arch_floor = (coor >= fline).float() #B1HW
    arch_wall = 1 - arch_floor

    uv = uvgrid(W, H, B)
    pix = uv2pix(uv)
    sph = pix2sph(pix)
    phi = sph[:,1:2]
    # assume camera height = 1
    rho_floor = -1 / torch.sin(phi)
    sph_floor = torch.cat((sph, rho_floor), 1)
    pos_floor = sph2cart(sph_floor) # floor position
    pos_wall_xy_lower = torch.gather(pos_floor[:,:2], 2, fline_lower.expand(B, 2, H, W))
    pos_wall_xy_upper = torch.gather(pos_floor[:,:2], 2, fline_upper.expand(B, 2, H, W))
    pos_wall_xy = pos_wall_xy_upper * fline_alpha + pos_wall_xy_lower * (1 - fline_alpha)
    pos_wall_z = torch.linalg.norm(pos_wall_xy, dim=1, keepdim=True) * torch.tan(phi)
    pos_wall = torch.cat((pos_wall_xy, pos_wall_z), 1)
    pos = pos_floor * arch_floor + pos_wall * arch_wall

    arch = torch.zeros_like(sem)
    arch[arch_floor[:,0] > 0.5] = ARCH_FLOOR
    arch[arch_wall[:,0] > 0.5] = ARCH_WALL

    return arch, pos


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    opt = parse_args()
    print(opt)

    reader = get_reader('zind')
    if opt.list is None:
        records = [opt.record.replace('+',' '),]
    else:
        records = read_list(opt.list)

    for i, record in enumerate(records):
        print(i, '/', len(records), record)
        rgb = reader.get_rgb_image(record)[None]
        sem = reader.get_semantic_image(record)[None]
        arch, pos = sem2geo(sem)
        arch_floor = (arch == ARCH_FLOOR)[:,None].float()
        arch_wall = (arch == ARCH_WALL)[:,None].float()
        sem_floor = (sem == SEM_FLOOR)[:,None].float() * arch_floor
        sem_wall = (sem == SEM_WALL)[:,None].float() * arch_wall
        sem_window = (sem == SEM_WINDOW)[:,None].float() * arch_wall

        suns, azimuths, elevations = get_sun_range(0, 360, 5, 15, 75, 5)
        scores = search_sun(suns, rgb, pos, arch_floor, arch_wall, sem_floor, sem_window)

        scores = scores.reshape(len(azimuths), len(elevations))
        scores, azimuth_idx = torch.max(scores, dim=0)
        elevation_idx = torch.arange(len(elevations))
        idx = azimuth_idx * len(elevations) + elevation_idx
        sorted_idx = torch.argsort(scores, descending=True)[:opt.topk]
        idx = idx[sorted_idx]

        mask_w2f = project_w2f(-suns[idx], pos, sem_window)
        mask_w2w = project_w2w(-suns[idx], pos, sem_window)

        coarse_sunl = mask_w2f.max(0).values * arch_floor + mask_w2w.max(0).values * arch_wall
        save_im(Path(opt.out) / (record.split(' ')[1] + '_csunl.png'), coarse_sunl)
