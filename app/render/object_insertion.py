import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import argparse
import sh
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

import lxml.etree as ET
from xml.dom import minidom

from datasets.pano_reader import get_reader
from utils.misc import *
from utils.transform import *
from utils.mtlparser import MTLParser
from utils.ops import *
from renderer_helper import *

#test+1326+floor_01+pano_3
# test+1326+floor_01+pano_19
# test+1326+floor_01+pano_12
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--house', type=str, default='1037')
    parser.add_argument('--floor', type=str, default='floor_02')
    parser.add_argument('--pano', type=str, default='pano_43')

    parser.add_argument('--cube_only', action='store_false')

    parser.add_argument('--log', type=str, default='out_obj')
    parser.add_argument('--cache', type=str, default='obj_cache')
    parser.add_argument('--floor_texture', type=str, default=None)
    parser.add_argument('--floor_material', type=str, default=None)

    parser.add_argument('--sun_thresh_floor_sunl', type=int, default=0.0)
    parser.add_argument('--sun_thresh_wall_sunl', type=int, default=0.0)

    parser.add_argument('--sun_thresh_floor', type=int, default=0.0)
    parser.add_argument('--sun_thresh_wall', type=int, default=0.0)
    parser.add_argument('--floor2window_thresh', type=int, default=0.0)
    parser.add_argument('--wall2window_thresh', type=int, default=0.0)
    parser.add_argument('--manual', action='store_false')
    parser.add_argument('--is_high_res', action='store_false')
    parser.add_argument('--use_window_fitting', action='store_true')
    parser.add_argument('--use_ambi_hdr', action='store_false')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--mask-samples', type=int, default=1)
    parser.add_argument('--sun-samples', type=int, default=1)
    parser.add_argument('--spec-samples', type=int, default=1)
    parser.add_argument('--blur-ksize', type=int, default=0)
    parser.add_argument('--tex_res', type=int, default=2048)
    parser.add_argument('--delta_azimuth', type=int, default=0)
    parser.add_argument('--delta_ele', type=int, default=0)
    parser.add_argument('--pano_scale', type=int, default=1)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--is_cuda', action='store_true')
    parser.add_argument('--use_cache_ambi', action='store_true')
    parser.add_argument('--floor_erosion', type=int, default=3)
    parser.add_argument('--use_old_method', action='store_true')
    parser.add_argument('--paste_boundary', action='store_false')
    parser.add_argument('--use_guided_filter', action='store_true')
    parser.add_argument('--guided_thresh', type=float, default=0.73)
    parser.add_argument('--guided_r', type=int, default=9)
    parser.add_argument('--is_no_sun', action='store_true')
    parser.add_argument('--is_tripod', action='store_true')

    parser.add_argument('--layout-path', type=str, default='../data/zind/scenes/layout_merge')

    opt = parser.parse_args()
    opt.cache = f'{opt.cache}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}'
    os.makedirs(opt.cache, exist_ok=True)
    opt.log = f'{opt.log}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}'

    remkdir(opt.log)
    with open(f'{opt.log}/opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    return opt



if __name__ == '__main__':
    start_time = time.time()
    opt = parse_args()

    print(opt)
    # ambi_i_hdr_dir = None
    if opt.blur_ksize > 0:
        blur_tex(f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_wall.exr', f'{opt.cache}/tex_wall_blur.exr', ksize=opt.blur_ksize)
        blur_tex(f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.exr', f'{opt.cache}/tex_ceiling_blur.exr', ksize=opt.blur_ksize)

    reader = get_reader('zind', opt.height)
    reader_hr = get_reader('zind', height=256)
    reader_lr = get_reader('zind', height=256)

    record = f'{opt.split} {opt.house} {opt.floor} {opt.pano}'
    is_high_res = opt.is_high_res

    sem_i_hr = reader_hr.get_arch_image(record, 'arch_hr')[None]
    rgb_i = reader.get_rgb_image(record, 'rgb')[None]

    if opt.is_high_res:
        diff_i = reader.get_rgb_image(record, 'hrdiff')[None]
        ambi_i = reader.get_rgb_image(record, 'hrambi')[None]
        spec_i = reader.get_rgb_image(record, 'hrspec')[None]

        sem_i = reader.get_arch_image(record, 'arch_hr')[None]
        # sunlight_i = reader.get_rgb_image(record, 'hrsunl')[None]
        sunlight_i = torch.clamp(rgb_i - ambi_i, min=0)
        print(sunlight_i.min())

    else:
        diff_i = reader.get_rgb_image(record, 'diff')[None]
        ambi_i = reader.get_rgb_image(record, 'ambi')[None]
        spec_i = reader.get_rgb_image(record, 'spec')[None]
        # spec_i = rgb_i - diff_i
        sem_i = reader.get_arch_image(record, 'arch')[None]

    extr = np.asarray([0, 0, 0, 1, 1.6, 7], dtype=np.float32)
    extr = torch.from_numpy(extr)[None]

    json_path = f'{opt.layout_path}/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/pred.json'
    layout_json = json.load(open(json_path))
    sem_floor_i = (sem_i == ARCH_FLOOR).float()[:, None]
    sem_wall_i = (sem_i == ARCH_WALL).float()[:, None]
    zf = ZillowFloor(layout_json)

    # for reproduction
    to_world_path = f'../data/zind/scenes/floormesh/{opt.split}/{opt.split}+{opt.house}+{opt.floor}+{opt.pano}/toworld.xml'
    if os.path.exists(to_world_path):
        tree = ET.parse(to_world_path)
        root = tree.getroot()
        for child in root:
            if child.tag == 'translate':
                ratio = -float(child.attrib['z'])/ extr[0, 4].item()

        for key in zf.pano_data.keys():
            zf.pano_data[key] = zf.pano_data[key] * ratio



    # get sun
    sun = reader.get_sundir_vector(record)[None]
    is_no_sun = sun[0, 0].item() < 0.01 or opt.is_no_sun
    is_no_change_sun = (opt.delta_azimuth == 0) and (opt.delta_ele == 0)

    if not opt.use_cache_ambi:
        AmbiF = render_ambient(opt, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
        if opt.denoise:
            AmbiF1 = render_ambient(opt, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
            AmbiF2 = render_ambient(opt, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
            AmbiF = torch.median(torch.stack((AmbiF, AmbiF1, AmbiF2), 0), 0).values
        save_im(f'{opt.log}/render_AmbiF.png', AmbiF ** (1 / 2.2))
        save_hdr(f'{opt.cache}/render_AmbiF.exr', AmbiF)

        AmbiE = render_ambient(opt, furn=False, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
        if opt.denoise:
            AmbiE1 = render_ambient(opt, furn=False, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
            AmbiE2 = render_ambient(opt, furn=False, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
            AmbiE = torch.median(torch.stack((AmbiE, AmbiE1, AmbiE2), 0), 0).values
        save_im(f'{opt.log}/render_AmbiE.png', AmbiE ** (1 / 2.2))
        save_hdr(f'{opt.cache}/render_AmbiE.exr', AmbiE)


        AmbiR = render_ambient(opt, white_floor=False, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
        if opt.denoise:
            AmbiR1 = render_ambient(opt, white_floor=False, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
            AmbiR2 = render_ambient(opt, white_floor=False, wall_tex_dir=None, floor_tex_dir=None if is_no_change_sun else ambi_floor_tex_dir)
            AmbiR = torch.median(torch.stack((AmbiR, AmbiR1, AmbiR2), 0), 0).values
        save_im(f'{opt.log}/render_AmbiR.png', AmbiR ** (1 / 2.2))
        save_hdr(f'{opt.cache}/render_AmbiR.exr', AmbiR)
    else:
        print(f'Use rendered Ambi image in {opt.cache}')
        AmbiE = torch.from_numpy(cv2.imread(f'{opt.cache}/render_AmbiE.exr', -1)[:, :, [2, 1, 0]]).permute(2, 0, 1)[None]
        AmbiF = torch.from_numpy(cv2.imread(f'{opt.cache}/render_AmbiF.exr', -1)[:, :, [2, 1, 0]]).permute(2, 0, 1)[None]
        AmbiR = torch.from_numpy(cv2.imread(f'{opt.cache}/render_AmbiR.exr', -1)[:, :, [2, 1, 0]]).permute(2, 0, 1)[None]

    if is_no_sun:
        SunM, SunE = 0, 0
        I = diff_i
        SunR = 0
    else:


        if not opt.use_old_method:
            azimuth_ori, elevation_ori = sun[0, 1].item(), sun[0, 2].item()
            azimuth_new, elevation_new = azimuth_ori + opt.delta_azimuth, elevation_ori + opt.delta_ele
            sun_new = sun_angle2tri(azimuth_new, elevation_new)[None]

            sun = wld2cam(sun[:, 3:, None, None], extr, no_trans=True)[:, :, 0, 0]
            sun_new = wld2cam(sun_new[:, :, None, None], extr, no_trans=True)[:, :, 0, 0]

            # if opt.is_high_res:
            #     pos = arch2pos(sem_i_hr, camera_height=extr[0, 4], scale=1)
            # else:
            #     pos = arch2pos(sem_i_hr, camera_height=extr[0, 4], scale=4)
            pos = render_pos(opt, furn=False)

            if opt.is_tripod:
                sem = reader_lr.get_semantic_image(record)

                sem = sem[None, None]
                sem_tripod = (sem == 0).float()
                # sem_tripod = morpho_dilate_pano(sem_tripod, 9)
                sem_tripod = torch2np_im(sem_tripod)[..., 0]
                sem_tripod = cv2.resize(np.uint8(np.array(sem_tripod)), (512, 256), interpolation=cv2.INTER_NEAREST)
                sem_tripod = np2torch_im(sem_tripod[..., None])
                save_im(f'{opt.log}/sunlight_i_ori.png', sunlight_i)

                sunlight_i = sunlight_i * (1-sem_tripod)
                print(sunlight_i.min())
            # compute the mask of the "window" for the light source

            sunlight_i = (sunlight_i - opt.sun_thresh_floor_sunl) / (1 - opt.sun_thresh_floor_sunl) * sem_floor_i + \
                         (sunlight_i - opt.sun_thresh_wall_sunl) / (1 - opt.sun_thresh_wall_sunl) * sem_wall_i
            sunlight_i = sunlight_i ** 2.2

            save_im(f'{opt.log}/sunlight_i.png', sunlight_i)

            # diff to ambi ratio
            sunlight_i_div = diff_i / torch.clamp(ambi_i, 0.001) - 1
            save_im(f'{opt.log}/sunlight_i_div.png', sunlight_i_div)


            pano_sun_div, pano_sun_div_floor_mask, pano_sun_div_wall_mask = project_sun_forward(opt, sun, sun_new, sunlight_i_div, ambi_i, pos, sem_floor_i,
                                                                                                sem_wall_i, use_main_color=False, is_backward=False, return_mask=True,
                                                                                                sun_thresh_wall=opt.sun_thresh_wall, sun_thresh_floor=opt.sun_thresh_floor)
            pano_sun, pano_sun_floor_mask, pano_sun_wall_mask = project_sun_forward(opt, sun, sun_new, sunlight_i, ambi_i, pos, sem_floor_i, sem_wall_i, use_main_color=False, is_backward=False, return_mask=True)
            save_im(f'{opt.log}/pano_sun.png', pano_sun)


            # build wall mask tex
            wall_tex, _, _ = zf.build_tex(None, tex_pano=None, pano_im=torch2np_im(pano_sun), res=opt.tex_res)
            # plt.imshow(wall_tex)
            # plt.show()

            '''
            get the wall_id for the sunlight region on wall (i.e., windows)
            '''
            bright_wall_ids, dark_wall_ids = zf.get_wall_id_for_division(wall_tex, opt.tex_res)

            # build texture (diff to ambi ratio)
            # dark_wall_tex, dark_floor_tex, _ = zf.build_tex(
            #     im_template=reader.template['ambi'].format(split=opt.split, house=opt.house, floor=opt.floor,
            #                                                pano=opt.pano), tex_pano=None, pano_im=None, res=opt.tex_res,  wall_ids=dark_wall_ids)
            bright_wall_tex, _, _ = zf.build_tex(None, tex_pano=None, pano_im=torch2np_im(pano_sun), res=opt.tex_res, wall_ids=bright_wall_ids)
            bright_wall_tex = torch.Tensor(bright_wall_tex).permute(2, 0, 1)[None]
            bright_wall_tex = bright_wall_tex.mean(1, keepdim=True)
            # dark_floor_tex = np.flip(dark_floor_tex, axis=0)
            # save_im(f'{opt.cache}/tex_floor_ambi.png', torch.Tensor(dark_floor_tex.copy()).permute(2, 0, 1)[None] /255)

            # build mask of texture (diff to ambi ratio)
            bright_wall_tex_floor_mask, _, _ = zf.build_tex(None, tex_pano=None, pano_im=torch2np_im(pano_sun_floor_mask), res=opt.tex_res,
                                                 wall_ids=bright_wall_ids, mode='nearest')
            bright_wall_tex_floor_mask = torch.Tensor(bright_wall_tex_floor_mask).permute(2, 0, 1)[None]
            bright_wall_tex_wall_mask, _, _ = zf.build_tex(None, tex_pano=None, pano_im=torch2np_im(pano_sun_wall_mask), res=opt.tex_res,
                                                 wall_ids=bright_wall_ids, mode='nearest')
            bright_wall_tex_wall_mask = torch.Tensor(bright_wall_tex_wall_mask).permute(2, 0, 1)[None]

            # build mesh
            bright_wall_mesh, _, _ = zf.build_mesh(None, None, None, wall_ids=bright_wall_ids)
            dark_wall_mesh, _, _ = zf.build_mesh(None , None, None, wall_ids=dark_wall_ids)
            bright_wall_mesh_dir = f'{opt.cache}/bright_wall_mesh'
            dark_wall_mesh_dir = f'{opt.cache}/dark_wall_mesh'
            save_obj(bright_wall_mesh_dir, bright_wall_mesh, )
            save_obj(dark_wall_mesh_dir, dark_wall_mesh, )

            # build retreat mesh
            bright_rt_wall_mesh, _, _ = zf.build_mesh(None, None, None, wall_ids=bright_wall_ids, retreat=True, eps=-4.5e-3)
            dark_rt_wall_mesh, _, _ = zf.build_mesh(None, None, None, wall_ids=dark_wall_ids, retreat=True, eps=4.5e-3)
            bright_rt_wall_mesh_dir = f'{opt.cache}/bright_rt_wall_mesh'
            dark_rt_wall_mesh_dir = f'{opt.cache}/dark_rt_wall_mesh'
            save_obj(bright_rt_wall_mesh_dir, bright_rt_wall_mesh, )
            save_obj(dark_rt_wall_mesh_dir, dark_rt_wall_mesh, )


            # compute the normal for the scene
            normal = render_normal(opt)


            '''
            fit the shape of "window" based on the window prior (horizontal, vertical straight line etc)
            '''

            if opt.use_window_fitting:
                # find the bounding box
                tex_fit_mask = fit_bbox(bright_wall_tex)

                # decide which part as source based on the area
                if opt.window_fitting_src is None:
                    wall_as_source = bright_wall_tex_wall_mask.sum() > bright_wall_tex_floor_mask.sum()
                elif opt.window_fitting_src == 'wall':
                    wall_as_source = True
                elif opt.window_fitting_src == 'floor':
                    wall_as_source = False

                if wall_as_source:
                    save_im(f'{opt.log}/bright_wall_tex_wall_mask.png', bright_wall_tex_wall_mask)
                    print('Choose Wall as source for window fitting')
                    bright_wall_tex = fill_im(bright_wall_tex, bright_wall_tex_wall_mask, tex_fit_mask,
                                              opt.wall2window_thresh, use_main_color=False)
                else:
                    save_im(f'{opt.log}/bright_wall_tex_wall_mask.png', bright_wall_tex_floor_mask)

                    print('Choose Floor as source for window fitting')
                    bright_wall_tex = fill_im(bright_wall_tex, bright_wall_tex_floor_mask, tex_fit_mask,
                                              opt.floor2window_thresh, use_main_color=False)
            else:
                bright_wall_tex = ((bright_wall_tex * bright_wall_tex_wall_mask) > opt.wall2window_thresh) + \
                                  ((bright_wall_tex * bright_wall_tex_floor_mask) > opt.floor2window_thresh)

            '''
            Sunlight on the wall and floor
             Mask of Sun cast on the floor and wall 
             '''

            SunM_Floor = render_sunlight(opt, sun_new,  torch.ones_like(bright_wall_tex[:, :1]).float(), bright_wall_mesh_dir,
                               dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, maxd=4, furn_bsdf='black', wall_bsdf='black',  furn=True,
                               samples=opt.mask_samples, normal=normal)


            SunM_Wall = render_sunlight(opt, sun_new, torch.ones_like(bright_wall_tex[:, :1]).float(), bright_wall_mesh_dir,
                               dark_wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, maxd=4, furn_bsdf='black', floor_bsdf='black',  furn=True,
                               samples=opt.mask_samples, normal=normal)


            SunM_Floor = SunM_Floor
            SunM_Wall = SunM_Wall
            SunM = SunM_Floor + SunM_Wall

            '''
            Sunlight on the objects
            '''
            if opt.use_window_fitting:
                pano_sun = pano_sun.mean(1, keepdim=True)
                main_sun_color = get_dominant_color(torch2np_im(pano_sun), torch2np_im(pano_sun_floor_mask ))
                bright_wall_tex_obj = (bright_wall_tex > 0) * torch.Tensor(main_sun_color[None, :, None, None])
            else:
                bright_wall_tex_obj, _, _ = zf.build_tex(None, tex_pano=None, pano_im=torch2np_im(pano_sun),
                                                     res=opt.tex_res, wall_ids=bright_wall_ids)
                bright_wall_tex_obj = torch.Tensor(bright_wall_tex_obj).permute(2, 0, 1)[None]
                bright_wall_tex_obj = bright_wall_tex_obj.mean(1, keepdim=True)
            #
            SunR = render_sunlight(opt, sun_new, bright_wall_tex_obj, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_wall_mesh_dir, dark_wall_mesh_dir,
                                   floor_bsdf='black', wall_bsdf='black', furn_bsdf='full', maxd=4, samples=opt.sun_samples, normal=None)
            #
            # SunR = render_sunlight(opt, sun_new, bright_wall_tex_obj, bright_wall_mesh_dir, dark_wall_mesh_dir, bright_wall_mesh_dir, dark_wall_mesh_dir,
            #                        floor_bsdf='white', wall_bsdf='white', maxd=4, samples=opt.sun_samples, normal=None, furn=False)
            if opt.denoise:
                SunR1 = render_sunlight(opt, sun_new, bright_wall_tex_obj, bright_wall_mesh_dir, dark_wall_mesh_dir,bright_wall_mesh_dir, dark_wall_mesh_dir,
                                        floor_bsdf='black', wall_bsdf='black', maxd=4, samples=opt.sun_samples, normal=None)
                SunR2 = render_sunlight(opt, sun_new, bright_wall_tex_obj, bright_wall_mesh_dir, dark_wall_mesh_dir,bright_wall_mesh_dir, dark_wall_mesh_dir,
                                        floor_bsdf='black', wall_bsdf='black', maxd=4, samples=opt.sun_samples, normal=None)
            save_im(f'{opt.log}/render_SunR.png', SunR)



    SpecM = render_specular(opt, torch.ones_like(spec_i[:, :1]), samples=opt.spec_samples)
    SpecF = spec_i * SpecM
    SpecE = spec_i
    save_im(f'{opt.log}/render_SpecM.png', SpecM)
    save_im(f'{opt.log}/render_SpecF.png', SpecF)
    save_im(f'{opt.log}/render_SpecE.png', SpecE)

    F = AmbiF
    E = AmbiE
    R = AmbiR + SunR
    save_im(f'{opt.log}/render_E.png', E ** (1 / 2.2))
    save_im(f'{opt.log}/render_F.png', F ** (1 / 2.2))
    save_im(f'{opt.log}/render_R.png', R ** (1 / 2.2))

    Mfloor = sem_floor_i
    MWall = sem_wall_i
    Mobj = render_furnmask(opt)
    Mall = torch.max(Mfloor, Mobj)
    Mall = torch.max(Mall, MWall)

    save_im(f'{opt.log}/render_Mall.png', Mall ** (1 / 2.2))
    save_im(f'{opt.log}/render_Mobj.png', Mobj ** (1 / 2.2))
    save_im(f'{opt.log}/render_Mfloor.png', Mfloor ** (1 / 2.2))

    T = (F + 1e-6) / (E + 1e-6)

    # f_line = sem_floor_i.argmax(dim=2, keepdim=True).flatten()
    # f_value = T[:, :, :, f_line]

    f_value = T.mean(dim=2, keepdim=True)

    T = T * sem_floor_i + (1-sem_floor_i) * f_value



    # T = T * (sem_floor_i + sem_wall_i) + (1 - (sem_floor_i + sem_wall_i))
    # T = T * (sem_floor_i + sem_wall_i) + (1 - (sem_floor_i + sem_wall_i))

    save_im(f'{opt.log}/render_T_ori.png', T)

    T_mask = (Mall - Mobj)
    # T = T * T_mask


    # assume the no soft shadow on the wall and boundary
    # sem_floor_i_er= morpho_erode_pano(sem_floor_i, k=opt.floor_erosion)
    if opt.use_guided_filter:
        sem_floor_i_filter = guided_filter(rgb_i, sem_floor_i, opt.guided_thresh, opt.guided_r)
        save_im(f'{opt.log}/sem_floor_i_filter.png', sem_floor_i_filter)
    else:
        sem_floor_i_filter = sem_floor_i


    T = T * sem_floor_i_filter * T_mask + T_mask * (1-sem_floor_i_filter) * 1
    save_im(f'{opt.log}/render_T_final.png', T)


    # sem_floor_i_filter - sem_floor_i

    # save_im(f'{opt.log}/render_T_ori.png', T)

    # change floor texture
    if opt.floor_texture is not None:
        ambi_new = render_ambient(opt, furn=False, white_floor=False, white_wall=False, wall_tex_dir=ambi_wall_tex_dir,
                                  floor_tex_dir=opt.floor_texture, sample_scale=1)
        save_im(f'{opt.log}/ambi_new.png', ambi_new * (1 / 2.2))

    if opt.floor_material is not None:
        ambi_new = render_diffuse(opt, furn=False, white_floor=False, white_wall=False, wall_tex_dir=ambi_wall_tex_dir,
                                  floor_tex_dir=opt.floor_texture, sample_scale=1)
        save_im(f'{opt.log}/ambi_new.png', ambi_new * (1 / 2.2))


    if opt.floor_texture is None:
        if opt.delta_azimuth == 0 and  opt.delta_ele == 0:
            # does not change the sun direction, paste the floor and wall in the original region
            # SunM = ( (SunM < 1.2)).float()
            SunF = (diff_i - ambi_i) * (SunM)
            save_im(f'{opt.log}/diff_ambi.png', (diff_i - ambi_i))
            # save_im(f'{opt.log}/SunM.png', SunM)

            I = SunF + ambi_i * T + rgb_i * (1 - Mall)
            # I = I * sem_floor_i + rgb_i * (1-sem_floor_i)
        else:
            SunF = (Sun_Floor + Sun_Wall) * SunM
            I =  SunF  + ambi_i * T + diff_i * (1 - Mall)
    else:
        SunF = (Sun_Floor + Sun_Wall) * SunM
        I = SunF + ambi_i * T * (T_mask-sem_floor_i) + ambi_new * T * sem_floor_i + diff_i * (1 - Mall)
    save_im(f'{opt.log}/SunF.png', SunF)

    save_im(f'{opt.log}/render_I.png', I)

    if opt.paste_boundary:
        # paste boundary from Input Image
        sem_floor_i_rm, _ = remove_small_island(sem_floor_i, min_size=100)
        sem_floor_i_erose = morpho_erode_pano(sem_floor_i_rm, 3)
        sem_floor_i_dilate = morpho_dilate_pano(sem_floor_i_rm, 3)

        b_mask = sem_floor_i_dilate - sem_floor_i_erose
        b_mask = b_mask * (1-Mobj)
        I = I * (1-b_mask) + b_mask * ambi_i
        save_im(f'{opt.log}/b_mask.png', b_mask)

    save_im(f'{opt.log}/render_I_clean.png', I)


    C = Mobj * (R ** (1/2.2)) + I
    save_im(f'{opt.log}/render_T.png', T)
    save_im(f'{opt.log}/render_CR.png', C)
    C = C + (Mall - Mobj) * SpecF
    save_im(f'render_obj.png', C)

    end_time = time.time()
    print(end_time - start_time)
