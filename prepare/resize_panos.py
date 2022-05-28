import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import argparse
import shutil
import cv2
import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', type=str, default='../data/zind/raw')
    parser.add_argument('--out-path', type=str, default='../data/zind/scenes')
    parser.add_argument('--list-path', type=str, default='../lists')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--height', type=int, default=256)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    rgb_path = Path(opt.out_path) / 'rgb' / opt.split
    lowres_path = Path(opt.out_path) / 'lowres' / opt.split
    rgb_path.mkdir(parents=True, exist_ok=True)
    lowres_path.mkdir(parents=True, exist_ok=True)
    for line in open(f'{opt.list_path}/zind_panos_{opt.split}.txt').readlines():
        _, key = line.strip().split()
        house_id, floor_id, room_id, pano_id = key.split('_')
        os.symlink(os.path.realpath(f'{opt.raw_path}/{house_id}/panos/floor_{floor_id}_partial_room_{room_id}_pano_{pano_id}.jpg'), f'{rgb_path}/{key}.jpg')
        im = cv2.imread(f'{opt.raw_path}/{house_id}/panos/floor_{floor_id}_partial_room_{room_id}_pano_{pano_id}.jpg')
        im = cv2.resize(im, (opt.height * 2, opt.height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{lowres_path}/{key}.png', im)
