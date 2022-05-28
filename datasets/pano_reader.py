import configparser
import os
import torch
import math
import numpy as np
import cv2
import json
from pathlib import Path

def get_reader(dset, height=256, width=None):
    if dset == 'zind':
        reader = ZInDReader(height, width)
    else:
        assert(False)
    return reader


class PanoReader:
    def __init__(self, dset, height, width=None):
        super().__init__()
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_file)
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
        self.template = {}
        for key in config[dset].keys():
            self.template[key] = base_path + config[dset][key]
        self.height = height
        if width is None:
            self.width = height * 2
        else:
            self.width = width


class ZInDReader(PanoReader):

    def __init__(self, height, width=None):
        super().__init__('zind', height, width)

    def get_arbitrary_image(self, record_id):
        im = cv2.imread(record_id)[:,:,[2,1,0]].astype(np.float32) / 255.0
        im = torch.from_numpy(im).permute(2,0,1)
        return im

    def get_rgb_image(self, record_id, key='rgb'):
        record = record_id.split(' ')
        split_id, image_id = record
        if key in self.template:
            im = cv2.imread(self.template[key].format(split=split_id, image=image_id))[:, :, [2,1,0]].astype(np.float32) / 255.0
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2,0,1)
        return im

    def get_gray_image(self, record_id, key):
        record = record_id.split(' ')
        split_id, image_id = record
        im = cv2.imread(self.template[key].format(split=split_id, image=image_id), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im)[None]
        return im

    def get_hdr_image(self, record_id, key='hdr'):
        record = record_id.split(' ')
        split_id, image_id = record
        im = cv2.imread(self.template[key].format(split=split_id, image=image_id), cv2.IMREAD_UNCHANGED)[:, :, [2,1,0]].astype(np.float32)
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2,0,1)
        return im

    def get_semantic_image(self, record_id, modal='semantic', key=None):
        # H x W
        if key is not None:
            im = cv2.imread(key)[:,:,[2,1,0]]
        else:
            record = record_id.split(' ')
            split_id, image_id = record[:4]
            im = cv2.imread(self.template[modal].format(split=split_id, image=image_id))[:, :, [2,1,0]]
        label = np.zeros_like(im[:,:,0])
        pallete = self._create_color_palette_room()
        for k, v in enumerate(pallete):
            mask = (im[:,:,0] == v[1][0]) * (im[:,:,1] == v[1][1]) * (im[:,:,2] == v[1][2])
            label[mask] = k
        if key is None:
            label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        label = torch.from_numpy(label)
        return label

    def _create_color_palette_room(self):
        return [
            ('none', (0, 0, 0)),
            ('floor', (152, 223, 138)),
            ('ceiling', (78, 71, 183)),
            ('wall', (174, 199, 232)),
            ('door', (214, 39, 40)),
            ('window', (197, 176, 213)),
            ('lamp', (96, 207, 209)),
            ('prop', (31, 119, 180)),
            ('struct', (255, 187, 120))]

    def get_sundir_vector(self, record_id, key='sundir'):
        record = record_id.split(' ')
        split_id, image_id = record[:4]
        f = open(self.template[key].format(split=split_id, image=image_id), 'r')
        lines = f.readlines()
        f.close()
        score, azimuth, elevation = lines[0].split(' ')
        score, azimuth, elevation = float(score), float(azimuth), float(elevation)
        vec = torch.Tensor([score, azimuth, elevation, math.cos(azimuth * math.pi/180), math.sin(azimuth * math.pi/180), -math.tan(elevation * math.pi/180)])
        return vec
