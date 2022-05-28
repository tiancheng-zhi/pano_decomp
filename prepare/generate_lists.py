import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', type=str, default='../data/zind/raw')
    parser.add_argument('--out-path', type=str, default='../data/zind/scenes')
    parser.add_argument('--list-path', type=str, default='../lists')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_args()
    raw_path = Path(opt.raw_path)
    out_path = Path(opt.out_path)
    list_path = Path(opt.list_path)
    splits = ['train', 'val', 'test']
    house_split = json.load(open(raw_path / 'zind_partition.json'))
    house_split['all'] = sorted(house_split['train'] + house_split['val'] + house_split['test'])
    for split_id in splits + ['all']:
        print('\n'.join(house_split[split_id]), file=open(list_path / f'zind_houses_{split_id}.txt', 'w'))

    pano_split = {'train':[], 'val':[], 'test':[]}
    for split_id in splits:
        for house_id in house_split[split_id]:
            house_data = json.load(open(raw_path / house_id / 'zind_data.json'))
            floors = sorted(house_data['merger'].keys())
            for floor_id in floors:
                complete_rooms = house_data['merger'][floor_id].keys()
                for complete_room_id in complete_rooms:
                    partial_rooms = house_data['merger'][floor_id][complete_room_id].keys()
                    for partial_room_id in partial_rooms:
                        panos = sorted(house_data['merger'][floor_id][complete_room_id][partial_room_id].keys())
                        for pano_id in panos:
                            house = house_id
                            floor = floor_id[len('floor_'):]
                            room = partial_room_id[len('partial_room_'):]
                            pano = pano_id[len('pano_'):]
                            pano_split[split_id].append(f'{split_id} {house}_{floor}_{room}_{pano}')
    pano_split['all'] = sorted(pano_split['train'] + pano_split['val'] + pano_split['test'])
    for split_id in splits + ['all']:
        print('\n'.join(pano_split[split_id]), file=open(list_path / f'zind_panos_{split_id}.txt', 'w'))
