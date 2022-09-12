import os
import sys
from skimage import morphology
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import sys
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import json
from imageio import imread, imwrite
from tqdm import tqdm
import pathlib
import LED2Net
import cv2
import PIL

floor = (152, 223, 138)
tripod = (0,0,0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for LED^2-Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config/config_zind.yaml', help='config.yaml path')
    parser.add_argument('--src', type=str,
                        default='data/zind/scenes/rgb',
                        help='The folder that contain *.png or *.jpg')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--src_semantic', type=str,
                        default='data/zind/scenes/semantic',
                        help='The folder that contain *.png or *.jpg')
    parser.add_argument('--dst', type=str, default='data/zind/scenes/layout', help='The folder to save the output')
    parser.add_argument('--is_merge', action='store_false', help='The folder to save the output')
    parser.add_argument('--ckpt', type=str,
                        default='./model_2021-12-10-17-38-55_00029.pkl',
                        help='Your pretrained model location (xxx.pkl)')
    args = parser.parse_args()

    is_merge = args.is_merge

    args.src = f'{args.src}/{args.split}'
    args.src_semantic = f'{args.src_semantic}/{args.split}'
    if not is_merge:
        args.dst = f'{args.dst}/{args.split}'
    else:
        args.dst = f'{args.dst}_merge/{args.split}'


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = config['exp_args']['device']
    equi_shape = config['exp_args']['visualizer_args']['equi_shape']
    model = LED2Net.Network(**config['network_args']).to(device)
    params = torch.load(args.ckpt)
    model.load_state_dict(params)
    model.eval()

    tmp = [torch.FloatTensor(x).to(device)[None, ...]
           for x in LED2Net.Dataset.SharedFunctions.create_grid(equi_shape)
           ]
    _, unit_lonlat, unit_xyz = tmp
    infer_height = LED2Net.PostProcessing.InferHeight()
    visualizer = LED2Net.LayoutVisualizer(**config['exp_args']['visualizer_args'])


    root_dir = os.path.dirname(os.path.realpath(__file__)) + '/..'

    src = root_dir + '/' + args.src
    src_semantic = root_dir + '/' + args.src_semantic
    dst = root_dir + '/' + args.dst
    lst = sorted(glob.glob(src + '/*.png') + glob.glob(src + '/*.jpg'))
    lst_semantic = sorted(glob.glob(src_semantic + '/*.png') + glob.glob(src_semantic + '/*.jpg'))

    assert len(lst) == len(lst_semantic)
    for i in tqdm(range(len(lst))):
        one = lst[i]
        name = one.split('/')[-1]
        sem_name = name.replace('.png', '_semantic.png')
        sem_name = lst_semantic[i].split('/')[-1]
        one_semantic = f'{src_semantic}/{sem_name}'

        img = LED2Net.Dataset.SharedFunctions.read_image(one, equi_shape)
        semantic = LED2Net.Dataset.SharedFunctions.read_image(one_semantic, equi_shape)
        # map back to 255
        semantic = np.asarray(np.uint8(semantic * 255))
        # semantic = semantic + (semantic == np.array(tripod)[None, None]) * np.array(floor)[None, None]

        # extract floor mask
        semantic = semantic == np.array(floor)[None, None]
        # img = img * (1-semantic)
        # plt.imshow(img)
        # plt.show()
        semantic = semantic.astype(np.float)[..., 0]

        # extract wall floor boundary
        semantic_ind = np.argmax(semantic!=0, axis=0)

        batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].to(device)
        with torch.no_grad():
            pred = model(batch)

        pred_lonlat_up = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 0, :, None]], dim=-1)
        pred_lonlat_down = torch.cat([unit_lonlat[:, :, 0:1], pred[:, 1, :, None]], dim=-1)
        pred_ratio = infer_height(pred_lonlat_up, pred_lonlat_down)
        pred_corner_num = torch.zeros(pred.shape[0]).to(device).long() + pred.shape[2]
        pred_xyz_down = LED2Net.Conversion.lonlat2xyz(pred_lonlat_down, mode='torch')
        scale = config['exp_args']['camera_height'] / pred_xyz_down[..., 1:2]
        pred_xyz_down *= scale
        # plot semantic down
        semantic_pos_down = np.where(semantic == 1)
        semantic_pos_down = np.array(semantic_pos_down).astype(np.float)
        semantic_pos_down[0] = semantic_pos_down[0] / (equi_shape[0]-1)
        semantic_pos_down[0] =  semantic_pos_down[0] * np.pi - np.pi / 2
        semantic_pos_down[1] = semantic_pos_down[1] / (equi_shape[1]-1)
        semantic_pos_down[1] =  semantic_pos_down[1] * 2 * np.pi - np.pi
        semantic_lonlat_down = np.stack([semantic_pos_down[1], semantic_pos_down[0]], axis=-1)[None]
        semantic_lonlat_down = torch.Tensor(semantic_lonlat_down)


        semantic_xyz_down = LED2Net.Conversion.lonlat2xyz(semantic_lonlat_down, mode='torch')
        scale = config['exp_args']['camera_height'] / semantic_xyz_down[..., 1:2]
        semantic_xyz_down *= scale
        semantic_corner_num = torch.Tensor([semantic_xyz_down.shape[1]]).type(torch.int64)

        # this is to adjust meters image by image to avoid out of index
        offset = 0
        for m in range(10):
            offset = offset + m * 5
            try:
                semantic_fp_down = visualizer.plot_fp(semantic_xyz_down, semantic_corner_num, mode='fillDiscrete', offset=offset)[0, 0, ...].data.cpu().numpy()
                pred_fp_down = visualizer.plot_fp(pred_xyz_down, pred_corner_num, offset=offset)[0, 0, ...].data.cpu().numpy()
                break
            except:
                continue
        fp_down_diff = semantic_fp_down - semantic_fp_down * pred_fp_down
        fp_down_diff = morphology.remove_small_objects(fp_down_diff.astype(bool), min_size=500, connectivity=1).astype(int)
        pred_fp_down = pred_fp_down + fp_down_diff
        # plt.imshow(pred_fp_down)
        # plt.show()
        try:
            pred_fp_down_man, pred_fp_down_man_pts = LED2Net.DuLaPost.fit_layout(pred_fp_down, is_merge)
        except:
            print(one)
            continue
        # plt.imshow(pred_fp_down_man)
        # plt.show()
        # pred_fp_down_man_pts = pred_xyz_down[0].permute(1, 0)[::2, :].cpu().numpy()
        # print(pred_fp_down_man_pts[0].max())
        # print(pred_fp_down_man_pts[1].max())

        ratio = pred_ratio[0].data.cpu().numpy()
        pred_height = (ratio + 1) * config['exp_args']['camera_height']
        json_data = LED2Net.XY2json(
            pred_fp_down_man_pts.T[:, ::-1],
            y=config['exp_args']['camera_height'],
            h=pred_height,
            dim=config['exp_args']['visualizer_args']['fp_dim'],
            meters=config['exp_args']['visualizer_args']['fp_meters'] + offset,
        )

        dst_dir = dst + '/%s' % (one.split('/')[-1])
        dst_dir = dst_dir.replace('.png', '')
        dst_dir = dst_dir.replace('.jpg', '')
        pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)

        imwrite(dst_dir + '/color.jpg', (img * 255).astype(np.uint8))
        # imwrite(dst_dir+'/color_align.jpg', (img_align[..., ::-1]*255).astype(np.uint8))
        with open(dst_dir + '/pred.json', 'w') as f:
            f.write(json.dumps(json_data, indent=4) + '\n')
