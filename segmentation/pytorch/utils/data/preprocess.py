import json
from functools import reduce
from pathlib import Path

import torch
from PIL import Image
import cv2
import scipy.io as scio
import numpy as np
import os
import sys

box_cls = {"003_cracker_box": 2,
           "004_sugar_box": 3,
           "008_pudding_box": 7,
           "009_gelatin_box": 8,
           "036_wood_block": 16,
           "061_foam_brick": 21}


def ycb_preprocess():
    root = Path('../DenseFusion/datasets/ycb')
    data = root / 'YCB_Video_Dataset' / 'data'

    for video in data.glob('*'):
        for frame in video.glob('*'):
            if frame.suffix == '.mat':
                meta = scio.loadmat(frame)
                objs_cls = meta['cls_indexes'].flatten().astype(np.int32)
                if not set(objs_cls).isdisjoint(box_cls.values()):
                    idx = int(frame.stem[:-5])
                    label = cv2.imread(f'{frame.parent}/{idx:06}-label.png')
                    color = cv2.imread(f'{frame.parent}/{idx:06}-color.png')

                    label[~np.isin(label, list(box_cls.values()))] = 0


def unity_splits():
    root = Path('./data/unity/sym5/data')
    examples = []

    for subdir in root.glob('*/'):
        if not subdir.is_dir(): continue
        rgbs = list(subdir.glob('RGB*/'))[0]
        labels = list(subdir.glob('SemanticSegmentation*/'))[0]

        for x, y in zip(sorted(rgbs.glob('*.png'), key=lambda x: int(str(x.stem)[4:])),
                        sorted(labels.glob('*.png'), key=lambda x: int(str(x.stem)[13:]))):
            x = reduce(lambda a, b: a / b, x.parts[-3:], Path())
            y = reduce(lambda a, b: a / b, y.parts[-3:], Path())
            examples.append((x, y))

    train = examples[:int(len(examples) * 0.8)]
    eval = examples[int(len(examples) * 0.8):]

    out_dir = (root.parent / 'splits')
    out_dir.mkdir(exist_ok=True)
    for split, file in zip((train, eval), ('train.txt', 'eval.txt')):
        fp = (out_dir / file).open('w+')
        print(f'sym3 {str(root)}', file=fp)
        for x, y in split:
            print(f'sym3 {str(x)} {str(y)}', file=fp)
        fp.close()


def scrape_dae(root=Path('../../Downloads/ADE20K')):
    data = root / 'ADE20K_2021_17_01/images'

    i = 0

    frames = []
    n_frames = {}

    old_fn = None
    for fn in data.rglob('*.json'):  # utility box
        data = json.load(fn.open('r'))
        data = data['annotation']
        for obj in data[
            'object']:  # street box, file box, bread box, tissue box, tools box, ceramic box, plant box, cigar box, storage box,
            # if 'box' in l and not ('television' in l or 'refrigerator' in l or
            #                         'letter' in l or 'office' in l or 'squeeze' in l or
            #                         'juke' in l or 'street' in l or 'post' in l or
            #                         'telephone' in l or 'plant' in l or 'electricity' in l or
            #                         'power' in l or 'breaker' in l):
            if "box" == obj['raw_name']:
                if fn != old_fn:
                    old_fn = fn
                    frames += [(root / data['folder'], data['filename'], obj['instance_mask'])]
                n_frames[fn] = n_frames[fn] + 1 if fn in n_frames.keys() else 1
                # print(obj['name'])
            # if ' # 0 # 0 # box # box # \"\"' in obj: # or ' # 0 # 0 # box # boxes # \"\"' in l[3:]
            #     frames += [str(fn)]
            #     n_frames[fn] = n_frames[fn] + 1 if fn in n_frames.keys() else 1

    for i, [k, v] in enumerate(n_frames.items()):
        if v == 1:
            data = json.load(k.open('r'))
            data = data['annotation']

            base = frames[i][0]
            rgb = frames[i][1]
            seg = frames[i][2]

            frame = cv2.imread(str(base / rgb))
            # segmented = cv2.imread(str(k.parent / f'{k.stem[:-4]}_seg.png'))

            class_mask = []
            instance_mask = []
            with Image.open(base / seg) as io:
                seg = np.array(io)

                # Obtain the segmentation mask, bult from the RGB channels of the _seg file
            seg = seg.astype(bool)

            seg_img = np.zeros([*seg.shape, 3])
            seg_img[seg] = np.array([255, 0, 0])
            seg_img = seg_img.astype(np.uint8)

            # Obtain the instance mask from the blue channel of the _seg file
            # Minstances_hat = np.unique(B, return_inverse=True)[1]
            # Minstances_hat = np.reshape(Minstances_hat, B.shape)
            # instance_mask = Minstances_hat

            res = cv2.addWeighted(frame, 0.4, seg_img, 0.3, 0)

            f = min([1280 / res.shape[1], 720 / res.shape[0]])

            print(str(base / rgb))
            cv2.imshow('frame', cv2.resize(res, (round(res.shape[1] * f), round(res.shape[0] * f))))
            cv2.waitKey(0)
            pass


def unity_preprocess(path='./data/unity/sym5/data'):
    root = Path(path)
    i = 0
    import socket
    tot = 0
    for subdir in root.glob('*/'):
        if not subdir.is_dir(): continue
        rgbs = list(subdir.glob('RGB*/'))[0]
        labels = list(subdir.glob('SemanticSegmentation*/'))[0]

        for f, l in zip(sorted(rgbs.glob('*.png'), key=lambda x: int(str(x.stem)[4:])),
                        sorted(labels.glob('*.png'), key=lambda x: int(str(x.stem)[13:]))):

            rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            label = cv2.imread(str(l))

            if np.any(label != 0):
                i += 1
            tot += 1
            label[label != 0] = 1
            label = label[..., 0]

            Image.fromarray(label).save(f'{l.parent}/segmentation_{int(str(l.stem)[13:]):06}.png')
            Image.fromarray(rgb).save(f'{f.parent}/rgb_{int(str(f.stem)[4:]):06}.png')

            l.unlink()
            f.unlink()
    print(i)
    print(tot)

def real_preprocess(path='data/real/data/mask'):
    root = Path(path)
    i = 0

    tot = 0
    for file in root.glob('*.png'):

        label = cv2.imread(str(file))

        if np.any(label != 0):
            i += 1
        tot += 1
        label[label != 255] = 1
        label[label == 255] = 0
        label = label[..., 0]

        Image.fromarray(label).save(f'{file.parent}/{file.name.replace("rgb", "")}')

        file.unlink()
    print(i)
    print(tot)

def real_splits():
    root = Path('./data/real/data')
    examples = []

    for rgb, mask in zip((root / 'rgb').glob('*'), (root / 'mask').glob('*')):

        x = reduce(lambda a, b: a / b, rgb.parts[-2:], Path())
        y = reduce(lambda a, b: a / b, mask.parts[-2:], Path())

        examples.append((x, y))

    train = examples[:int(len(examples) * 0.0)]
    eval = examples[int(len(examples) * 0.0):]

    out_dir = (root.parent / 'splits')
    out_dir.mkdir(exist_ok=True)
    for split, file in zip((train, eval), ('train.txt', 'eval.txt')):
        fp = (out_dir / file).open('w+')
        print(f'real {str(root)}', file=fp)
        for x, y in split:
            print(f'real {str(x)} {str(y)}', file=fp)
        fp.close()


if __name__ == '__main__':
    unity_splits()

