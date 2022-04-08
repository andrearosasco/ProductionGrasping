import random
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

import torchvision.transforms as T

box_cls = {"003_cracker_box": 2,
           "004_sugar_box": 3,
           "008_pudding_box": 7,
           "009_gelatin_box": 8,
           "036_wood_block": 16,
           "061_foam_brick": 21}


def scrape_ycb(root=Path('../DenseFusion/datasets/ycb/YCB_Video_Dataset'),
               splits={'train': 0.0, 'valid': 0.0, 'test': 1.0},
               fps_start=30, fps_end=1, positive=True,
               targets=["003_cracker_box", "004_sugar_box", "008_pudding_box",
                        "009_gelatin_box", "036_wood_block", "061_foam_brick"]):
    # fps_start: frame rate of the video
    # fps_end: desired number of examples per second e.g. 1/3 is a frame every 3 seconds
    # positive: if True returns example with the target object inside

    data = root / 'data'

    frames = []
    labels = []
    for video in data.glob('*/'):
        i = 0
        for fn in video.glob('*.txt'):
            objs = [line.split()[0] for line in fn.open('r').readlines()]
            if not (positive ^ (not set(objs).isdisjoint(set(targets)))):  # xnor to invert the condition on positive
                if not (positive ^ (len(set(objs).intersection(set(targets))) == 1)):
                    if i % (fps_start / fps_end) == 0:
                        frames += [Path(fn.parent.parts[-2]) / fn.parent.parts[-1] / f'{fn.stem[:-4]}-color.png']
                        labels += [Path(fn.parent.parts[-2]) / fn.parent.parts[-1] / f'{fn.stem[:-4]}-label.png']
                    i += 1

    no_train = round(len(frames) * splits['train'])
    no_valid = round(len(frames) * splits['valid'])

    ret = {'train': [[], []], 'valid': [[], []], 'test': [[], []]}
    for i, [frame, label] in enumerate(zip(frames, labels)):
        if i < no_train:
            split = 'train'
        elif no_train <= i < no_train + no_valid:
            split = 'valid'
        else:
            split = 'test'

        ret[split][0] += [frame]
        ret[split][1] += [label]
    return ret

import scipy.io as scio




def create_ycb_split():
    ycb_root = Path('../DenseFusion/datasets/ycb/YCB_Video_Dataset')
    ycb_positive_plits = scrape_ycb(root=ycb_root)
    ycb_negative_plits = scrape_ycb(root=ycb_root, positive=False)

    for split in ['train', 'valid', 'test']:
        with open(f'{split}.txt', 'w+') as out:

            print(f'path ycb {ycb_root}', file=out)

            for frame, label in zip(*ycb_positive_plits[split]):
                print(f'ycb {frame} {label}', file=out)

            for frame, label in zip(*ycb_negative_plits[split]):
                print(f'ycb {frame} {label}', file=out)




def create_dae_split():
    scrape_dae()


class TestSet(Dataset):

    def __init__(self, splits=Path('./data'), paths=None, transform=T.ToTensor(),
                 target_transform=lambda x: torch.tensor(np.array(x))):

        self.paths = {}
        self.examples = []

        self.transform = transform
        self.target_transform =target_transform

        with (splits / 'test.txt').open('r') as file:
            for line in file.readlines():
                if line.split()[0] == 'path':
                    self.paths[line.split()[1]] = line.split()[2]
                    continue

                self.examples += [line]

        if paths is not None:
            for k, v in paths.items():
                self.paths[k] = v

    def __getitem__(self, idx):

        line = self.examples[idx]

        if line.split()[0] == 'ycb':

            image = Image.open(Path(self.paths[line.split()[0]]) / line.split()[1])
            label = Image.open(Path(self.paths[line.split()[0]]) / line.split()[2])

            image = np.array(image)
            label = np.array(label)

            image = self.transform(image)
            label = self.target_transform(label)

            label[~torch.isin(label, torch.tensor(list(box_cls.values())))] = 0
            label[torch.isin(label, torch.tensor(list(box_cls.values())))] = 1


            return image, label



    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    data = TestSet()
    for f, l in data:
        pass
    # create_ycb_split()
