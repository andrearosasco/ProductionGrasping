from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def unity_visualize():
    root = Path('data/unity')
    labels = root / 'SemanticSegmentation'
    rgbs = root / 'RGB'

    for f, l in zip(sorted(rgbs.glob('*.png'), key=lambda x: int(str(x.stem)[4:])),
                 sorted(labels.glob('*.png'), key=lambda x: int(str(x.stem)[13:]))):

        rgb = cv2.imread(str(f))

        label = cv2.imread(str(l))
        label[..., 0][label[..., 0] == 1] = 255

        res = cv2.addWeighted(rgb, 0.4, label, 0.3, 0)
        cv2.imshow('frame', res)
        cv2.waitKey(0)


if __name__ == '__main__':
    unity_visualize()