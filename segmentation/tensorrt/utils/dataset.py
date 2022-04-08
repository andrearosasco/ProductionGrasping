from pathlib import Path

import tensorrt as trt

import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class DataSet:

    def __init__(self, iterations):
        self.iterations = iterations
        self.root = Path('./segmentation/tensorrt/assets/real_data')

        tr = T.Compose([T.ToTensor(),
                        T.Resize((192, 256), InterpolationMode.BILINEAR),
                        T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                    std=[0.229, 0.224, 0.225])])

        self.items = []
        for i in range(22):
            # self.items.append(tr(np.array(Image.open(self.root / f'ecub{i}_rgb.png'))).numpy()[None, ...])
            self.items.append(np.array(Image.open(self.root / f'ecub{i}_rgb.png')))

    def __getitem__(self, i):
        if i >= self.iterations:
            raise StopIteration

        i = i % 22
        return {'input': self.items[i].astype(trt.nptype(trt.float32)).ravel()}

    def __len__(self):
        return self.iterations
