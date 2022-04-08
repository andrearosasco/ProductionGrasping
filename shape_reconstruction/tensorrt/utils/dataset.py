from pathlib import Path

import numpy as np
from PIL import Image


class DataSet:

    def __init__(self, iterations):
        self.iterations = iterations
        self.root = Path('./shape_reconstruction/tensorrt/assets/real_data')

        self.items = []
        for i in range(22):
            self.items.append(np.load(str(self.root / f'partial{i}.npy'))[None, ...])

    def __getitem__(self, i):
        if i >= self.iterations:
            raise StopIteration

        i = i % 22
        return {'input': self.items[i]}

    def __len__(self):
        return self.iterations
