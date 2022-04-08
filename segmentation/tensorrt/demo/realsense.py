import copy

import numpy as np
import pylab as pl
import torch
import tqdm
from torch import nn

from forge.sobel import Sobel, apply_brightness_contrast
from segmentation.tensorrt.utils.inference import Infer

model = Infer('./segmentation/tensorrt/assets/seg_int8.engine')

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode, ToTensor, Pad
from utils.input import RealSense
import cv2


def main():
    camera = RealSense()
    sobel = Sobel()
    snap = Snap()

    tr = T.Compose([T.ToTensor(),
                    T.Resize((192, 256), InterpolationMode.BILINEAR),
                    T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                std=[0.229, 0.224, 0.225])])

    for _ in tqdm.tqdm(range(10000)):
    # while True:
        rgb, depth = camera.read()

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        mask = model(rgb)

        res = process(rgb, mask)

        categories = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)
        overlay = copy.deepcopy(rgb)
        overlay[categories == 0] = np.array([0, 0, 0])

        # # Sobel
        # contrast = apply_brightness_contrast(rgb, 0, 64)
        # gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # sr = cv2.Canny(blurred, 30, 150)
        # sr = sr[..., None]
        #
        # # sc = sobel(Pad(1, padding_mode='edge')(torch.tensor(categories)).unsqueeze(0).unsqueeze(0).to(torch.double))
        #
        # sc = cv2.Canny((categories * 255).astype(np.uint8), 30, 150)
        # sc = sc[..., None]
        # # sc = sc.numpy().squeeze(0).transpose(1, 2, 0)
        #
        # sr = np.tile(sr, [1, 1, 3])
        # sr[sr[..., 0] != 0] = np.array([1, 0, 0])
        #
        # sc = np.tile(sc, [1, 1, 3])
        # sc[sc[..., 0] != 0] = np.array([0, 1, 0])
        # # sb = np.zeros([, ])
        #
        # # End sobel
        # res3 = snap(torch.tensor(sc).permute(2, 0, 1)[1:2], torch.tensor(sr).permute(2, 0, 1)[0:1])
        # res3 = res3.numpy().squeeze(0).transpose(1, 2, 0)
        # res3 = np.tile(res3, [1, 1, 3])
        # res3[res3[..., 0] != 0] = np.array([0, 0, 1])
        #
        # cv2.imshow('Sobel categories', (sc * 255).astype(np.uint8))
        # cv2.imshow('Sobel real', (sr * 255).astype(np.uint8))
        # cv2.imshow('Sobel', cv2.addWeighted(cv2.addWeighted((sr * 255).astype(np.uint8), 1, (sc * 255).astype(np.uint8), 0.5, 0), 1, (res3 * 255).astype(np.uint8), 0.5, 0))
        # # cv2.imshow('Snap', res3.numpy().squeeze(0).transpose(1, 2, 0))

        cv2.imshow('Segmentation 1', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        cv2.imshow('Segmentation 2', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) == ord('q'):
            exit()


def process(frame, mask):
    # mask = torch.softmax(torch.tensor(mask), dim=1)
    # mask = torch.argmax(mask, dim=1).permute([1, 2, 0])

    categories = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

    overlay = copy.deepcopy(frame)
    if np.any(categories == 1):
        overlay[categories == 1] = np.array([0, 0, 128])
    res = cv2.addWeighted(frame, 1, overlay, 0.5, 0)

    return res

class Snap(nn.Module):
    def __init__(self):
        super().__init__()
        k_size = 3

        self.filter1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=k_size, stride=1, padding=k_size // 2, padding_mode='replicate', bias=False)

        Gx = torch.zeros([k_size, k_size]).to(torch.double)
        Gx[k_size // 2, k_size // 2] = 1
        Gy = -torch.ones([k_size, k_size]).to(torch.double)
        Gy[k_size // 2, k_size // 2] = 0
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(0)
        self.filter1.weight = nn.Parameter(G, requires_grad=False)

        self.filter2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=k_size, stride=1, padding=k_size // 2, padding_mode='replicate', bias=False)

        F = torch.ones([k_size, k_size]).to(torch.double)
        F = F.unsqueeze(0).unsqueeze(0)
        self.filter2.weight = nn.Parameter(F, requires_grad=False)

    def forward(self, mask, edge):

        img = torch.cat([mask, edge], dim=0).unsqueeze(0).to(torch.double)
        x1 = self.filter1(img).squeeze(0)
        x1[x1 < 0] = 0

        x2 = self.filter2(mask.unsqueeze(0).to(torch.double))
        x2 = x2 * edge

        res = x1 + x2
        res[res > 1] = 1

        return res



if __name__ == '__main__':
    main()
