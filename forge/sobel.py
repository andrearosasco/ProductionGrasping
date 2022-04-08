import cv2
import numpy as np
from scipy.signal.windows import gaussian
from torch import nn
import torch
from torchvision.transforms import ToTensor, Pad
import pandas as pd
import matplotlib.pyplot as plt

from utils.input import RealSense


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def main():
    camera = RealSense()
    sobel = Sobel()

    while True:
        rgb, depth = camera.read()
        gs = ToTensor()(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).to(torch.double)
        gs = 0.299 * gs[0] + 0.587 * gs[1] + 0.114 * gs[2]
        gs = Pad(1, padding_mode='edge')(gs)

        # res = sobel(gs.unsqueeze(0) .unsqueeze(0))
        contrast = apply_brightness_contrast(rgb, 0, 64)
        gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        res = cv2.Canny(blurred, 30, 150)
        # res = res.numpy().squeeze(0).transpose(1, 2, 0)

        # df = pd.DataFrame(res.ravel())
        # df[df != 0].plot.hist()
        # plt.show()
        # res[res < 1] = 0

        cv2.imshow('Sobel', res)
        cv2.waitKey(1)





class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]]).to(torch.double)
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]]).to(torch.double)
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


if __name__ == '__main__':
    main()