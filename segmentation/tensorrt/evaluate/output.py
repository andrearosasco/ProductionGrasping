import copy

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from polygraphy.common import TensorMetadata
from polygraphy.comparator import DataLoader
from torch import nn
from torchvision import models

from segmentation.tensorrt.utils.dataset import DataSet
from segmentation.tensorrt.utils.inference import Infer


def main(runtime):

    data_loader = DataSet(iterations=1000)

    if runtime == 'trt':
        ####  Setup TensorRT Engine
        backbone = Infer('./segmentation/tensorrt/assets/seg.engine')
        ####  Run Evaluation
        for i, x in tqdm.tqdm(enumerate(data_loader)):
            res = torch.softmax(torch.tensor(backbone(x['input'])), dim=1)
            res = torch.argmax(res, dim=1).permute([1, 2, 0])

            categories = cv2.resize(res.numpy(), dsize=(640, 480), interpolation=cv2.INTER_NEAREST)
            frame = np.array(Image.open(f'./segmentation/tensorrt/assets/real_data/eCub{i}_rgb.png'))
            overlay = copy.deepcopy(frame)
            if np.any(categories == 1):
                overlay[categories == 1] = np.array([0, 0, 128])
            res = cv2.addWeighted(frame, 1, overlay, 0.5, 0)
            # Image.fromarray(res.numpy().astype(np.uint8)).save(f'./segmentation/tensorrt/assets/segmentations/trt_mask{i}.png')
            cv2.imshow('test', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

    if runtime == 'pt':
        model = models.segmentation.fcn_resnet101(pretrained=False)
        model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.load_state_dict(torch.load('./segmentation/pytorch/checkpoints/sym5/epoch23'), strict=False)
        model.eval()
        model.cuda()

        with torch.no_grad():
            for i, x in tqdm.tqdm(enumerate(data_loader)):
                res = torch.softmax(model(torch.tensor(x['input'], device='cuda')), dim=1).squeeze()
                res = torch.argmax(res, dim=0)

                Image.fromarray(res).save(f'./segmentation/tensorrt/assets/segmentations/pt_mask{i}.png')

if __name__ == '__main__':
    main('trt')