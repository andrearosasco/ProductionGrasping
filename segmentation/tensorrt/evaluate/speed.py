import numpy as np
import torch
import tqdm
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
        backbone = Infer('./segmentation/tensorrt/assets/seg_test.engine')
        ####  Run Evaluation
        for i, x in tqdm.tqdm(enumerate(data_loader)):
            res = backbone(x['input'])

    if runtime == 'pt':
        model = models.segmentation.fcn_resnet101(pretrained=False)
        model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model.load_state_dict(torch.load('./segmentation/pytorch/checkpoints/sym5/epoch23'), strict=False)
        model.eval()
        model.cuda()

        with torch.no_grad():
            for i, x in tqdm.tqdm(enumerate(data_loader)):
                res = model(torch.tensor(x['input'], device='cuda'))

if __name__ == '__main__':
    main('trt')