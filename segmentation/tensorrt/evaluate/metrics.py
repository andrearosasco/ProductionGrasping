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
from torchmetrics import JaccardIndex, Precision, F1Score, Recall, MeanMetric

from segmentation.pytorch.config.server_config import Config
from segmentation.pytorch.utils.data.splitset import SplitDataset
from segmentation.tensorrt.utils.dataset import DataSet
from segmentation.tensorrt.utils.inference import Infer

import torchvision.transforms as T


def eval_trt(engine):
    data_loader = SplitDataset(splits='../Segmentation/data/real/splits', mode='eval')

    metrics = {
        'jaccard': JaccardIndex(num_classes=2),  # TODO , reduction='none'
        'precision': Precision(multiclass=False, average='samples'),
        'recall': Recall(multiclass=False, average='samples'),
        'f1score': F1Score(multiclass=False, average='samples')}

    ####  Setup TensorRT Engine
    model = Infer(engine)

    ####  Run Evaluation
    precision, recall = 0, 0
    for i, data in tqdm.tqdm(enumerate(data_loader)):
        x, y = data

        res = model(x)
        res = cv2.resize(res, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

        print([f'Instant {m}: {metrics[m](torch.tensor(res).unsqueeze(0), torch.tensor(y).unsqueeze(0))}'
               for m in metrics])

        # for categories in [categories, y]:
        #     overlay = copy.deepcopy(x)
        #     if np.any(categories == 1):
        #         overlay[categories == 1] = np.array([0, 0, 128])
        #     res = cv2.addWeighted(x, 1, overlay, 0.5, 0)
        #
        #     cv2.imshow('test', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)
    print(*[f'Average {m}: {metrics[m].compute()}' for m in metrics], sep='\n')

def eval_pt():
    data_loader = SplitDataset(splits='../Segmentation/data/real/splits', mode='eval',
                                   transform=T.Compose([T.ToTensor(),
                                                        T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                                                    std=[0.229, 0.224, 0.225])]),  # 0.229, 0.224, 0.225
                                   target_transform=T.Compose([lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0),
                                                               ]))

    metrics = {
        'jaccard': JaccardIndex(num_classes=2),  # TODO , reduction='none'
        'precision': Precision(multiclass=False, average='samples'),
        'recall': Recall(multiclass=False, average='samples'),
        'f1score': F1Score(multiclass=False, average='samples')}

    model = models.segmentation.fcn_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load('./segmentation/pytorch/checkpoints/sym5/epoch23'), strict=False)
    model.eval()

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            x, y = data
            res = model(x.unsqueeze(0))['out']
            res = torch.softmax(res, dim=1)
            res = torch.argmax(res, dim=1).permute([1, 2, 0])
            res = cv2.resize(res.numpy(), dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

            print([f'Instant {m}: {metrics[m](torch.tensor(res).unsqueeze(0), torch.tensor(y).unsqueeze(0))}'
                   for m in metrics])

        print(*[f'Average {m}: {metrics[m].compute()}' for m in metrics], sep='\n')

if __name__ == '__main__':
    # eval_trt('./segmentation/tensorrt/assets/seg_int8.engine')
    eval_pt()
