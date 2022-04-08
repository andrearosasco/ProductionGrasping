import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Precision, Recall, F1Score
from torchvision import models
from torchvision.transforms import InterpolationMode

from utils.data.testset import TestSet
from utils.data.splitset import SplitDataset
import torchvision.transforms as T

import utils.model.wrappers

@torch.no_grad()
def valid():
    model = models.segmentation.fcn_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)) # checkpoints/seg_model_f10.4585658013820648 works well
    model.load_state_dict(torch.load('checkpoints/sym4/sgdCE'), strict=False)
    model.eval()
    model.cuda()

    model1 = models.segmentation.fcn_resnet101(pretrained=False)
    model1.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))  # checkpoints/seg_model_f10.4585658013820648 works well
    model1.load_state_dict(torch.load('checkpoints/seg_model_f10.4585658013820648'), strict=False)
    model1.eval()
    model1.cuda()

    real_validset = SplitDataset(splits='./data/real/splits', mode='eval',
                                 transform=T.Compose([T.ToTensor(),
                                                      # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                                      T.Resize((192, 256), InterpolationMode.BILINEAR),
                                                      T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                                                  std=[0.229, 0.224, 0.225])]),  # 0.229, 0.224, 0.225
                                 target_transform=T.Compose([lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0),
                                                             # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                                             T.Resize((192, 256), InterpolationMode.NEAREST)
                                                             ]))

    real_validloader = DataLoader(real_validset, batch_size=1, shuffle=False)

    "With reduction none the mean is performed across examples but not across classes" \
    "If a class is not present in an example, and it's not predicted IoU is 0 but it's not added to the mean of thath class" \
    "If it's not present but it is predicted we will see a decrease of jaccard on the other class"

    "To have a more precise measure of the instances in which it's incorrectly predicted we should measure the precision as well"
    metrics = [JaccardIndex(num_classes=2, threshold=0.5, reduction='none').to('cuda'),
               Precision(threshold=0.5, multiclass=False).to('cuda'),
               Recall(threshold=0.5, multiclass=False).to('cuda'),
               F1Score(threshold=0.5, multiclass=False).to('cuda')]

    for i, (img_batch, lbl_batch) in enumerate(real_validloader):
        img_batch, lbl_batch = img_batch.cuda(), lbl_batch.cuda()
        logits = model(img_batch)['out']

        logits1 = model1(img_batch)['out']

        # for mtr in metrics:
        #     mtr(torch.argmax(logits, dim=1), lbl_batch)

        segmented, classes = utils.model.wrappers.Segmentator.postprocess(logits[0])

        segmented1, classes1 = utils.model.wrappers.Segmentator.postprocess(logits1[0])

        tr = T.Compose([
            lambda x: x.div(1 / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(x.device)),
            lambda x: x.sub(-torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(x.device)),
            T.ToPILImage(),
            T.Resize((480, 640), InterpolationMode.BILINEAR),
            lambda x: np.array(x)])

        overlay = copy.deepcopy(cv2.cvtColor(tr(img_batch[0]), cv2.COLOR_RGB2BGR))
        if np.any(classes == 1):
            overlay[classes == 1] = np.array([128, 0, 0])
        res = cv2.addWeighted(cv2.cvtColor(tr(img_batch[0]), cv2.COLOR_RGB2BGR), 1, overlay, 0.5, 0)
        cv2.imshow('segmented', res)

        overlay1 = copy.deepcopy(cv2.cvtColor(tr(img_batch[0]), cv2.COLOR_RGB2BGR))
        if np.any(classes1 == 1):
            overlay1[classes1 == 1] = np.array([128, 0, 0])
        res1 = cv2.addWeighted(cv2.cvtColor(tr(img_batch[0]), cv2.COLOR_RGB2BGR), 1, overlay1, 0.5, 0)
        cv2.imshow('segmented1', res1)

        cv2.imshow('original', cv2.cvtColor(tr(img_batch[0]), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)

    for m in metrics:
        print(m.compute())



if __name__ == '__main__':
    valid()
