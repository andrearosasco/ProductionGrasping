import random

import cv2

import torch
import tqdm
import wandb
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, Precision, F1Score, Recall, MeanMetric
from torchvision.transforms import InterpolationMode

from utils.framework.config import BaseConfig as Config

from utils.data.splitset import SplitDataset
from torchvision import models

import utils.model.wrappers
from utils.misc.reproducibility import make_reproducible


def main():
    make_reproducible(1)
    epoch = Config.Train.epoch

    model = models.segmentation.fcn_resnet101(pretrained=True).train()
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.to(Config.General.device)

    if Config.Eval.wandb:
        wandb.init(project='segmentation', config=Config.to_dict())
        wandb.watch(model, log='all')

    #
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([0.5, 0.5]).to(
            Config.General.device))  # TODO change weights weight=torch.tensor([0.3, 0.7]).to(Config.General.device)

    # criterion = DiceLoss()

    optim = SGD(params=model.parameters(), lr=Config.Train.lr, momentum=Config.Train.momentum)

    # train_set = SplitDataset(splits='./data/unity/sym3/splits', mode='train',
    #                      transform=T.Compose([T.Resize((256, 192), InterpolationMode.BILINEAR),
    #                                           T.ToTensor(),
    #                                           T.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                       std=[0.229, 0.224, 0.225])]),
    #                      target_transform=T.Compose([T.Resize((192, 256), InterpolationMode.NEAREST),
    #                                                  lambda x: torch.tensor(np.array(x), dtype=torch.long)]))

    synth_trainset = SplitDataset(splits='./data/unity/sym5/splits', mode='train',
                         transform=T.Compose([T.ToTensor(),
                                              # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                              # T.Resize((192, 256), InterpolationMode.BILINEAR),
                                              T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                                          std=[0.229, 0.224, 0.225])]),  # 0.229, 0.224, 0.225
                         target_transform=T.Compose([lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0),
                                                     # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                                     # T.Resize((192, 256), InterpolationMode.NEAREST)
                                                     ]))

    synth_validset = SplitDataset(splits='./data/unity/sym5/splits', mode='eval',
                               transform=T.Compose([T.ToTensor(),
                                                    # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                                    # T.Resize((192, 256), InterpolationMode.BILINEAR),
                                                    T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                                                std=[0.229, 0.224, 0.225])]),  # 0.229, 0.224, 0.225
                               target_transform=T.Compose([lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0),
                                                           # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                                           # T.Resize((192, 256), InterpolationMode.NEAREST)
                                                           ]))

    synth_trainloader = DataLoader(synth_trainset, batch_size=Config.Data.Train.mb_size, shuffle=True,
                              num_workers=Config.Data.num_worker, drop_last=True)

    # valid_set_real = TestSet(splits=Path('./data'), paths=Config.Data.Eval.paths,
    #                         transform=T.Compose([T.ToTensor(),
    #                                              # T.Pad([0, 80], fill=0, padding_mode='constant'),
    #                                              T.Resize((192, 256), InterpolationMode.BILINEAR),
    #                                              T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
    #                                                          std=[0.229, 0.224, 0.225])]),
    #                         target_transform=T.Compose([lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0),
    #                                                     # T.Pad([0, 80], fill=0, padding_mode='constant'),
    #                                                     T.Resize((192, 256), InterpolationMode.NEAREST)
    #                                                     ]))

    real_validset = SplitDataset(splits='./data/real/splits', mode='eval',
                                   transform=T.Compose([T.ToTensor(),
                                                        # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                                        # T.Resize((192, 256), InterpolationMode.BILINEAR),
                                                        T.Normalize(mean=[0.485, 0.456, 0.406],  # 0.485, 0.456, 0.406
                                                                    std=[0.229, 0.224, 0.225])]),  # 0.229, 0.224, 0.225
                                   target_transform=T.Compose([lambda x: torch.tensor(x, dtype=torch.long).unsqueeze(0),
                                                               # T.Pad([0, 80], fill=0, padding_mode='constant'),
                                                               # T.Resize((192, 256), InterpolationMode.NEAREST)
                                                               ]))

    real_validloader = DataLoader(real_validset, batch_size=Config.Data.Eval.mb_size, shuffle=False,
                                  num_workers=Config.Data.num_worker)

    synth_validloader = DataLoader(synth_validset, batch_size=Config.Data.Eval.mb_size, shuffle=False,
                                    num_workers=Config.Data.num_worker)

    metrics = {
        'jaccard': JaccardIndex(num_classes=2, threshold=0.5).to(Config.General.device),  # TODO , reduction='none'
        'precision': Precision(threshold=0.5, multiclass=False).to(Config.General.device),
        'recall': Recall(threshold=0.5, multiclass=False).to(Config.General.device),
        'f1score': F1Score(threshold=0.5, multiclass=False).to(Config.General.device)}
    avg_loss = MeanMetric().to(Config.General.device)

    global_step = 0
    best_score = 0
    for e in range(epoch):
        print('Starting epoch ', e)

        with torch.no_grad():
            fixed_img, fixed_gt, sample_img, sample_gt = {}, {}, {}, {}
            print('Validating...')
            model.eval()
            for valid_loader, s in [[real_validloader, 'ycb'], [synth_validloader, 'unity']]:
                sample = random.randint(0, len(valid_loader) - 1)
                for i, (img_batch, lbl_batch) in enumerate(tqdm.tqdm(valid_loader)):
                    img_batch, lbl_batch = img_batch.to(Config.General.device), lbl_batch.to(
                        Config.General.device).squeeze(1)

                    with torch.autocast(Config.General.device):
                        logits = model(img_batch)['out']
                        torch.use_deterministic_algorithms(False)

                        # loss = criterion(logits.permute(0, 2, 3, 1),
                        #                  torch.stack([-(lbl_batch - 1), lbl_batch], dim=1).permute(0, 2, 3, 1))

                        loss = criterion(logits.reshape(logits.shape[0], logits.shape[1], -1),
                                         lbl_batch.reshape(lbl_batch.shape[0], -1))

                    avg_loss(loss)
                    for mtr in metrics.values():
                        mtr(torch.argmax(logits, dim=1),
                            lbl_batch)

                    torch.use_deterministic_algorithms(True)

                    if i == sample:
                        sample_img[s] = img_batch
                        sample_gt[s] = lbl_batch

                    if i == len(valid_loader) - 1:
                        fixed_img[s] = img_batch
                        fixed_gt[s] = lbl_batch

                if s == 'ycb':
                    # score = list(metrics.values())[3].compute()
                    # if score > best_score:
                    #     best_score = score
                    #     torch.save(model.state_dict(), f'checkpoints/sym3/seg_model_f1{score}')
                    torch.save(model.state_dict(), f'checkpoints/sym5/epoch{e}')
                    torch.save(model.state_dict(), f'checkpoints/sym5/latest')

                if Config.Eval.wandb:
                    for k, v, in metrics.items():
                        wandb.log({f'valid/{k}_{s}': v.compute(),
                                   'global_step': global_step})
                        v.reset()

                wandb.log({f'valid/loss_{s}': avg_loss.compute(),
                           'global_step': global_step})

                avg_loss.reset()

            if Config.Eval.wandb:
                r_img, r_lbl = next(iter(synth_trainloader))
                for im, gt, idx, tx in [
                    [fixed_img['unity'], fixed_gt['unity'], 0, 'unity_fixed'],
                    [sample_img['unity'], sample_gt['unity'],
                     random.randint(0, sample_img['unity'].shape[0] - 1), 'unity_random'],
                    [synth_trainset[0][0].unsqueeze(0), synth_trainset[0][1], 0, 'fixed_train'],
                    [r_img, r_lbl.squeeze(), 0, 'random_train']
                ]:
                    tr = T.Compose([
                        lambda x: x.div(1 / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(x.device)),
                        lambda x: x.sub(-torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(x.device)),
                        T.Resize((480, 640), InterpolationMode.BILINEAR),
                        # lambda x: np.array(x.cpu()).transpose([1, 2, 0])
                    ])

                    with torch.autocast(Config.General.device):
                        y = model(im.to(Config.General.device))['out'][idx]
                        segmented, classes = utils.model.wrappers.Segmentator.postprocess(y, height=480, width=640)

                    wb_image = wandb.Image(tr(im[idx]),
                                           masks={'predictions': {'mask_data': classes,
                                                                  'class_labels': {0: 'background', 1: 'box'}},
                                                  'ground_truth': {'mask_data': cv2.resize(
                                                      gt[idx].cpu().numpy(),
                                                      dsize=(640, 480),
                                                      interpolation=cv2.INTER_NEAREST),
                                                      'class_labels': {0: 'background', 1: 'box'}}})
                    wandb.log({f'media/image_{tx}': wb_image})

                # [fixed_img['ycb'], fixed_gt['ycb'], 3, 'ycb_fixed'],
                # [sample_img['ycb'], sample_gt['ycb'],
                #  random.randint(0, sample_img['ycb'].shape[0] - 1), 'ycb_random'],

                for i, (img, label) in enumerate(real_validset):
                    img, label = img.unsqueeze(0), label.unsqueeze(0)
                    with torch.autocast(Config.General.device):
                        y = model(img.to(Config.General.device))['out']
                        segmented, classes = utils.model.wrappers.Segmentator.postprocess(y, height=480, width=640)

                    wb_image = wandb.Image(tr(img),
                                           masks={'predictions': {'mask_data': classes,
                                                                  'class_labels': {0: 'background', 1: 'box'}},
                                                  'ground_truth': {'mask_data': cv2.resize(
                                                      label.squeeze().numpy(),
                                                      dsize=(640, 480),
                                                      interpolation=cv2.INTER_NEAREST),
                                                      'class_labels': {0: 'background', 1: 'box'}}})
                    wandb.log({f'real/image{i}': wb_image})


        print('Training...')
        model.train()
        for i, (img_batch, lbl_batch) in enumerate(tqdm.tqdm(synth_trainloader)):
            img_batch, lbl_batch = img_batch.to(Config.General.device), lbl_batch.to(Config.General.device).squeeze()

            with torch.autocast(Config.General.device):
                logits = model(img_batch)['out']
                torch.use_deterministic_algorithms(False)

                # loss = criterion(logits.permute(0, 2, 3, 1),
                #                  torch.stack([-(lbl_batch - 1), lbl_batch], dim=1).permute(0, 2, 3, 1))

                loss = criterion(logits.reshape(logits.shape[0], logits.shape[1], -1),
                                 lbl_batch.reshape(lbl_batch.shape[0], -1)) # / Config.Train.update_step

            loss.backward()
            torch.use_deterministic_algorithms(True)

            if i % Config.Train.update_step == Config.Train.update_step - 1 or i == len(synth_trainloader) - 1:
                optim.step()
                optim.zero_grad()

                if i == len(synth_trainloader) - 1:
                    if Config.Eval.wandb:
                        for k, v, in metrics.items():
                            torch.use_deterministic_algorithms(False)
                            wandb.log({f'train/{k}': v(torch.argmax(logits, dim=1), lbl_batch),
                                       'global_step': global_step})
                            torch.use_deterministic_algorithms(True)
                            v.reset()

                        wandb.log({'train/loss': loss.item(),
                                   'global_step': global_step})

                global_step += i + (e * len(synth_trainloader))


class DiceLoss(nn.Module):

    def forward(self, scores, target):
        return soft_dice_loss(target, F.softmax(scores, dim=3))


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * (y_pred * y_true).sum(axes)
    denominator = (torch.square(y_pred) + torch.square(y_true)).sum(axes)

    return 1 - ((numerator + epsilon) / (denominator + epsilon)).mean()  # average over classes and batch


if __name__ == '__main__':
    main()
    # criterion = DiceLoss()
    #
    # prediction = torch.rand(1, 2, 192, 256)
    # prediction[:, 1, ...] = 1 - prediction[:, 0, ...]
    # prediction.permute(0, 2, 3, 1)
    #
    # ground_truth = torch.rand(1, 192, 256)
    # ground_truth = ground_truth.round().to(int)
    # ground_truth = torch.stack([ground_truth, -(ground_truth - 1)], dim=1)
    # ground_truth.permute(0, 2, 3, 1)
    #
    # criterion(prediction, ground_truth)
