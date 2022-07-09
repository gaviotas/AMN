
from locale import normalize
import torch
import torch.nn as nn
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import coco14.dataloader
from misc import pyutils, torchutils, imutils

import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.utils as vutils
import os
import imageio


def balanced_cross_entropy(logits, labels, one_hot_labels):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """

    N, C, H, W = logits.shape

    assert one_hot_labels.size(0) == N and one_hot_labels.size(1) == C, f'label tensor shape is {one_hot_labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    loss_structure = -torch.sum(log_logits * one_hot_labels, dim=1)  # (N)

    ignore_mask_bg = torch.zeros_like(labels)
    ignore_mask_fg = torch.zeros_like(labels)
    
    ignore_mask_bg[labels == 0] = 1
    ignore_mask_fg[(labels != 0) & (labels != 255)] = 1
    
    loss_bg = (loss_structure * ignore_mask_bg).sum() / ignore_mask_bg.sum()
    loss_fg = (loss_structure * ignore_mask_fg).sum() / ignore_mask_fg.sum()

    return (loss_bg+loss_fg)/2


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def run(args):

    model = getattr(importlib.import_module(args.amn_network), 'Net')()

    train_dataset = coco14.dataloader.COCO14SegmentationDataset(args.train_list,
                                                              label_dir=args.ir_label_out_dir,
                                                              coco14_root=args.coco14_root,
                                                              hor_flip=True,
                                                              crop_size=args.irn_crop_size,
                                                              crop_method="random",
                                                              rescale=(0.5, 1.5)
                                                             )
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    param_groups = model.trainable_parameters()

    optimizer = torch.optim.Adam(
        params=[
            {
                'params': param_groups[0],
                'lr': 5e-06,
                'weight_decay': 1.0e-4,
            },
            {
                'params': param_groups[1],
                'lr': 1e-04,
                'weight_decay': 1.0e-4,
            },
        ],
    )

    total_epochs = 5

    model = torch.nn.DataParallel(model).cuda()

    model.train()

    avg_meter = pyutils.AverageMeter()
    
    for ep in range(total_epochs):
        loader_iter = iter(train_data_loader)

        pbar = tqdm(
            range(1, len(train_data_loader) + 1),
            total=len(train_data_loader),
            dynamic_ncols=True,
        )

        for iteration, _ in enumerate(pbar):
            optimizer.zero_grad()
            try:
                pack = next(loader_iter)
            except:
                loader_iter = iter(train_data_loader)
                pack = next(loader_iter)

            img = pack['img'].cuda(non_blocking=True)
            label_seg = pack['label'].long().cuda(non_blocking=True)
            label_cls = pack['label_cls'].cuda(non_blocking=True).float()

            logit = model(img, label_cls)

            B, C, H, W = logit.shape

            label_seg = resize_labels(label_seg.cpu(), size=logit.shape[-2:]).cuda()

            label_ = label_seg.clone()
            label_[label_seg == 255] = 0

            given_labels = torch.full(size=(B, C, H, W), fill_value=args.EPS/(C-1)).cuda()
            given_labels.scatter_(dim=1, index=torch.unsqueeze(label_, dim=1), value=1-args.EPS)

            loss_seg = balanced_cross_entropy(logit, label_seg, given_labels)

            loss = loss_seg
            loss.backward()

            optimizer.step()

            avg_meter.add({'loss_seg': loss_seg.item()})

            pbar.set_description(f"[{ep}/{total_epochs}] "
                                f"S: [{avg_meter.pop('loss_seg'):.4f}]")
    
    torch.save(model.module.state_dict(), args.amn_weights_name + '.pth')
    torch.cuda.empty_cache()