import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils
import imageio
from tqdm import tqdm
from PIL import Image

cudnn.enabled = True


def _work(process_id, model, dataset, args):

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):

            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            label_amn = Image.open(os.path.join(args.ir_label_out_dir, img_name + '.png'))

            strided_size = imutils.get_strided_size(size, 4)

            label_amn_down = np.array(label_amn.resize((strided_size[1], strided_size[0]), resample=Image.NEAREST))
            label_amn_up = np.array(label_amn.resize((size[1], size[0]), resample=Image.NEAREST))

            outputs = [model(img[0].cuda(non_blocking=True), label.unsqueeze(0).expand((img[0].size(0), 20)).cuda())
                       for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in outputs]), 0) / len(outputs)

            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]] / len(outputs)

            valid_cat = torch.nonzero(label)[:, 0]
            keys = np.pad(valid_cat + 1, (1, 0), mode='constant')

            strided_cam = F.softmax(strided_cam, dim=0)
            highres_cam = F.softmax(highres_cam, dim=0)

            strided_cam = strided_cam[keys]
            highres_cam = highres_cam[keys]

            strided_cam = strided_cam.detach().cpu().numpy()
            highres_cam = highres_cam.detach().cpu().numpy()

            strided_cam_norm = (strided_cam - np.min(strided_cam, (1, 2), keepdims=True)) / (np.max(strided_cam, (1, 2), keepdims=True) - np.min(strided_cam, (1, 2), keepdims=True) + 1e-5)
            highres_cam_norm = (highres_cam - np.min(highres_cam, (1, 2), keepdims=True)) / (np.max(highres_cam, (1, 2), keepdims=True) - np.min(highres_cam, (1, 2), keepdims=True) + 1e-5)

            strided_mask_bg = torch.tensor(label_amn_down == 0).unsqueeze(0).expand(strided_cam.shape).numpy()
            highres_mask_bg = torch.tensor(label_amn_up == 0).unsqueeze(0).expand(highres_cam.shape).numpy()

            strided_cam[~strided_mask_bg] = strided_cam_norm[~strided_mask_bg]
            highres_cam[~highres_mask_bg] = highres_cam_norm[~highres_mask_bg]

            # save cams
            np.save(os.path.join(args.amn_cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": torch.tensor(strided_cam), "high_res": highres_cam})


def run(args):
    model = getattr(importlib.import_module(args.amn_network), 'CAM')()

    model.load_state_dict(torch.load(args.amn_weights_name + '.pth'), strict=True)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list,
                                                             voc12_root=args.voc12_root, scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)

    torch.cuda.empty_cache()