
import numpy as np
import os
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image
from tqdm import tqdm

def run(args):

    ids = open('coco14/train14.txt').readlines()
    ids = [i.split('\n')[0] for i in ids]
    preds = []
    labels = []
    n_images = 0

    for i, id in enumerate(tqdm(ids)):
        label = np.array(Image.open('/data/coco_2014/coco_seg_anno/%s.png' % id))
        n_images += 1
        cam_dict = np.load(os.path.join(args.amn_cam_out_dir, id + '.npy'), allow_pickle=True).item()
        if not ('high_res' in cam_dict):
            preds.append(np.zeros_like(label))
            labels.append(label)
            continue

        cams = cam_dict['high_res']
        cams = cams[1:, ...]

        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        labels.append(label)
        xx, yy = cls_labels.shape, label.shape
        if xx[0] != yy[0]:
            print(id, xx, yy)

    confusion = calc_semantic_segmentation_confusion(preds, labels)
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)

    return np.nanmean(iou)