import glob
import math
import os
import random
import time
import re

import cv2
import numpy as np
import copy

from torchvision.datasets import VisionDataset
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.utils import xyxy2ctwh
from lib.datasets.jde import letterbox, random_affine


def csv_to_gt(root):
    for seq_name in sorted(os.listdir(root)):
        frames = []
        for img in sorted(glob.glob(f'{root}/{seq_name}/images/*.jpg')):
            frames.append(int(re.search(r'\d+', img.split('/')[-1].split('.')[0]).group()))
        frames = np.array(frames)
        gt = np.nan_to_num(np.genfromtxt(glob.glob(f"{root}/{seq_name}/*.csv")[0], delimiter=','))
        gt[:, 0] = frames[gt[:, 0].astype(int)]
        np.savetxt(f'{root}/{seq_name}/gt.txt', gt, delimiter=',', fmt='%.3f', )
        pass
    pass


class MOTDataset(VisionDataset):
    def __init__(self, opt, root, img_size=(1024, 768), augment=False, transforms=None):
        super().__init__(root, transforms)
        self.opt = opt
        self.num_classes = 1
        self.imgs_seq, self.seq_gts = {}, {}
        self.seq_num_id, self.seq_start_id = {}, {}
        start_id = 0
        csv_to_gt(root)
        for seq_name in sorted(os.listdir(root)):
            for img in sorted(glob.glob(f'{root}/{seq_name}/images/*.jpg')):
                self.imgs_seq[img] = seq_name
            self.seq_gts[seq_name] = np.genfromtxt(f'{root}/{seq_name}/gt.txt', delimiter=',')
            self.seq_num_id[seq_name] = len(np.unique(self.seq_gts[seq_name][:, 1]))
            self.seq_start_id[seq_name] = start_id
            start_id += self.seq_num_id[seq_name]
        self.nID = start_id
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs_seq)

    def __getitem__(self, index):
        img_fname = list(self.imgs_seq.keys())[index]
        seq_name = self.imgs_seq[img_fname]

        img, labels = self.get_data(img_fname, self.seq_gts[seq_name])

        labels[labels[:, 1] > -1, 1] += self.seq_start_id[seq_name]

        output_h, output_w = self.height // self.opt.down_ratio, self.width // self.opt.down_ratio
        num_classes = 1
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)
        ids = np.zeros((self.max_objs,), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        # de-normalize
        labels[:, 2] *= output_w
        labels[:, 3] *= output_h
        labels[:, 4] *= output_w
        labels[:, 5] *= output_h
        for k in range(num_objs):
            bbox = copy.deepcopy(labels[k, 2:])
            cls_id = 0
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                # radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = labels[k, 1]
                bbox_xys[k] = ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2

        ret = {'input': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids,
               'bbox': bbox_xys}
        return ret

    def get_data(self, img_fname, gt, visualize=False):

        frame = int(re.search(r'\d+', img_fname.split('/')[-1].split('.')[0]).group())
        gt = gt[gt[:, 0] == frame, :6]

        height = self.height
        width = self.width
        img = cv2.imread(img_fname)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_fname))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        labels0 = gt

        # Normalized xywh to pixel xyxy format
        labels = labels0.copy()
        labels[:, 2] = ratio * labels0[:, 2] + padw
        labels[:, 3] = ratio * labels0[:, 3] + padh
        labels[:, 4] = ratio * (labels0[:, 2] + labels0[:, 4]) + padw
        labels[:, 5] = ratio * (labels0[:, 3] + labels0[:, 5]) + padh

        if visualize:
            import matplotlib.pyplot as plt
            for bbox in labels:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img, tuple(bbox[2:4]), tuple(bbox[4:]), (0, 255, 0), 2)
            plt.imshow(img)
            plt.show()

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        if visualize:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2ctwh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels


if __name__ == '__main__':
    from lib.opts import opts
    from torchvision.transforms import transforms as T

    opt = opts().parse()
    dataset = MOTDataset(opt, '/home/houyz/Data/cattle/train', augment=True, transforms=T.ToTensor())
    ret = dataset.__getitem__(len(dataset) - 1)
    pass
