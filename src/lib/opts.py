from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import torch


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--exp_id', default='cattle_dla34')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--load_model', default='../models/ctdet_coco_dla_2x.pth',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                                 help='visualization threshold.')

        # model
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                      'dla_34 | hrnet_18')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '256 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help='learning rate for batch size 12.')
        self.parser.add_argument('--lr_step', type=str, default='20',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=12,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and '
                                      'test on test set')

        # test
        self.parser.add_argument('--K', type=int, default=50,
                                 help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        # tracking
        self.parser.add_argument('--conf_thres', type=float, default=0.4, help='confidence thresh for tracking')
        self.parser.add_argument('--det_thres', type=float, default=0.3, help='confidence thresh for detection')
        self.parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresh for nms')
        self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
        self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
        self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
        self.parser.add_argument('--output-root', type=str, default='../demos', help='expected output root path')

        # mot
        self.parser.add_argument('--data_cfg', type=str,
                                 default='../src/lib/cfg/data.json',
                                 help='load data from cfg')
        self.parser.add_argument('--data_dir', type=str, default='/home/houyz/Data')

        # loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')

        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss', default='ce',
                                 help='reid loss: ce | focal')
        self.parser.add_argument('--id_weight', type=float, default=1,
                                 help='loss weight for id')
        self.parser.add_argument('--reid_dim', type=int, default=128,
                                 help='feature dim for reid')
        self.parser.add_argument('--ltrb', default=False,
                                 help='regress left, top, right, bottom of bbox')
        self.parser.add_argument('--multi_loss', default='uncertainty', help='multi_task loss: uncertainty | fix')

        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus = [gpu for gpu in range(torch.cuda.device_count())]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp')
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def init(self, args=''):
        default_dataset_info = {'width': 768, 'height': 1024,
                                'num_classes': 1, 'nID': 10,
                                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278]}

        class Struct:
            def __init__(self, entries):
                for k in entries:
                    self.__setattr__(k, entries[k])

        opt = self.parse(args)
        dataset = Struct(default_dataset_info)
        opt = update_dataset_info_and_set_heads(opt, dataset)
        return opt


def update_dataset_info_and_set_heads(opt, dataset):
    opt.num_classes = dataset.num_classes

    opt.heads = {'hm': opt.num_classes,
                 'wh': 2 if not opt.ltrb else 4,
                 'id': opt.reid_dim}
    if opt.reg_offset:
        opt.heads.update({'reg': 2})
    opt.nID = dataset.nID
    opt.img_size = (dataset.width, dataset.height)
    print('heads', opt.heads)
    return opt
