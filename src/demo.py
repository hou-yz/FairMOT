import logging
import os
import glob
import shutil
import os.path as osp

import numpy as np
import cv2

from lib.opts import opts
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import lib.datasets.jde as datasets
from track import eval_seq

logger.setLevel(logging.INFO)


def extract_headshot(result_filename, dataloader, save_dir):
    res = np.genfromtxt(result_filename, delimiter=',')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for i, (path, img, img0) in enumerate(dataloader):
        if isinstance(path, int):
            frame = path + 1
        else:
            frame = int(path)
        res_frame = res[res[:, 0] == frame, :]
        for line in res_frame:
            id = int(line[1])
            x, y, w, h = line[[2, 3, 4, 5]]
            x1 = int(min(max(x, 0), dataloader.vw))
            x2 = int(min(max(x + w, 0), dataloader.vw))
            y1 = int(min(max(y, 0), dataloader.vh))
            y2 = int(min(max(y + h, 0), dataloader.vh))
            cv2.imwrite(osp.join(save_dir, f'{id:02d}_f{frame:05d}.jpg'), img0[y1:y2, x1:x2])
    print(f'saved {len(res)} headshots from {len(dataloader)} frames')


def demo(opt, fname, out_fpath):
    mkdir_if_missing(out_fpath)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(fname, opt.img_size)
    result_filename = os.path.join(out_fpath, 'results.txt')

    frame_dir = None if opt.output_format == 'text' else osp.join(out_fpath, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir, show_image=False)

    # video
    os.system(f'/usr/bin/ffmpeg -y -f image2 -i {osp.join(out_fpath, "frame")}/%05d.jpg -vf scale=-1:720 '
              f'-b 5000k -c:v mpeg4 {osp.join(out_fpath, "results.mp4")}')
    # headshot
    # dataloader = datasets.LoadVideo(fname, opt.img_size)
    # extract_headshot(result_filename, dataloader, osp.join(out_fpath, 'headshot'))


if __name__ == '__main__':
    opt = opts().init()
    root = '/home/houyz/Data/cattle/test'
    for seq_name in sorted(os.listdir(root)):
        print(seq_name)
        demo(opt, glob.glob(f'{root}/{seq_name}/*.mp4')[0], f'../demos/{seq_name}')
        shutil.rmtree(f'../demos/{seq_name}/frame')

        # break
