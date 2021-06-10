import logging
import os
import glob
import os.path as osp
from lib.opts import opts
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import lib.datasets.jde as datasets
from track import eval_seq

logger.setLevel(logging.INFO)


def demo(opt, fname, out_fpath):
    mkdir_if_missing(out_fpath)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(fname, opt.img_size)
    result_filename = os.path.join(out_fpath, 'results.txt')

    frame_dir = None if opt.output_format == 'text' else osp.join(out_fpath, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename, save_dir=frame_dir, show_image=False)

    if opt.output_format == 'video':
        os.system(f'/usr/bin/ffmpeg -y -f image2 -i {osp.join(out_fpath, "frame")}/%05d.jpg '
                  f'-b 5000k -c:v mpeg4 {osp.join(out_fpath, "results.mp4")}')


if __name__ == '__main__':
    opt = opts().init()
    root = '/home/houyz/Data/cattle/test'
    for seq_name in sorted(os.listdir(root)):
        if '4-05-21' in seq_name:
            continue
        print(seq_name)
        demo(opt, glob.glob(f'{root}/{seq_name}/*.mp4')[0], f'../demos/{seq_name}')
