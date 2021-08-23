import os
import os.path as osp
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from load_video import LoadVideo


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 800.)
    text_thickness = max(1, image.shape[1] // 500)
    line_thickness = max(1, image.shape[1] // 500)

    radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 10 * text_thickness),
                    cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)
    return im


def show_anno(video_fname, out_fpath, show_image=True, save_image=False, compress_video=False,
              img_size=(3072, 2304), step=1):
    dataloader = LoadVideo(video_fname, img_size, step)
    results = np.loadtxt(f'{out_fpath}/results.txt', delimiter=',')

    for i, (frame_id, img, img0) in enumerate(dataloader):
        frame_result = results[results[:, 0] == frame_id]
        anno_ids, anno_tlwhs = frame_result[:, 1], frame_result[:, 2:6]
        anno_tlwhs[:, :2] = np.clip(anno_tlwhs[:, :2], 0, np.inf)
        anno_img = plot_tracking(img0, anno_tlwhs, anno_ids, frame_id=frame_id)
        if show_image:
            # Acquire default dots per inch value of matplotlib
            dpi = matplotlib.rcParams['figure.dpi']
            H, W, C = anno_img.shape
            fig = plt.figure(frameon=False)
            fig.set_size_inches(W / float(dpi), H / float(dpi))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(cv2.cvtColor(anno_img, cv2.COLOR_BGR2RGB), aspect='auto')
            plt.show()
        if save_image is not None:
            cv2.imwrite(os.path.join(f'{out_fpath}/frame', '{:05d}.jpg'.format(frame_id)), anno_img)

    # video
    if compress_video:
        assert save_image is True
        os.system(f'/usr/bin/ffmpeg -y -f image2 -i {osp.join(out_fpath, "frame")}/%05d.jpg -vf scale=-1:768 '
                  f'-b 5000k -c:v mpeg4 {osp.join(out_fpath, "results.mp4")}')


if __name__ == '__main__':
    root = '/home/houyz/Data/cattle/test'
    for seq_name in sorted(os.listdir(root)):
        print(seq_name)
        show_anno(glob.glob(f'{root}/{seq_name}/*.mp4')[0], f'../../demos/{seq_name}', step=100)
