import cv2
import numpy as np


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1024, 768), frame_step=1):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 5000)

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0
        self.step = frame_step

        print(f'num_frames: {self.vn}, step: {frame_step}')

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = 0
        self.cap = cv2.VideoCapture(self.path)
        return self

    def __next__(self):
        for _ in range(self.step - 1):
            self.cap.read()
        self.count += self.step
        if self.count > self.vn:
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        if img0 is None:
            img0 = np.zeros([self.vh, self.vw, 3], dtype=np.uint8)

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)  # C,H,W

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn // self.step


def letterbox(img, height=768, width=1024,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


if __name__ == '__main__':
    import os

    dataloader = LoadVideo(os.path.expanduser('~/Data/cattle/test/ANU_hiv00144_03-06-21/hiv00144.mp4'))
    for i, (frame, downsampled_pytorch_img, original_cv2_img) in enumerate(dataloader):
        pass
    print(f'index: {i}, frame: {frame}, downsampled pytorch img size: {downsampled_pytorch_img.shape}, '
          f'original cv2 img size: {original_cv2_img.shape}')
