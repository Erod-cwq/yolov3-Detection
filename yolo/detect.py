import glob
import os
from pathlib import Path
import cv2.cv2 as cv2
import requests
import torch
from model.yolo import Model
import numpy as np
import torch.nn as nn
from model.modules import Detect, Conv
from utils.general import non_max_suppression, scale_coords

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def detect(path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('yolov3_state_dict.pt', map_location=device)
    model = Model('./model/yolo.yaml', nc=None).to(device)
    model.to(device)
    model.load_state_dict(state_dict)
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()

    model.eval()
    file = requests.get(path)
    img0 = cv2.imdecode(np.frombuffer(file.content, np.uint8), 1)
    assert img0 is not None
    img, ratio, (dw, dh) = letterbox(img0, 640, stride=32)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print(img.shape)
    pred = model(img)[0]
    pred = non_max_suppression(pred)
    dets = []
    for i, det in enumerate(pred):

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            dets.append(det.tolist())
    return dets


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 计算padding
    ratio = r, r  # 宽，高 比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # 在图片四周填充
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


class LoadImages:
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())

        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
            print(files)
        elif os.path.isfile(p):
            files = p
        else:
            raise Exception(f'Error:{p} does not exist')
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = len(images)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        self.count += 1
        img0 = cv2.imread(path)
        assert img0 is not None
        img, ratio, (dw, dh) = letterbox(img0, self.img_size, stride=self.stride)
        # cv2.imwrite('padding.jpg', img)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0


if __name__ == '__main__':
    detect('http://127.0.0.1:8080/upload/umbrella.jpg')
