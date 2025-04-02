
from torchvision import transforms
import torch
from PIL import ImageDraw,Image,ImageFile, ExifTags
import cv2
import numpy as np

import torchvision
import os,json
from units import cfg,tools
from collections import Counter
# from detect_new.net_point import MainNet

ImageFile.LOAD_TRUNCATED_IMAGES = True

tf = transforms.Compose([transforms.ToTensor()])

#图片处理
class ImagePolice(torch.nn.Module):
    def __init__(self):
        super(ImagePolice, self).__init__()

    def forward(self, img):
        img1 = self.Sharpen(img)
        img2 = self.Outline1(img)
        img3 = self.Outline2(img)
        img_all = np.stack([img1, img2, img3])

        img_all_ = np.transpose(img_all, (1, 2, 0))
        img = Image.fromarray(img_all_)

        img_data = tf(img)


        return img_data,img

    def Sharpen(self,img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        dst = cv2.filter2D(img, -1, kernel=kernel)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        return dst

    def Outline1(self,img):
        img = cv2.convertScaleAbs(img, alpha=5, beta=0)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        canny = cv2.Canny(img, 50, 150)

        return canny

    def Outline2(self,img):

        img = cv2.GaussianBlur(img, (5, 5), 10)
        canny = cv2.Canny(img, 50, 150)

        return canny


#x片定点
class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.police = ImagePolice()


        self.SN_dis = 0

        self.net =  torch.jit.load('./weight/net_net_7_1.pt')
        self.name = ['S', 'N', ]


    def forward(self, path,isswith=False):

        img = cv2.imread(path)
        out = self.detect_dot(img)
        return out




    def detect_dot(self,img,box=[0,0]):
        w, h = img.shape[1], img.shape[0]
        x_w = w / 2420
        y_h = h / 2420
        imgs = cv2.resize(img, (2420, 2420))
        img_data, _ = self.police(imgs)
        img_data = img_data.view((-1, 3, 2420, 2420)).cuda()
        with torch.jit.optimized_execution(False):
            output = self.net(img_data).cpu().detach().numpy()

        e = {"S": [1156,823],"N": [1939,774]}
        out = {}
        for i in range(2):
            name = self.name[i]
            x_ = e[name][0]
            y_ = e[name][1]
            x = int((x_ - x_ * output[0][i * 2]) * x_w)+box[0]
            y = int((y_ - y_ * output[0][i * 2 + 1]) * y_h)+box[1]
            out[name ] = [x, y]
        return out



















