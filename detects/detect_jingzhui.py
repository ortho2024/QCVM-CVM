
from torchvision import transforms
import torch
from PIL import ImageDraw,Image,ImageFile, ExifTags
import cv2
import numpy as np

import torchvision
import os,json
from units import cfg,tools
from collections import Counter
from models.resnet_cbam import MainNet
from math import *

#检测颈椎位置
class Detector_Jingzhui(torch.nn.Module):

    def __init__(self):
        super(Detector_Jingzhui, self).__init__()
        self.net = torch.jit.load('./weight_jingzhui/net_jingzhui_detection.pt')
        self.tf = transforms.Compose([transforms.ToTensor()])

    def forward(self, path, thresh =0.5, anchors = cfg.ANCHORS_GROUPS):

        # image = Image.open(path).convert('RGB')
        img = cv2.imread(path)
        image = Image.fromarray(img, 'RGB')
        width, high = image.size
        x_w = width / 416
        y_h = high / 416
        cropimg = image.resize((416, 416))
        imgdata = self.tf(cropimg)
        imgdata = torch.FloatTensor(imgdata).view(-1, 3, 416, 416).cuda()
        with torch.jit.optimized_execution(False):
            output_13, output_26, output_52 = self.net(imgdata)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        boxes_all = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

        if boxes_all.size(0)==0:
            return torch.tensor([[]])

        boxes = tools.nms(boxes_all,0.3,)
        if len(boxes[0]) != 0:
            boxess = []
            for i in boxes:
                x1 = int((i[1]) * x_w)
                y1 = int((i[2]) * y_h)
                x2 = int((i[3]) * x_w)
                y2 = int((i[4]) * y_h)
                boxess.append([i[-2], x1, y1, x2, y2])
        return boxess

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        torch.sigmoid_(output[...,4])
        torch.sigmoid_(output[...,0:2])

        mask = output[..., 4] > thresh
        idxs = mask.nonzero(as_tuple=False)
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)
        vecs = vecs.cpu().data
        idxs = idxs.cpu().data
        if idxs.shape[0]==0:
            return torch.Tensor([])
        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框
        condince = vecs[:, 4]
        cy = (idxs[:, 1].float() + vecs[:, 1]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 0]) * t  # 原图的中心点x
        w = anchors[a, 0] * torch.exp(vecs[:, 2])
        h = anchors[a, 1] * torch.exp(vecs[:, 3])
        classname = vecs[:,5:]
        cls = torch.argmax(classname, dim=1).float() #类别
        w_0,h_0 = w/2,h/2
        x1,y1,x2,y2 = cx-w_0,cy-h_0,cx+w_0,cy+h_0

        return torch.stack([condince,x1,y1,x2,y2,cls,n.float()], dim=1)


class Detector_JingZhui_dot(torch.nn.Module):

    def __init__(self):
        super(Detector_JingZhui_dot, self).__init__()

        self.detect_CS = Detector_JingZhui_CS()
        self.SN_dis = 0
        self.net_jingzhui = Detector_Jingzhui()

        self.net =  torch.jit.load('./weight_jingzhui/net.pt')
        self.net_crop =  torch.jit.load('./weight_jingzhui/net_crop.pt')
        self.net_C2 =  torch.jit.load('./weight_jingzhui/net_C2.pt')
        self.net_C3 =  torch.jit.load('./weight_jingzhui/net_C3.pt')
        self.net_C4 =  torch.jit.load('./weight_jingzhui/net_C4.pt')
        self.tf = transforms.Compose([transforms.ToTensor()])
        self.name = ['C2p','C2d','C2a','C3up','C3ua','C3lp','C3la','C4up','C4um','C4am','C4lp','C4la','C3ld','C4ld']

    def forward(self, path,out_SN):

        S = out_SN['S']
        N = out_SN['N']
        self.SN_dis = ((S[0] - N[0]) ** 2 + (S[1] - N[1]) ** 2) ** 0.5
        img = cv2.imread(path)
        imgs = Image.fromarray(img, 'RGB')
        width, high = imgs.size
        boxes = self.net_jingzhui(path)

        if len(boxes[0]) != 0:
            for box in boxes:
                if int(box[3]) > width * 3 / 4 or int(box[2]) < high / 5:
                    continue

                imgcrop = imgs.crop((box[1:]))
                iscrop = True
                out = self.detect_dot(imgcrop,box[1:3],iscrop)

        else:

            out = self.detect_dot(imgs)
            print(out)

        self.outline = {}
        out = self.detect_C2(imgs, out)
        out = self.detect_C3(imgs, out)
        out = self.detect_C4(imgs, out)
        keys = out.keys()

        for name in list(keys):
            if '_' in name:
                del out[name]

        value = out.values()
        value = np.array(list(value))
        _max = np.max(value, axis=0)
        _min = np.min(value, axis=0)

        max_box = _max + np.array([self.SN_dis / 5, self.SN_dis / 3.5])
        min_box = _min - np.array([self.SN_dis / 2, self.SN_dis / 1.5])
        hight = imgs.size[1]
        if max_box[1] > hight:
            max_box[1] = hight - 20
        if min_box[0] < 0:
            min_box[0] = 0
        boxes = np.vstack((min_box, max_box)).reshape(-1, 4)
        result = self.detect_CS(path,out,self.outline,boxes[0].tolist())
        print(result)

        return out,self.outline,boxes[0].tolist()


    def detect_dot(self,img,box=[0,0],iscrop = False):
        if iscrop:

            w, h = img.size
            x_w = w / 208
            y_h = h / 208
            imgs = img.resize((208, 208), )
            # imgs = self.Equalization(np.array(imgs))
            img_data = self.tf(imgs)
            img_data = img_data.view((-1, 3, 208, 208)).cuda()
            with torch.jit.optimized_execution(False):
                output = self.net_crop(img_data).cpu().detach().numpy()


            f = open(r'./weight_jingzhui/maodian_jingzhui_crop.json')
            d = f.read()
            e = json.loads(d)
            out = {}
            for i in range(14):
                name = self.name[i]
                x_ = e[name][0]
                y_ = e[name][1]
                x = int((x_ - x_ * output[0][i * 2]) * x_w) + box[0]
                y = int((y_ - y_ * output[0][i * 2 + 1]) * y_h) + box[1]
                out[name] = [x, y]
            return out
        else:
            w, h = img.size
            x_w = w / 2420
            y_h = h / 2420
            imgs =img.resize((2420,2420),)
            img_data= self.tf(imgs)
            img_data = img_data.view((-1, 3, 2420, 2420)).cuda()
            with torch.jit.optimized_execution(False):
                output = self.net(img_data).cpu().detach().numpy()
                print(output,1111)
            f = open(r'./weight_jingzhui/maodian_jingzhui.json')
            d = f.read()
            e = json.loads(d)
            out = {}
            for i in range(14):
                name = self.name[i]
                x_ = e[name][0]
                y_ = e[name][1]
                x = int((x_ - x_ * output[0][i * 2]) * x_w)+box[0]
                y = int((y_ - y_ * output[0][i * 2 + 1]) * y_h) + box[1]
                out[name ] = [x, y]
            return out

    def detect_C2(self, img, boxes):

        plus_size = self.SN_dis // 2
        C2 = {}
        order = []
        cp = ((np.array(boxes['C2p']) + np.array(boxes['C2a'])) / 2).tolist()
        cp[1] = cp[1] - self.SN_dis / 3
        img_data = []
        w_h_data = []

        x, y = cp[0], cp[1]
        img_crop = img.crop((x - plus_size, y - plus_size, x + plus_size, y + plus_size))
        w, h = img_crop.size
        w_h_data.append([w, h])
        img_crop = img_crop.resize((416, 416))
        img_data.append(self.tf(img_crop))
        img_data = torch.stack(img_data).cuda()
        with torch.jit.optimized_execution(False):
            out = self.net_C2(img_data).cpu().detach().numpy()
        boxes_copy = {}
        dot_names = [
            ["C2_1", "C2_2", "C2p", "C2_3", "C2d", "C2_4", "C2a", "C2_5", "C2_6", "C2_7", "C2_8", "C2_9", "C2_10"]]
        for i, _name in enumerate(dot_names):
            for j, name in enumerate(_name):
                _x_Bo = out[i][j * 2] * w_h_data[i][0] + cp[0] - plus_size
                _y_Bo = out[i][j * 2 + 1] * w_h_data[i][1] + cp[1] - plus_size
                boxes_copy[name] = [round(_x_Bo), round(_y_Bo)]
                if '_' in name:
                    C2[name] = [round(_x_Bo), round(_y_Bo)]
                    order.append(name)
                else:
                    C2['{}_O'.format(name)] = [round(_x_Bo), round(_y_Bo)]
                    order.append('{}_O'.format(name))
        order.append(order[0])
        C2['order'] = order
        self.outline['C2'] = C2
        for key in boxes_copy.keys():
            boxes[key] = boxes_copy[key]

        return boxes

    def detect_C3(self, img, boxes):

        plus_size = self.SN_dis // 5
        C3 = {}
        order = []
        cp = ((np.array(boxes['C3up']) + np.array(boxes['C3la'])) / 2).tolist()
        img_data = []
        w_h_data = []

        x, y = cp[0], cp[1]
        img_crop = img.crop((x - plus_size, y - plus_size, x + plus_size, y + plus_size))
        w, h = img_crop.size
        w_h_data.append([w, h])
        img_crop = img_crop.resize((416, 416))
        img_data.append(self.tf(img_crop))
        img_data = torch.stack(img_data).cuda()
        with torch.jit.optimized_execution(False):
            out = self.net_C3(img_data).cpu().detach().numpy()
        boxes_copy = {}
        dot_names = [["C3up", "C3_1", "C3pm", "C3_2", "C3lp", "C3_3", "C3ld", "C3_4", "C3la", "C3_5", "C3am", "C3_6", "C3ua", "C3_7", "C3um", "C3_8"]]
        for i, _name in enumerate(dot_names):
            for j, name in enumerate(_name):

                _x_Bo = out[i][j * 2] * w_h_data[i][0] + cp[0] - plus_size
                _y_Bo = out[i][j * 2 + 1] * w_h_data[i][1] + cp[1] - plus_size
                boxes_copy[name] = [round(_x_Bo), round(_y_Bo)]
                if '_' in name:
                    C3[name] = [round(_x_Bo), round(_y_Bo)]
                    order.append(name)
                else:
                    C3['{}_O'.format(name)] = [round(_x_Bo), round(_y_Bo)]
                    order.append('{}_O'.format(name))
        order.append(order[0])
        C3['order'] = order
        self.outline['C3'] = C3

        for key in boxes_copy.keys():
            boxes[key] = boxes_copy[key]
        return boxes

    def detect_C4(self, img, boxes):

        plus_size = self.SN_dis // 5
        C4,order = {},[]
        cp = ((np.array(boxes['C4up']) + np.array(boxes['C4la'])) / 2).tolist()
        img_data = []
        w_h_data = []

        x, y = cp[0], cp[1]
        img_crop = img.crop((x - plus_size, y - plus_size, x + plus_size, y + plus_size))
        w, h = img_crop.size
        w_h_data.append([w, h])
        img_crop = img_crop.resize((416, 416))
        img_data.append(self.tf(img_crop))
        img_data = torch.stack(img_data).cuda()
        with torch.jit.optimized_execution(False):
            out = self.net_C4(img_data).cpu().detach().numpy()
        boxes_copy = {}
        dot_names = [["C4up", "C4_1", "C4pm", "C4_2", "C4lp", "C4_3", "C4ld", "C4_4", "C4la", "C4_5", "C4am", "C4_6", "C4ua", "C4_7", "C4um", "C4_8"]]
        for i, _name in enumerate(dot_names):
            for j, name in enumerate(_name):
                _x_Bo = out[i][j * 2] * w_h_data[i][0] + cp[0] - plus_size
                _y_Bo = out[i][j * 2 + 1] * w_h_data[i][1] + cp[1] - plus_size
                boxes_copy[name] = [round(_x_Bo), round(_y_Bo)]
                if '_' in name:
                    C4[name] = [round(_x_Bo), round(_y_Bo)]
                    order.append(name)
                else:
                    C4['{}_O'.format(name)] = [round(_x_Bo), round(_y_Bo)]
                    order.append('{}_O'.format(name))
        order.append(order[0])
        C4['order'] = order
        self.outline['C4'] = C4
        for key in boxes_copy.keys():
            boxes[key] = boxes_copy[key]
        return boxes

class Detector_JingZhui_CS(torch.nn.Module):
    def __init__(self):
        super(Detector_JingZhui_CS, self).__init__()
        self.tf = transforms.Compose([transforms.ToTensor()])
        self.net = MainNet()
        self.net.cuda().eval()
        self.net.load_state_dict(torch.load(r'./weight_jingzhui/weights_resnet_new.pt'))
        self.dics_CS = {0:'CS1',1:'CS2',2:'CS3',3:'CS4',4:'CS5',5:'CS6'}

    def angle_counting(self,vector1, vector2):  # 计算两直线夹角
        cos = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)+1e-6)
        angle = np.arccos(cos)
        angle = angle / np.pi * 180.
        return angle

    def forward(self,path,out_jingzhui,outline_jingzhui,box):

        boxes = {}
        boxes['point1'] = [box[0], box[1]]
        boxes['point2'] = [box[2], box[1]]
        boxes['point3'] = [box[2], box[3]]
        boxes['point4'] = [box[0], box[3]]

        out_jingzhui['C2sp'] = outline_jingzhui['C2']['C2_10']

        C2sp = np.array(out_jingzhui['C2sp'])
        C4lp = np.array(out_jingzhui['C4lp'])

        angle = self.angle_counting(C2sp - C4lp, np.array([1, 0]))

        img = cv2.imread(path)
        h, w = img.shape[:2]
        roate = 90 - angle
        M1 = cv2.getRotationMatrix2D((w / 2, h / 2), roate, 1)

        new_w = int(h * fabs(sin(radians(roate))) + w * fabs(cos(radians(roate))))
        new_h = int(w * fabs(sin(radians(roate))) + h * fabs(cos(radians(roate))))

        M1[0, 2] += (new_w - w) / 2
        M1[1, 2] += (new_h - h) / 2

        for p_str in out_jingzhui:
            p = np.array(out_jingzhui[p_str])
            _p = np.append(p, np.array([1.]))

            _p = np.dot(M1, _p.T).T
            out_jingzhui[p_str] = _p.tolist()

        for p_str in boxes:
            p = np.array(boxes[p_str])
            _p = np.append(p, np.array([1.]))

            _p = np.dot(M1, _p.T).T
            boxes[p_str] = _p.tolist()

        xs = [boxes['point1'][0], boxes['point2'][0], boxes['point3'][0], boxes['point4'][0]]
        ys = [boxes['point1'][1], boxes['point2'][1], boxes['point3'][1], boxes['point4'][1]]
        min_x = int(min(xs))
        max_x = int(max(xs))
        min_y = int(min(ys))
        max_y = int(max(ys))
        img = cv2.warpAffine(img, M1, (new_w, new_h))
        allimg = img[min_y:max_y, min_x:max_x, :]
        # allimg = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        allimg = allimg[:,:,::-1]
        image = Image.fromarray(allimg,'RGB')

        img = Image.new('RGB', (max(image.size), max(image.size)), (0, 0, 0))
        img.paste(image, (0, 0))

        image = img.resize((112, 112))
        image = self.tf(image).cuda()
        # print(torch.max(image))
        image = image.unsqueeze(0)
        with torch.no_grad():
            outputs = self.net(image.cuda())
        out = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(out, 1)
        result = self.dics_CS[pred.item()]
        return result