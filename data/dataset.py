import numpy as np
from torchvision import transforms
from PIL import Image,ImageDraw
from torch.utils.data import Dataset,DataLoader
from data.autoaugment import ImageNetPolicy
import torch
import json


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


tf = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):

    def __init__(self, dirs):
        self.dataset = []
        self.policy = ImageNetPolicy()
        f = open(dirs,'r')
        for lines in f.readlines():
            line = lines.strip().split(',')

            file_name = line[0]
            jpg_name = file_name.replace('labels', 'pics').replace('json', 'jpg')
            self.dataset.append((jpg_name, file_name))




    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        lines = self.dataset[index]
        img = Image.open(lines[0])



        img_data = tf(img)


        f = open(lines[1])
        d = f.read()
        e = json.loads(d)
        label = e['ps']
        label = torch.FloatTensor(np.array(label,dtype='float32')/416)
        conf = e['mask']
        conf = torch.LongTensor(np.array(conf,dtype='int16'))


        return  img_data,label,conf









