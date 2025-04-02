import numpy as np
from torchvision import transforms
from PIL import Image,ImageDraw
from torch.utils.data import Dataset,DataLoader

import cv2
import os
import torch



tf = transforms.Compose([transforms.ToTensor()])

class MyDataset(Dataset):

    def __init__(self, dir):
        self.dataset = []
        self.labels = []
        f = open(dir,'r')
        for linecontent in f.readlines():

            linecontent = linecontent.strip().split()
            self.dataset.append(linecontent[0])
            self.labels.append([int(linecontent[1])])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        input_data = Image.open(data)
        input_data = tf(input_data)
        # img_data = img_data.view(-1,12)
        labels = np.array(self.labels[index])


        return input_data,labels




