# coding='utf-8'

import torch

import torch.nn as nn
import torch.optim as optim
import pylab as pl



from data.dataset import   MyDataset
from models.net_point import MainNet
import matplotlib.pylab as plt
import shutil
import os

import argparse

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default="100", type=int)
parser.add_argument("--lr", default="0.001", type=float)
args = parser.parse_args()
batch_size = 8
def train(point,name):


    net = MainNet(point)
    # net=nn.DataParallel(nets)
    loss_mse_fn = nn.MSELoss()

    # Optimizer and learning rate
    opt = optim.Adam(net.parameters(), lr=args.lr,weight_decay=1e-5)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr)




    net = net.cuda()
    if os.path.exists('../weight/weights_{}.pt'.format(name)):
        print(11111111111)
        # net.load_state_dict(torch.load(r'../weight/weights_{}.pt'.format(name)))


    lr = args.lr
    plt.figure(figsize=(8, 6), dpi=80)

    # 打开交互模式getOverlap_Line
    plt.ion()
    train_=[]
    val_ = []

    for epoch in range(args.epochs):
        print("[i]: {}".format(epoch))

        for dir in   ['train.txt','val.txt']:
            if dir =='train.txt':
                if epoch%40==3 and epoch>40:
                    lr = lr*0.95**epoch
                    opt = optim.Adam(net.parameters(), lr=lr,weight_decay=1e-5 )
                    # optimizer = optim.SGD(net.parameters(), lr=args.lr)



                dataloader = torch.utils.data.DataLoader(MyDataset(r'train.txt'.format(name)),
                                                         batch_size=batch_size,
                                                         shuffle=True, num_workers=0, pin_memory=True)
                loss_train = 0
                for step, samples in enumerate(dataloader):
                    images, targets,conf= samples[0].cuda(),samples[1].cuda(),samples[2]

                    # Forward and backward
                    opt.zero_grad()
                    outputs = net(images).view(-1,point,2)


                    loss = 0
                    regularization_loss = 0
                    for param in net.parameters():
                        regularization_loss += torch.sum(abs(param))
                    for i in range(point):
                        mask = conf[..., i] >0
                        obj_targets = targets[:,i,:][mask]
                        obj_outputs = outputs[:,i,:][mask]
                        loss1 = loss_mse_fn(obj_targets,obj_outputs)
                        loss +=loss1
                    loss_train +=loss.item()
                    # loss = loss + 0.001*regularization_loss

                    loss.backward()
                    opt.step()


                    if step%10==0:
                        # print(loss.item())
                        print('[loss]: {}'.format(loss.item()))
                    if step % 100 == 99:
                        torch.save(net.state_dict(), '../weight/weights_{}.pt'.format(name))
                        # torch.save(nets.state_dict(), '../weight/weights_.pt')
                        shutil.copy('../weight/weights_{}.pt'.format(name), '../weight/weights_copy_{}.pt'.format(name))
                        print('save......')

                if epoch%1==0:
                        print('epoch:' ,epoch,'loss:',loss.item())
                        # torch.save(net.state_dict(),'../weight/weights_Bri_Id2_{}.pt'.format(epoch))
                        # torch.save(nets.state_dict(), '../weight/weights_.pt')
                        # shutil.copy('../weight/weights_Bri_Id2.pt','../weight/weights_copy_Bri_Id2.pt')
                loss_train_mean = loss_train/step
                print(loss_train_mean)
                train_.append(loss_train_mean)

            else:
                dataloader = torch.utils.data.DataLoader(MyDataset(r'val.txt'.format(name)),
                                                         batch_size=batch_size,
                                                         shuffle=True, num_workers=0, pin_memory=True)
                loss_val=0
                net.eval()
                for step, samples in enumerate(dataloader):
                    images, targets, conf = samples[0].cuda(), samples[1].cuda(), samples[2]
                    opt.zero_grad()
                    with torch.no_grad():
                        outputs = net(images).view(-1, point, 2)
                    loss = 0
                    for i in range(point):
                        mask = conf[..., i] > 0
                        obj_targets = targets[:, i, :][mask]
                        obj_outputs = outputs[:, i, :][mask]
                        loss1 = loss_mse_fn(obj_targets, obj_outputs)
                        loss += loss1

                    loss_val +=loss.item()
                loss_val_mean = loss_val/step
                print('vsl_loss:',loss_val_mean,1111111)
                val_.append(loss_val_mean)




def A(point,name):
    loss_mse_fn = nn.MSELoss()
    net = MainNet(point).cuda()
    net.eval()
    net.load_state_dict(torch.load(r'../weight/weights_{}.pt'.format(name)))
    dataloader = torch.utils.data.DataLoader(MyDataset(r'test.txt'.format(name)),
                                             batch_size=32,
                                             shuffle=True, num_workers=0, pin_memory=True)
    loss_test = 0
    for step, samples in enumerate(dataloader):
        images, targets, conf = samples[0].cuda(), samples[1].cuda(), samples[2]

        with torch.no_grad():
            outputs = net(images).view(-1, point, 2)

        loss = 0
        for i in range(point):
            mask = conf[..., i] > 0
            obj_targets = targets[:, i, :][mask]
            obj_outputs = outputs[:, i, :][mask]
            loss1 = loss_mse_fn(obj_targets, obj_outputs)
            loss += loss1
            print(loss)
        loss_test  += loss.item()
    loss_test_mean = loss_test  / step
    print('test_loss:', loss_test_mean, 1111111)


if __name__ == "__main__":
    names = [[9,'C2'],[11,'C3'],[11,'C4'],]

    for name in names:
        print(name)
        train(name[0],name[1])
        A(name[0],name[1])

