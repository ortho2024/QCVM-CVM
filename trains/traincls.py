# coding='utf-8'

import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset_cls import MyDataset





from models.resnet_cbam import MainNet


import shutil
import os


def train():
    epochs = 400
    net = MainNet()
    loss_mse_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-5)
    lr = 0.0001
    net = net.cuda()
    net.train()
    if os.path.exists('../weight/weights_cls.pt'):
        net.load_state_dict(torch.load(r'../weight/weights_cls.pt'))
        pass
    batch_size=128
    for epoch in range(epochs):
        print("[i]: {}".format(epoch))
        if epoch%100==1 and epoch>10:
            lr = lr*0.95
            opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        accuracy_tran = 0
        size_all = 0

        dataloader = torch.utils.data.DataLoader(MyDataset(r'label.txt'),
                                                 batch_size=batch_size,
                                                 shuffle=True, num_workers=0, pin_memory=True)
        for step,(image, label,) in enumerate(dataloader):
            # Forward and backward
            targets = label.view(-1).cuda()
            opt.zero_grad()
            outputs = net(image.cuda())
            loss = loss_mse_fn(outputs, targets)
            out = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(out, 1)
            accuracy_tran += sum(pred==targets).item()
            size_all = size_all + image.shape[0]
            loss.backward()
            opt.step()

            if step%10==0:
                print('[loss]: {}'.format(loss.item()))
            if step % 1000 == 99:
                torch.save(net.state_dict(), '../weight/weights_cls.pt')
                shutil.copy('../weight/weights_cls.pt', '../weight/weights_cls_copy.pt')
                print('save......')

        if epoch%1==0:
                print('epoch:' ,epoch,'loss:',loss.item())
                torch.save(net.state_dict(),'../weight/weights_cls.pt')
                shutil.copy('../weight/weights_cls.pt','../weight/weights_cls_copy.pt')
        print(accuracy_tran/(size_all))





if __name__ == "__main__":
    train()
