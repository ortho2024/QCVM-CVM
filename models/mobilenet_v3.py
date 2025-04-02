import torch
import time
import torch.onnx
import onnx



class MobileLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, multiple, stride=1):
        super(MobileLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.LeakyReLU(0.1, True),
            torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            return self.sub_module(x) + x
        else:
            return self.sub_module(x)


class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()
        self.pixelShuffle = torch.nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixelShuffle(x)


class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            MobileLayer(in_channels, out_channels, 3),
            # MobileLayer(out_channels, out_channels, 3),
        )

    def forward(self, x):
        return self.sub_module(x)


class MainNet(torch.nn.Module):

    def __init__(self,cls_num):
        super(MainNet, self).__init__()

        self.trunk_52 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),  # 416
            torch.nn.Conv2d(32, 64, 1, 2, 0),  # 208

            MobileLayer(64, 64, 3),
            MobileLayer(64, 128, 1, 2),  # 104

            MobileLayer(128, 128, 3),
            # MobileLayer(128, 128, 3),
            MobileLayer(128, 256, 1, 2),  # 52
        )

        self.trunk_26 = torch.nn.Sequential(
            MobileLayer(256, 256, 3),
            # MobileLayer(256, 256, 3),
            # MobileLayer(256, 256, 3),
            MobileLayer(256, 512, 1, 2),  # 26
        )

        self.trunk_13 = torch.nn.Sequential(
            MobileLayer(512, 512, 3),
            MobileLayer(512, 512, 3),
            # MobileLayer(512, 512, 3),
            # MobileLayer(512, 512, 3),
            MobileLayer(512, 1024, 1, 2),  # 13
        )

        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        self.detetion_13 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 3*(5+cls_num), 3, 1, 1)
        )

        self.up_26 = torch.nn.Sequential(
            MobileLayer(512, 512, 3),
            UpsampleLayer()
        )

        self.convset_26 = torch.nn.Sequential(
            ConvolutionalSet(640, 256)
        )

        self.detetion_26 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 3*(5+cls_num), 3, 1, 1)
        )

        self.up_52 = torch.nn.Sequential(
            MobileLayer(256, 256, 3),
            UpsampleLayer()
        )

        self.convset_52 = torch.nn.Sequential(
            ConvolutionalSet(320, 128)
        )

        self.detetion_52 = torch.nn.Sequential(
            MobileLayer(128, 128, 2),
            torch.nn.Conv2d(128, 3*(5+cls_num), 3, 1, 1)
        )

    def forward(self, x):
        # start_time = time.time()
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)

        # end_time = time.time()
        # print("........................",end_time - start_time)

        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52


