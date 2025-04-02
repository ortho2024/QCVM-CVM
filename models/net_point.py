import torch
from torch import nn



class CommonNorm(nn.Module):

    def __init__(self, dim, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.elementwise_affine = elementwise_affine

    def forward(self, input):
        mean = torch.mean(input, dim=self.dim, keepdim=True)
        var = torch.mean((input - mean) ** 2, dim=self.dim, keepdim=True)
        norm = (input - mean) / (var + 1e-5) ** 0.5
        if self.elementwise_affine:
            return self.alpha * norm + self.beta
        else:
            return norm
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()



        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        # print(**kwargs)
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

class MobileLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, multiple, stride=1):
        super(MobileLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            CommonNorm(1),
            torch.nn.LeakyReLU(0.1, True),
            torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            CommonNorm(1),
            torch.nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            return self.sub_module(x) + x
        else:
            return self.sub_module(x)








class MainNet(torch.nn.Module):

    def __init__(self,point):
        super(MainNet, self).__init__()
        self.coordconv = CoordConv(1024, 64, with_r=False, kernel_size=1, )

        self.trunk_52 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, 1, 1),  # 416
            torch.nn.Conv2d(8, 16, 1, 2, 0),  # 208

            MobileLayer(16, 32, 3),
            MobileLayer(32, 64, 1, 2),  # 104

            MobileLayer(64, 64, 3),
            MobileLayer(64, 64, 3),
            MobileLayer(64, 128, 1, 2),  # 52

            MobileLayer(128, 128, 3),
            MobileLayer(128, 128, 3),
            MobileLayer(128, 128, 3),
            MobileLayer(128, 256, 1, 2),  # 26

            MobileLayer(256, 256, 3),
            MobileLayer(256, 256, 3),
            MobileLayer(256, 256, 3),
            MobileLayer(256, 256, 3),
            MobileLayer(256, 512, 1, 2),  # 13

            MobileLayer(512, 512, 3),
            MobileLayer(512, 512, 3),
            MobileLayer(512, 512, 3),
            MobileLayer(512, 512, 3),
            MobileLayer(512, 1024, 1, 2),  # 13
            # nn.Conv2d(1024, 64, 1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
        )

        self.ouput_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LayerNorm(1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(256, 2*point),
        )

    def forward(self, x):
        # start_time = time.time()
        h_52 = self.trunk_52(x)
        h_52 = self.coordconv(h_52)
        h_52 = h_52.view(-1,7*7*64)
        out = self.ouput_layer(h_52)
        return out



