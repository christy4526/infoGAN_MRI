from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

b = torch.ones(1).cuda()

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = None

        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
                nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, 1, padding)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)

        return out


class ResNetBlockT(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, output_padding=1):
        super(ResNetBlockT, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.upsample = None

        self.convT1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(True))
        self.convT2 = nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1,
                              output_padding=output_padding)

        if stride != 1 or in_channels != out_channels:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 1, stride, 0,
                                  output_padding=output_padding)

    def forward(self, x):
        residual = x
        out = self.convT1(x)
        out = self.convT2(out)

        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.activation(out)

        return out


class ResGenerator2D(nn.Module):
    def __init__(self, FG, output_dim=1):
        super(ResGenerator2D, self).__init__()
        self.input_dim = FG.z
        self.output_dim = output_dim
        self.discrete_code = FG.d_code # categorical distribution(i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)
        self.axis = FG.axis
        self.activation = FG.activation

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim+self.continuous_code, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True))
        if self.axis == 0 or self.axis == 2:
            size = 6*5
        elif self.axis ==1:
            size = 5*5
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256*size),
            nn.BatchNorm1d(256*size),
            nn.ReLU(True))

        self.layer1 = ResNetBlockT(256, 128, 2)
        self.layer2 = ResNetBlockT(128, 64, 2)
        self.layer3 = ResNetBlockT(64, 32, 2)
        self.layer4 = ResNetBlockT(32, 16, 2, output_padding=0)
        self.layer5 = ResNetBlockT(16, 8, 2, output_padding=0)
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(8, self.output_dim, 1, 1, 0),
            nn.Tanh())

    def forward(self, x, cont_code, axis):
        x = torch.cat([x, cont_code],1)
        x = self.fc1(x)
        # print('f1:',x.shape)
        x = self.fc2(x)
        # print('f2:',x.shape)
        if axis == 0:
            x = x.view(-1,256,6,5)
        elif axis ==1:
            x = x.view(-1,256,5,5)
        else :
            x = x.view(-1,256,5,6)
        x = self.layer1(x)
        # print('1:',x.shape)
        x = self.layer2(x)
        # print('2:',x.shape)
        x = self.layer3(x)
        # print('3:',x.shape)
        x = self.layer4(x)
        # print('4:',x.shape)
        x = self.layer5(x)
        # print('5:',x.shape)
        x = self.layer6(x)
        # print('6:',x.shape)
        return x



class ResDiscriminator2D(nn.Module):
    def __init__(self, FG, input_dim=1, output_dim=1):
        super(ResDiscriminator2D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)
        self.axis = FG.axis

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 8, 3, 2, 1),
            nn.LeakyReLU(0.2))
        #self.layer1 = ResNetBlock(1, 16, 3, 2, 1, batch_norm=False)
        self.layer2 = ResNetBlock(8, 16, 3, 2)
        self.layer3 = ResNetBlock(16, 32, 3, 2)
        self.layer4 = ResNetBlock(32, 64, 3, 2)
        self.layer5 = ResNetBlock(64, 128, 3, 2)
        self.layer6 = ResNetBlock(128, 256, 1, 1, 0)

        if self.axis == 0 or self.axis ==2:
            self.fc1 = nn.Sequential(
                nn.Linear(256*6*5, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True))
        elif self.axis == 1:
            self.fc1 = nn.Sequential(
                nn.Linear(256*5*5, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2, inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(512,self.output_dim+self.continuous_code),
            # nn.Sigmoid(),
        )

    def forward(self, x, axis):
        if x.shape[0] == 1:
            return b, b, b
        else :
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            if axis == 0 or axis == 2:
                x = x.view(-1, 256*6*5)
            elif axis == 1:
                x = x.view(-1, 256*5*5)

            x = self.fc1(x)
            x = self.fc2(x)

            a = F.sigmoid(x[:, self.output_dim])
            cont = x[:, self.output_dim:]
            return a,cont


# class ResGenerator2D(nn.Module):
#     def __init__(self, FG, output_dim=1):
#         super(ResGenerator2D, self).__init__()
#         self.input_dim = FG.z
#         self.output_dim = output_dim
#         self.discrete_code = FG.d_code # categorical distribution(i.e. label)
#         self.continuous_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)
#         self.axis = FG.axis
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(self.input_dim+self.discrete_code+self.continuous_code, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(True))
#         if self.axis == 0 or self.axis == 2:
#             size = 6*5
#         elif self.axis ==1:
#             size = 5*5
#         self.fc2 = nn.Sequential(
#             nn.Linear(512, 256*size),
#             nn.BatchNorm1d(256*size),
#             nn.ReLU(True))
#
#         self.layer1 = ResNetBlockT(256, 128, 2)
#         self.layer2 = ResNetBlockT(128, 64, 2)
#         self.layer3 = ResNetBlockT(64, 32, 2)
#         self.layer4 = ResNetBlockT(32, 16, 2, output_padding=0)
#         self.layer5 = nn.Sequential(
#             nn.ConvTranspose2d(16, self.output_dim, 1, 1, 0),
#             nn.Tanh())
#
#     def forward(self, x, cont_code, dist_code, axis):
#         x = torch.cat([x, cont_code, dist_code],1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         if axis == 0:
#             x = x.view(-1,256,6,5)
#         elif axis ==1:
#             x = x.view(-1,256,5,5)
#         else :
#             x = x.view(-1,256,5,6)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x
#
#
#
# class ResDiscriminator2D(nn.Module):
#     def __init__(self, FG, input_dim=1, output_dim=1):
#         super(ResDiscriminator2D, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
#         self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(self.input_dim, 16, 3, 2, 1),
#             nn.LeakyReLU(0.2))
#         #self.layer1 = ResNetBlock(1, 16, 3, 2, 1, batch_norm=False)
#         self.layer2 = ResNetBlock(16, 32, 2)
#         self.layer3 = ResNetBlock(32, 64, 2)
#         self.layer4 = ResNetBlock(64, 128, 2)
#         self.layer5 = ResNetBlock(128, 256, 2)
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(256*3*3, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2, inplace=True))
#         self.fc2 = nn.Sequential(
#             nn.Linear(512,self.output_dim+self.continuous_code+self.discrete_code),
#             # nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         if x.shape[0] == 1:
#             return b, b, b
#         else :
#             x = self.layer1(x)
#             x = self.layer2(x)
#             x = self.layer3(x)
#             x = self.layer4(x)
#             x = self.layer5(x)
#             x = x.view(-1, 256*3*3)
#
#             x = self.fc1(x)
#             x = self.fc2(x)
#
#             a = F.sigmoid(x[:, self.output_dim])
#             b = x[:, self.output_dim:self.output_dim+self.continuous_code]
#             c = x[:, self.output_dim+self.continuous_code:]
#
#             return a,b,c
