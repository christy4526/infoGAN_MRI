from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .base import Conv3d, ConvTranspose3d, ConvT3d

b = torch.ones(1).cuda()

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dowmsample = None

        self.conv1 = Conv3d(in_channels, out_channels, 3, stride, padding=1,
                            activation = self.activation)
        self.conv2 = Conv3d(out_channels, out_channels, 3, 1, 1, activation=None)

        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv3d(in_channels, out_channels, 1, stride, activation=None)

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

        self.convT1 = Conv3d(in_channels, in_channels, 3, 1, 1)
        self.convT2 = ConvT3d(in_channels, out_channels, 3, 2, 1,
                              output_padding, activation=None)
        self.activation = nn.ReLU(inplace=True)
        self.upsample = None
        if stride != 1 or in_channels != out_channels:
            self.upsample = ConvT3d(in_channels, out_channels, 1, stride, 0,
                                  output_padding, activation=None)

    def forward(self, x):
        residual = x
        out = self.convT1(x)
        out = self.convT2(out)

        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.activation(out)

        return out


class ResGenerator(nn.Module):
    def __init__(self, FG):
        super(ResGenerator, self).__init__()
        self.input_dim = FG.z
        self.input_size = 79
        self.discrete_code = FG.d_code # categorical distribution(i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim+self.discrete_code+self.continuous_code, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256*5*6*5),
            nn.BatchNorm1d(256*5*6*5),
            nn.ReLU(True))

        self.layer1 = ResNetBlockT(256, 128, 2)
        self.layer2 = ResNetBlockT(128, 64, 2)
        self.layer3 = ResNetBlockT(64, 32, 2)
        self.layer4 = ResNetBlockT(32, 16, 2, output_padding=0)
        self.layer5 = ConvT3d(16, 1, 1, 1, 0, batch_norm=False, activation=nn.Tanh())

    def forward(self, x, cont_code, dist_code):
        x = torch.cat([x, cont_code, dist_code],1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1,256,5,6,5)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x



class ResDiscriminator(nn.Module):
    def __init__(self, FG):
        super(ResDiscriminator, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.input_size = 79
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)

        self.layer1 = Conv3d(1, 16, 3, 2, 1, batch_norm=False, activation=nn.LeakyReLU(0.2, inplace=True))
        #self.layer1 = ResNetBlock(1, 16, 3, 2, 1, batch_norm=False)
        self.layer2 = ResNetBlock(16, 32, 2)
        self.layer3 = ResNetBlock(32, 64, 2)
        self.layer4 = ResNetBlock(64, 128, 2)
        self.layer5 = ResNetBlock(128, 256, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(256*3*3*3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(512,self.output_dim+self.continuous_code+self.discrete_code),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        if x.shape[0] == 1:
            return b, b, b
        else :
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = x.view(-1, 256*3*3*3)

            x = self.fc1(x)
            x = self.fc2(x)

            a = F.sigmoid(x[:, self.output_dim])
            b = x[:, self.output_dim:self.output_dim+self.continuous_code]
            c = x[:, self.output_dim+self.continuous_code:]

            return a,b,c


class Plane(nn.Module):
    def __init__(self, num_classes):
        super(Plane, self).__init__()

        self.stem = nn.Sequential(
            Conv3d(1, 16, 3, 2, 1),
            nn.MaxPool3d(2, 2),
            Conv3d(16, 16, 3, 1, 1))
        self.layer1 = nn.Sequential(
            Conv3d(16, 32, 3, 1, 1),
            nn.MaxPool3d(2, 2))
        self.layer2 = nn.Sequential(
            Conv3d(32, 64, 3, 1, 1),
            nn.MaxPool3d(2, 2))
        self.layer3 = nn.Sequential(
            Conv3d(64, 64, 3, 1, 1),
            Conv3d(64, 128, 1, 1, 0))
        self.avgpool = nn.AvgPool3d((4,5,3))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
