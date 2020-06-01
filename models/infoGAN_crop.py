from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .base import Conv3d, ConvTranspose3d, ConvT3d
from .base import DropoutLinear

b = torch.ones(1).cuda()

class cropGenerator(nn.Module):
    def __init__(self, FG):
        super(cropGenerator, self).__init__()
        self.input_dim = FG.z
        self.input_size = 79
        self.discrete_code = FG.d_code # categorical distribution(i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim+self.discrete_code+self.continuous_code, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256*2*3*2),
            nn.BatchNorm1d(256*2*3*2),
            nn.ReLU(True))

        self.layer1 = ConvT3d(256, 128, 3, 2, 0, output_padding=1)
        self.layer2 = ConvT3d(128, 64, 3, 2, 0, output_padding=1)
        self.layer3 = ConvT3d(64, 32, 3, 2, 0, output_padding=1)
        self.layer4 = ConvT3d(32, 16, 3, 2, 0, output_padding=1)
        self.layer5 = ConvT3d(16, 1, 3, 1, 0, output_padding=0,
                              batch_norm=False, activation=nn.Tanh())
        # self.layer6 = ConvT3d(16, 1, 1, 1, 0, output_padding=0,
        #                       batch_norm=False, activation=nn.Tanh())

    def forward(self, x, cont_code, dist_code):
        #print('G running')
        x = torch.cat([x, cont_code, dist_code],1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1,256,2,3,2)
        x = self.layer1(x)
        # print('1 :',x.shape)
        x = self.layer2(x)
        # print('2 :',x.shape)
        x = self.layer3(x)
        # print('3 :',x.shape)
        x = self.layer4(x)
        # print('4 :',x.shape)
        x = self.layer5(x)
        # print('5 :',x.shape)
        # x = self.layer6(x)
        # print('6 :',x.shape)
        return x


class cropDiscriminator(nn.Module):
    def __init__(self, FG):
        super(cropDiscriminator, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.input_size = 79
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.layer1 = Conv3d(1, 16, 3, 2, 1, batch_norm=False, activation=self.activation)
        self.layer2 = Conv3d(16, 32, 3, 2, 1, activation=self.activation)
        self.layer3 = Conv3d(32, 64, 3, 2, 1, activation=self.activation)
        self.layer4 = Conv3d(64, 128, 3, 2, 1, activation=self.activation)
        self.layer5 = Conv3d(128, 256, 3, 2, 1, activation=self.activation)

        self.fc1 = nn.Sequential(
            nn.Linear(256*2*3*2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc2 = nn.Linear(512,self.output_dim+self.continuous_code+self.discrete_code)

    def forward(self, x):
        if x.shape[0] == 1:
            return b, b, b
        else :
            # print('D running')
            # print('i :',x.shape)
            x = self.layer1(x)
            # print('1 :',x.shape)
            x = self.layer2(x)
            # print('2 :',x.shape)
            x = self.layer3(x)
            # print('3 :',x.shape)
            x = self.layer4(x)
            # print('4 :',x.shape)
            x = self.layer5(x)
            # print('5 :',x.shape)
            #exit()
            x = x.view(-1, 256*2*3*2)

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
            Conv3d(1, 16, 3, 2, padding=1),
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
        self.avgpool = nn.AvgPool3d((4,4,4))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        #print(x.shape)
        x = self.stem(x)
        #print(x.shape)
        x = self.layer1(x)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
