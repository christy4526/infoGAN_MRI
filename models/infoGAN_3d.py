from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from .base import Conv3d, ConvTranspose3d, ConvT3d
from .base import DropoutLinear

b = torch.ones(1).cuda()

class Generator(nn.Module):
    def __init__(self, FG):
        super(Generator, self).__init__()
        self.input_dim = FG.z
        self.output_dim = 1
        self.input_size = 79
        self.discrete_code = FG.d_code # categorical distribution(i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)

        self.fc1 = nn.Sequential(
            #nn.Linear(self.input_dim+self.discrete_code+self.continuous_code, 128),
            nn.Linear(self.input_dim+self.continuous_code, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64*5*6*5),
            nn.BatchNorm1d(64*5*6*5),
            nn.ReLU(True))

        self.layer1 = ConvT3d(64, 64, 3, 2, 0, activation=nn.ReLU(inplace=True))
        self.layer2 = ConvT3d(64, 32, 3, 2, 1, activation=nn.ReLU(inplace=True))
        self.layer3 = ConvT3d(32, 16, 3, 2, 1, activation=nn.ReLU(inplace=True))
        self.layer4 = ConvT3d(16, 8, 3, 2, 2, activation=nn.ReLU(inplace=True))
        self.layer5 = Conv3d(8, 1, 3, 1, 1, batch_norm=False, activation=nn.Tanh())

        # self.fc1 = nn.Sequential(
        #     nn.Linear(self.input_dim+self.continuous_code, 256, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True))
        # self.fc2 = nn.Sequential(
        #     nn.Linear(256, 128*5*6*5, bias=False),
        #     nn.BatchNorm1d(128*5*6*5),
        #     nn.ReLU(inplace=True))
        #
        # self.layer1 = nn.Sequential(
        #     nn.ConvTranspose3d(128, 64, 3, 2, 0, bias=False),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True))
        # self.layer2 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, 3, 2, 1, bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True))
        # self.layer3 = nn.Sequential(
        #     nn.ConvTranspose3d(32, 16, 3, 2, 1, bias=False),
        #     nn.BatchNorm3d(16),
        #     nn.ReLU(inplace=True))
        # self.layer4 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, 2, 2, bias=False),
        #     nn.BatchNorm3d(8),
        #     nn.ReLU(inplace=True))
        # self.layer5 = nn.Sequential(
        #     nn.ConvTranspose3d(8, self.output_dim, 1, 1, 0),
        #     nn.Tanh())

    #def forward(self, x, cont_code, dist_code):
    def forward(self, x, cont_code):
        #print('G running')
        #x = torch.cat([x, cont_code, dist_code],1)
        x = torch.cat([x, cont_code],1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1,64,5,6,5)
        #x = x.view(-1,128,5,6,5)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, FG):
        super(Discriminator, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.input_size = 79
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)

        self.layer1 = Conv3d(1, 8, 3, 2, 1, batch_norm=False, activation=nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = Conv3d(8, 16, 3, 2, 1, activation=nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = Conv3d(16, 32, 3, 2, 1, activation=nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = Conv3d(32, 64, 3, 2, 1, activation=nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = Conv3d(64, 64, 3, 1, 1, activation=nn.LeakyReLU(0.2, inplace=True))

        self.fc1 = nn.Sequential(
            #nn.Linear(64*3*3*3, 128),
            nn.Linear(64*5*6*5, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True))
        #self.fc2 = nn.Linear(128,self.output_dim+self.continuous_code+self.discrete_code)
        self.fc2 = nn.Linear(128,self.output_dim+self.continuous_code)

        # self.layer1 = nn.Sequential(
        #     nn.Conv3d(self.input_dim, 8, 3, 2, 1),
        #     nn.LeakyReLU(0.2, inplace=True))
        # self.layer2 = nn.Sequential(
        #     nn.Conv3d(8, 16, 3, 2, 1, bias=False),
        #     nn.BatchNorm3d(16),
        #     nn.LeakyReLU(0.2, inplace=True))
        # self.layer3 = nn.Sequential(
        #     nn.Conv3d(16, 32, 3, 2, 1, bias=False),
        #     nn.BatchNorm3d(32),
        #     nn.LeakyReLU(0.2, inplace=True))
        # self.layer4 = nn.Sequential(
        #     nn.Conv3d(32, 64, 3, 2, 1, bias=False),
        #     nn.BatchNorm3d(64),
        #     nn.LeakyReLU(0.2, inplace=True))
        # self.layer5 = nn.Sequential(
        #     nn.Conv3d(64, 128, 3, 2, 1, bias=False),
        #     nn.BatchNorm3d(128),
        #     nn.LeakyReLU(0.2, inplace=True))
        #
        # self.fc1 = nn.Sequential(
        #     nn.Linear(128*3*3*3, 256, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2, inplace=True))
        # #self.fc2 = nn.Linear(256,self.output_dim+self.continuous_code+self.discrete_code)
        # self.fc2 = nn.Linear(256,self.output_dim+self.continuous_code)

    def forward(self, x):
        if x.shape[0] == 1:
            return b, b, b
        else :
            #print('D running')
            #print(x.shape)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            #print(x.shape)
            x = x.view(x.size(0), -1)

            x = self.fc1(x)
            x = self.fc2(x)

            a = F.sigmoid(x[:, self.output_dim])
            cont = x[:, self.output_dim:]
            #cont = x[:, self.output_dim:self.output_dim+self.continuous_code]
            #disc = x[:, self.output_dim+self.continuous_code:]
            #return a,cont,disc
            return a,cont


class Plane(nn.Module):
    def __init__(self, num_classes):
        super(Plane, self).__init__()

        self.stem = nn.Sequential(
            Conv3d(1, 16, 5, 2, padding=1),
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
        self.avgpool = nn.AdaptiveAvgPool3d(1)
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
