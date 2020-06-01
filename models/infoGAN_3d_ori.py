from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from config import train_args, argument_report
from .base import Conv3d, ConvTranspose3d, ConvT3d
from .base import DropoutLinear


b = torch.ones(1).cuda()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.ConvTranspose3d):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class oriGenerator(nn.Module):
    def __init__(self, FG):
        super(oGenerator, self).__init__()
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
            nn.BatchNorm3d(256),
            nn.ReLU(True))

        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, 2, 0, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 1, 1, 0, bias=False),
            nn.Tanh())
        #initialize_weights(self)

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


class oriDiscriminator(nn.Module):
    def __init__(self, FG):
        super(oDiscriminator, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.input_size = 79
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc1 = nn.Sequential(
            nn.Linear(256*3*3*3, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(512,self.output_dim+self.continuous_code+self.discrete_code),
            # nn.Sigmoid(),
        )
        #initialize_weights(self)

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
