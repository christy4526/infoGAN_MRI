from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

b = torch.ones(1).cuda()

class Generator2D(nn.Module):
    def __init__(self, FG, output_dim=1):
        super(Generator2D, self).__init__()
        self.input_dim = FG.z
        self.discrete_code = FG.d_code # categorical distribution(i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)
        self.axis = FG.axis
        self.isize = FG.isize
        self.kernel = 4
        self.vsize = 0
        self.layer5 = None

        if self.isize == 157:
            if self.axis == 1:
                self.vsize = 4*4
            else:
                self.vsize = 5*4
            self.layer5 = nn.Sequential(
                nn.ConvTranspose2d(16, 1, 4, 2, 1),
                nn.Tanh())
        elif self.isize == 79:
            if self.axis == 0 or self.axis == 2:
                self.vsize = 6*5
            elif self.axis ==1:
                self.vsize = 5*5
            self.layer5 = nn.Sequential(
                nn.ConvTranspose2d(16, 1, 1, 1, 0),
                nn.Tanh())

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim+self.continuous_code, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256*self.vsize, bias=False),
            nn.BatchNorm1d(256*self.vsize),
            nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

    def forward(self, x, cont_code, axis):
        x = torch.cat([x, cont_code],1)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.isize == 157:
            if axis == 0:
                x = x.view(x.size(0),256,5,4)
            elif axis ==1:
                x = x.view(x.size(0),256,4,4)
            else :
                x = x.view(x.size(0),256,4,5)
        else:
            if axis == 0:
                x = x.view(x.size(0),256,6,5)
            elif axis ==1:
                x = x.view(x.size(0),256,5,5)
            else :
                x = x.view(x.size(0),256,5,6)

        x = self.layer1(x)
        # print('G 1:', x.shape)
        x = self.layer2(x)
        # print('G 2:', x.shape)
        x = self.layer3(x)
        # print('G 3:', x.shape)
        x = self.layer4(x)
        # print('G 4:', x.shape)
        x = self.layer5(x)
        # print('G 5:', x.shape)
        return x


class Discriminator2D(nn.Module):
    def __init__(self, FG, input_dim=1, output_dim=1):
        super(Discriminator2D, self).__init__()
        self.output_dim = output_dim
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)
        self.axis = FG.axis
        self.isize = FG.isize
        self.osize = 0

        if self.isize == 157:
            if self.axis == 1:
                self.osize = 4*4
            else:
                self.osize = 5*4
        else :
            self.osize = 3*3

        """               5 layers             """
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc1 = nn.Sequential(
            nn.Linear(256*self.osize, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc2 = nn.Linear(512,self.output_dim+self.continuous_code)

    def forward(self, x):
        if x.shape[0] == 1:
            return b, b
        else :
            # print(x.shape, type(x))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = x.view(x.size(0), -1)
            # print('D v:', x.shape)
            x = self.fc1(x)
            x = self.fc2(x)
            a = F.sigmoid(x[:, :self.output_dim])
            #cont = x[:, self.output_dim:self.output_dim+self.continuous_code]
            #disc = x[:, self.output_dim+self.continuous_code:]
            #return a,cont,disc
            cont = x[:, self.output_dim:]
            return a, cont


class Plane2D(nn.Module):
    def __init__(self, FG, num_classes):
        super(Plane2D, self).__init__()
        self.isize = FG.isize
        # self.stem = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, 2, padding=1),
        #     nn.MaxPool2d(2, 2),
        #     nn.Conv2d(16, 16, 3, 1, 1),
        #     nn.MaxPool2d(2, 2))
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, 1, 1))
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 1, 1, 0))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        # print('Cs: ', x.shape)
        x = self.layer1(x)
        # print('C1: ', x.shape)
        x = self.layer2(x)
        # print('C2: ', x.shape)
        x = self.layer3(x)
        #print('C3: ', x.shape)
        x = self.avgpool(x)
        #print('CA: ', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
