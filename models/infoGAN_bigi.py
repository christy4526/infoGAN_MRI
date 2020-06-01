from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

b = torch.ones(1).cuda()
class bigiGenerator(nn.Module):
    def __init__(self, FG, output_dim=1):
        super(bigiGenerator, self).__init__()
        self.input_dim = FG.z
        self.output_dim = output_dim
        self.discrete_code = FG.d_code # categorical distribution(i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)
        self.axis = FG.axis
        self.isize = FG.isize
        self.kernel = 4
        self.vsize = 0

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim+self.continuous_code, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        if self.axis == 0 or self.axis ==2:
            self.vsize = 5*4
        elif self.axis == 1:
            self.vsize = 4*4

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256*self.vsize, bias=False),
            nn.BatchNorm1d(256*self.vsize),
            nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, self.kernel, 2, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, self.kernel, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, self.kernel, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, self.kernel, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, self.kernel, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(8, self.output_dim, 1, 1, 0),
            nn.Tanh())

    #def forward(self, x, cont_code, dist_code, axis):
    def forward(self, x, cont_code, axis):
        #x = torch.cat([x, cont_code, dist_code],1)
        x = torch.cat([x, cont_code],1)
        x = self.fc1(x)
        x = self.fc2(x)

        if axis == 0:
            x = x.view(-1,256,5,4)
        elif axis ==1:
            x = x.view(-1,256,4,4)
        else :
            x = x.view(-1,256,4,5)

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
        x = self.layer6(x)
        # print('G 6:', x.shape)
        return x


class bigiDiscriminator(nn.Module):
    def __init__(self, FG, input_dim=1, output_dim=1):
        super(bigiDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.discrete_code = FG.d_code   # categorical distribution (i.e. label)
        self.continuous_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)
        self.axis = FG.axis
        self.isize = FG.isize
        self.kernel = 4
        self.osize = 0

        if self.axis == 0 or self.axis ==2:
            self.osize = 5*4
        elif self.axis == 1:
            self.osize = 3*3

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 8, self.kernel, 2, 1),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, self.kernel, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, self.kernel, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, self.kernel, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, self.kernel, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, self.kernel, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc1 = nn.Sequential(
            nn.Linear(256*self.osize, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True))
        #self.fc2 = nn.Linear(512,self.output_dim+self.continuous_code+self.discrete_code)
        self.fc2 = nn.Linear(512,self.output_dim+self.continuous_code)

    def forward(self, x, axis):
        if x.shape[0] == 1:
            return b, b
        else :
            x = self.layer1(x)
            # print('D 1:', x.shape)
            x = self.layer2(x)
            # print('D 2:', x.shape)
            x = self.layer3(x)
            # print('D 3:', x.shape)
            x = self.layer4(x)
            # print('D 4:', x.shape)
            x = self.layer5(x)
            # print('D 5:', x.shape)
            x = self.layer6(x)
            # print('D 6:', x.shape)

            if axis == 0 or axis == 2:
                x = x.view(-1, 256*5*4)
            elif axis == 1:
                x = x.view(-1, 256*3*3)

            x = self.fc1(x)
            x = self.fc2(x)
            a = F.sigmoid(x[:, self.output_dim])
            #cont = x[:, self.output_dim:self.output_dim+self.continuous_code]
            #disc = x[:, self.output_dim+self.continuous_code:]
            #return a,cont,disc
            cont = x[:, self.output_dim:]
            return a, cont
