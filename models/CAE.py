from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torch.nn as nn


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

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
        self.layer6 = nn.Sequential(
            nn.Conv3d(256, 1, 3, 2, 0, bias=False),
            nn.ReLU(True))

        self.layer7 = nn.Linear(1, 256*5*6*5)
        self.layer8 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, 2, 0, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True))
        self.layer9 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 3, 2, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True))
        self.layer10 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, 2, 1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(True))
        self.layer11 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 2, 2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(True))
        self.layer12 = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 1, 1, 0, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.view(-1,256,5,6,5)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        return x
