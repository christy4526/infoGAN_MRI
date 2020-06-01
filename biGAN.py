from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, FG, output_dim=1):
        super(Generator, self).__init__()
        self.input_dim = FG.z
        self.output_dim = len(FG.d_code)
        self.d_code = FG.d_code # categorical distribution(i.e. label)
        self.c_code = FG.c_code # gaussian distribution(e.g. rotation, thickness)
        self.axis = FG.axis
        self.isize = FG.isize
        self.vsize = 0
        self.FG = FG

        if self.axis == 0 or self.axis == 2:
            self.vsize = 6*5
        elif self.axis ==1:
            self.vsize = 5*5

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim+self.c_code+len(self.d_code), 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128*self.vsize, bias=False),
            nn.BatchNorm1d(128*self.vsize),
            nn.ReLU(inplace=True))

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, 2, 2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(8, self.output_dim, 1, 1, 0),
            nn.Tanh())

        #self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, self.FG.std)
                m.bias.data.zero_()

    def forward(self, z, cont_code, dist_code, axis):
    #def forward(self, x, z, cont_code, axis):
        #print(z.shape, cont_code.shape, dist_code.shape)
        x = torch.cat([z, cont_code, dist_code], 1)
        #x = torch.cat([z, cont_code],1)
        x = self.fc1(x)
        x = self.fc2(x)

        if axis == 0:
            x = x.view(-1,128,6,5)
        elif axis ==1:
            x = x.view(-1,128,5,5)
        else :
            x = x.view(-1,128,5,6)

        x = self.layer1(x)
        #print('G 1:', x.shape)
        x = self.layer2(x)
        #print('G 2:', x.shape)
        x = self.layer3(x)
        #print('G 3:', x.shape)
        x = self.layer4(x)
        #print('G 4:', x.shape)
        x = self.layer5(x)
        #print('G 5:', x.shape)
        return x


class Q(nn.Module):
    """Inference net (encoder, real) Q(z|x)"""
    def __init__(self, FG):
        super(Q, self).__init__()
        self.FG = FG
        self.z = FG.z
        self.axis = FG.axis
        FG.num_channels = 1
        self.input_dim = len(FG.d_code)

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_dim, 8, 3, 1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1))
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
        #self.fc1 = nn.Linear(128*self.FG.batch_size*5*4, 128)
        #self.fc2 = nn.Linear(128, FG.batch_size)
        #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, FG.std)
                m.bias.data.zero_()

    def forward(self, x):
        if x.shape[0] == 1:
            return b, b
        else :
            x = self.stem(x)
            # print('Q 1:', x.shape)
            x = self.layer1(x)
            # print('Q 1:', x.shape)
            x = self.layer2(x)
            # print('Q 2:', x.shape)
            x = self.layer3(x)
            # print('Q 3:', x.shape)
            x = self.avgpool(x)
            # print('Q av:', x.shape) #[b, 128, 1, 1]
            x = x.view(x.shape[0], -1)
            # print('Q v:', x.shape)
            #x = self.fc1(x)
            #x = self.fc2(x)
            # print('Q fc:', x.shape)

            return x


class AP(nn.Module):
    """Alzheimer predictor net AP(z)"""
    def __init__(self, FG):
        super(AP, self).__init__()
        self.FG = FG
        self.z = FG.z
        self.axis = FG.axis
        FG.num_channels = 1

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, 1),
            nn.MaxPool1d(2, 2))
        self.layer2 = nn.Conv1d(8, 16, 3, 1, 1)
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.MaxPool1d(2, 2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.MaxPool1d(2, 2))
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, 1, 1, 0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
        #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, self.FG.std)
                m.bias.data.zero_()

    def forward(self, z):
        #print(type(z))
        x = z.unsqueeze(1)
        # print(type(x))
        # print('AP i: ', z.shape)
        x = self.layer1(x)
        # print('AP1: ', x.shape)
        x = self.layer2(x)
        # print('AP2: ', x.shape)
        x = self.layer3(x)
        # print('AP3: ', x.shape)
        x = self.layer4(x)
        # print('AP4: ', x.shape)
        x = self.layer5(x)
        # print('AP5: ', x.shape)
        x = self.avgpool(x)
        # print('APa: ', x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x, 1)
        return x


class Discriminator(nn.Module):
    """
    Discriminator net D(x, z)
    """
    def __init__(self, FG):
        super(Discriminator, self).__init__()
        FG.num_channels = 1
        self.input_dim = len(FG.d_code)
        self.isize = 79*95
        self.FG = FG
        self.output_dim = 1
        self.d_code = FG.d_code   # categorical distribution (i.e. label)
        self.c_code = FG.c_code # gaussian distribution (e.g. rotation, thickness)


        self.layer1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 8, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 32, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))

        self.fc1 = nn.Sequential(
            nn.Linear(386, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc2 = nn.Linear(256, self.output_dim+self.c_code+len(self.d_code))
        #self.fc2 = nn.Linear(256,self.output_dim+self.continuous_code)
        #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, self.FG.std)
                m.bias.data.zero_()

    def forward(self, x, ap, z):
        x = self.layer1(x)
        # print('D l1:', x.shape)
        x = self.layer2(x)
        # print('D l2:', x.shape)
        x = self.layer3(x)
        # print('D l3:', x.shape)
        x = self.layer4(x)
        # print('D l4:', x.shape)
        x = self.layer5(x)
        # print('D l5:', x.shape, ap.shape, z.shape)
        x = x.view(x.shape[0], -1)
        #print('x :', x.shape, ap.shape, z.shape)
        #print('cat:', torch.cat((x, z, ap),1).shape)
        output = torch.cat((x, ap, z), 1) #[20, 386]
        output = self.fc1(output)
        # print('D f1 :', output.shape)
        output = self.fc2(output)
        #print('D f2 :', output.shape)

        output = F.sigmoid(x[:, self.output_dim])
        cont = x[:, self.output_dim:self.output_dim+self.c_code]
        #disc = x[:, self.output_dim+self.c_code:]
        return output, cont
        #return nn.functional.sigmoid(outout.view(-1))

# resnet block with reflect padding
# class resnet_block(nn.Module):
#     def __init__(self, FG, in_channel, out_channel, kernel, stride, padding):
#         super(resnet_block, self).__init__()
#         self.std = FG.std
#         self.downsample = None
#         self.activation = nn.ReLU(inplace=True)
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel, stride=stride, padding=padding),
#             nn.ReLU(inplace=True))
#         self.conv2 = nn.Conv2d(out_channel, out_channel, kernel, stride, padding)
#
#         if stride != 1 or in_channel != out_channel:
#             self.downsample = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
#
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.normal_(0.0, self.std)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         out = self.activation(out)
#         return out
#
#
# class Generator(nn.Module):
#     """
#     Generator net (decoder, fake) G(x|z)
#     """
#     def __init__(self, FG):
#         super(Generator, self).__init__()
#         self.FG = FG
#         FG.seq_len = 79*95
#         FG.num_channels = 1
#
#         self.fc = nn.Linear(FG.z+FG.c_code,79*95*FG.num_hidden)
#             # input dim: num_hidden x seq_len
#         self.layer1 = resnet_block(FG,8, 8, 3, 1, 1)
#         self.layer2 = resnet_block(FG,8, 16, 3, 1, 1)
#         self.layer3 = resnet_block(FG,16, 32, 3, 1, 1)
#         self.layer4 = resnet_block(FG,32, 64, 3, 1, 1)
#         self.layer5 = resnet_block(FG,64, 64, 3, 1, 1)
#         self.layer6 = nn.Conv2d(64, 1, 1)
#         # out dim: num_channels x seq_len
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0.0, self.FG.std)
#                 m.bias.data.zero_()
#
#     def forward(self, z, c):
#         x = self.fc(torch.cat([z, c],1))
#         print('g fc:', x.shape)
#         x = x.view(-1, self.FG.num_hidden, 95,79)
#         print('g vi:', x.shape)
#         x = self.layer1(x)
#         print('g l1:', x.shape)
#         x = self.layer2(x)
#         print('g l2:', x.shape)
#         x = self.layer3(x)
#         print('g l3:', x.shape)
#         x = self.layer4(x)
#         print('g l4:', x.shape)
#         x = self.layer5(x)
#         print('g l5:', x.shape)
#         x = self.layer6(x)
#         print('g l6:', x.shape)
#         #return nn.functional.softmax(x,dim=1)
#         return x
#
# class Q(nn.Module):
#     """
#     Inference net (encoder, real) Q(z|x)
#     """
#     def __init__(self, FG):
#         super(Q, self).__init__()
#         self.FG = FG
#         FG.num_channels = 1
#         FG.num_hidden = 8
#         FG.seq_len = 70
#         # input dim: num_channels x seq_len
#         self.layer1 = nn.Conv2d(1, out_channels=8, kernel_size=1)
#         # state dim: num_hidden x seq_len
#         self.layer2 = resnet_block(FG,8, 8, 5, 1, 2)
#         self.layer3 = resnet_block(FG,8, 16, 5, 1, 2)
#         self.layer4 = resnet_block(FG,16, 32, 5, 1, 2)
#         self.layer5 = resnet_block(FG,32, 64, 5, 1, 2)
#         self.layer6 = resnet_block(FG,64, 64, 5, 1, 2)
#
#         self.fc = nn.Linear(79*95*64, FG.z)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0.0, self.FG.std)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x = self.layer1(x)
#         print('Q l1:', x.shape)
#         x = self.layer2(x)
#         print('Q l2:', x.shape)
#         x = self.layer3(x)
#         print('Q l3:', x.shape)
#         x = self.layer4(x)
#         print('Q l4:', x.shape)
#         x = self.layer5(x)
#         print('Q l5:', x.shape)
#         x = self.layer6(x)
#         print('Q l6:', x.shape)
#         x = x.view(x.shape[0],64*79*95)
#         print('Q vi:', x.shape)
#         x = self.fc(x)
#         print('Q fc:', x.shape)
#
#         return x
#
#
#
# class Discriminator(nn.Module):
#     """
#     Discriminator net D(x, z)
#     """
#     def __init__(self, FG):
#         super(Discriminator, self).__init__()
#         self.FG = FG
#         FG.num_channels = 1
#         FG.num_hidden = 8
#         FG.seq_len = 79*95
#
#         self.conv = nn.Conv2d(FG.seq_len,1,1)
#         #self.fc_z = nn.Linear(FG.z,FG.num_hidden*FG.seq_len//2)
#         #self.conv_x = nn.Conv2d(FG.num_channels+1,FG.num_hidden//2,1)
#         self.fc_z = nn.Linear(FG.z,79*95*8)
#         self.conv_x = nn.Conv1d(1,8,1)
#
#         # input dim: num_hidden x seq_len
#         self.layer1 = resnet_block(FG,8, 8, 5, 1, 2)
#         self.layer2 = resnet_block(FG,8, 16, 5, 1, 2)
#         self.layer3 = resnet_block(FG,16, 32, 5, 1, 2)
#         self.layer4 = resnet_block(FG,32, 64, 5, 1, 2)
#         self.layer5 = resnet_block(FG,64, 64, 5, 1, 2)
#         self.layer6 = nn.Conv2d(64, 1, 1, stride=1, bias=True)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0.0, self.FG.std)
#                 m.bias.data.zero_()
#
#     def forward(self, x, z):
#         output_z = self.fc_z(z)   # z : torch.Size([20, 2])
#         print('D z:', output_z.shape)
#         output_x = self.conv_x(x) # x : torch.Size([20, 7505])
#         print('D x:', output_x.shape)
#
#         x = self.layer1(torch.cat((output_x, output_z.view(-1,self.FG.num_hidden//2,self.FG.seq_len)), 1))
#         print('D l1:', x.shape)
#         x = self.layer2(x)
#         print('D l2:', x.shape)
#         x = self.layer3(x)
#         print('D l3:', x.shape)
#         x = self.layer4(x)
#         print('D l4:', x.shape)
#         x = self.layer5(x)
#         print('D l5:', x.shape)
#         x = self.layer6(x)
#         print('D l6:', x.shape)
#
#         x = x.view(-1,self.FG.seq_len,1)
#         print('D vi:', x.shape)
#         x = self.conv(x)
#         print('D co:', x.shape)
#         return nn.functional.sigmoid(x.view(-1))
#
#
# class AP(nn.Module):
#     """
#     branchPoint predictor net BP(z)
#     """
#     def __init__(self, FG):
#         super(AP, self).__init__()
#         self.FG = FG
#         FG.num_channels = 1
#
#         self.fc1 = nn.Linear(FG.z, 79*95*8)
#         # input dim: num_hidden x seq_len
#         self.stem = nn.Sequential(
#             nn.Conv2d(8, 16, 3, 2, padding=1),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 16, 3, 1, 1))
#         self.layer1 = resnet_block(FG, 16, 32, 3, 1, 1)
#         self.layer2 = resnet_block(FG, 32, 64, 3, 1, 1)
#         self.layer3 = resnet_block(FG, 64, 64, 3, 1, 1)
#         self.layer4 = resnet_block(FG, 64, 64, 3, 1, 1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc2 = nn.Linear(1280, self.FG.d_code)
#         # out dim: num_channels x 1
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0.0, self.FG.std)
#                 m.bias.data.zero_()
#
#     def forward(self, z):
#         x = self.fc1(z)
#         print('AP f1:', x.shape)
#         x = x.view(-1, 8, 95, 79)
#         print('AP vi:', x.shape)
#
#         x = self.stem(x)
#         print('AP st:', x.shape)
#         x = self.layer1(x)
#         print('AP l1:', x.shape)
#         x = self.layer2(x)
#         print('AP l2:', x.shape)
#         x = self.layer3(x)
#         print('AP l3:', x.shape)
#         x = self.layer4(x)
#         print('AP l4:', x.shape)
#         #x = self.layer5(x)
#         #print('AP l5:', x.shape)
#         x = self.avgpool(x)
#         print('AP l6:', x.shape)
#         x = x.view(x.size(0), -1)
#         print('AP vi:', x.shape)
#         x = self.fc2(x)
#         print('AP f2:', x.shape)
#         x = torch.nn.functional.softmax(x, 1)
#         return x
