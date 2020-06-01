from __future__ import print_function, division, absolute_import
import os

import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import Compose
from transforms import RandomCrop, ToWoldCoordinateSystem, ToTensor, Crop
import nibabel as nib
from dataset import fold_split, ADNIDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom
import summary
from config import train_args, argument_report
from DCGAN import Generator, Discriminator
from cae import CAE

import itertools
import math
import time
from torch import optim
import numpy as np
from utils import print_model_parameters
import torchvision.utils as vutils
import torchvision.transforms as transforms
from IPython import display
outf = "result"
import torch.nn.parallel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def transform_preset(mode='random_crop'):
    transformer = None
    if mode == 'random_crop':
        transformer = Compose([
            ToWoldCoordinateSystem(),
            ToTensor()])

    return transformer


def make_dataloader(FG):
    x, y, train_idx, test_idx, ratio = fold_split(
        k=FG.fold, running_k=FG.running_fold, labels=FG.labels,
        subject_ids_path=os.path.join(FG.data_root, 'subject_ids.pkl'),
        diagnosis_path=os.path.join(FG.data_root, 'diagnosis.pkl'))
    # x = image, y=target
    trainset = ADNIDataset(FG.labels, FG.data_root, x[train_idx], y[train_idx],
                           transform_preset('random_crop'))
    validset = ADNIDataset(FG.labels, FG.data_root, x[test_idx], y[test_idx])

    trainloader = DataLoader(trainset, batch_size=FG.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=4)
    validloader = DataLoader(validset, batch_size=len(FG.devices),
                             shuffle=False, pin_memory=True,
                             num_workers=4)
    return trainloader, validloader


def main(FG):
    vis = Visdom(port=10001, env=str(FG.vis_env))
    vis.text(argument_report(FG, end='<br>'), win='config')
    FG.global_step=0

    cae = CAE().cuda()

    print_model_parameters(cae)
    #criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cae.parameters(), lr=FG.lr, betas=(0.5, 0.999))
    schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, FG.lr_gamma)
    printers = dict(
        loss = summary.Scalar(vis, 'loss', opts=dict(
            showlegend=True, title='loss', ytickmin=0, ytinkmax=2.0)),
        lr = summary.Scalar(vis, 'lr', opts=dict(
            showlegend=True, title='lr', ytickmin=0, ytinkmax=2.0)),
        input_printer = summary.Image3D(vis, 'input')
        output_printer = summary.Image3D(vis, 'output'))

    trainloader, validloader = make_dataloader(FG)

    z = 256
    batchSize = FG.batch_size
    imageSize = 64
    input = torch.FloatTensor(batchSize, 1, imageSize, imageSize, imageSize).cuda()
    noise = torch.FloatTensor(batchSize, z).cuda()
    fixed_noise = torch.FloatTensor(batchSize, z).normal_(0, 1).cuda()

    label = torch.FloatTensor(batchSize).cuda()
    real_label = 1
    fake_label = 0

    for epoch in range(FG.num_epoch):
        schedular.step()
        torch.set_grad_enabled(True)
        pbar = tqdm(total=len(trainloader), desc='Epoch {:>3}'.format(epoch))
        for i, data in enumerate(trainloader):
            real = data[0][0].cuda()

            output = cae(real)
            loss = criterion(output, real)
            loss.backward()
            optimizer.step()

            FG.global_step += 1
            printers['loss']('loss', FG.global_step/len(trainloader), loss)
            printers['input']('input', real)
            printers['output']('output', output/output.max())
            pbar.update()
        pbar.close()



if __name__ == '__main__':
    FG = train_args()
    main(FG)
    print()
