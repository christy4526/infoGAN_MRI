from __future__ import absolute_import, division, print_function, unicode_literals
import os

# visualizes
from tqdm import tqdm
from visdom import Visdom
import time

# deep learning framework
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Lambda
from torch.utils.data import DataLoader
from models import dcGenerator, dcDiscriminator
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.datasets as dset
import torchvision.utils

# project packages
from config import biGAN_parser, argument_report
from dataset import ADNIDataset, ADNIDataset2D, Trainset, fold_split
from transforms import ToFloatTensor, Normalize
from utils import print_model_parameters, ScoreReport, SimpleTimer, loss_plot
from utils import save_checkpoint
from summary import Scalar, Image3D
import itertools
import numpy as np
import imageio
import shutil
import argparse

#from utils import logging
from torch.autograd import Variable
from sklearn import metrics
from ori_utils import logging


if __name__ == '__main__':
    FG = biGAN_parser()
    if FG.clean_ckpt:
        shutil.rmtree(FG.checkpoint_root)
    if not os.path.exists(FG.checkpoint_root):
        os.makedirs(FG.checkpoint_root, exist_ok=True)
    logger = logging.Logger(FG.checkpoint_root)
    FG.seed = 1
    torch.manual_seed(FG.seed)
    torch.cuda.manual_seed(FG.seed)
    cudnn.benchmark = True
    EPS = 1e-12

    vis = Visdom(port=FG.vis_port, env=str(FG.vis_env))
    vis.text(argument_report(FG, end='<br>'), win='config')

    save_dir = str(FG.vis_env)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # torch setting
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])
    timer = SimpleTimer()

    printers = dict(
        lr = Scalar(vis, 'lr', opts=dict(
            showlegend=True, title='lr', ytickmin=0, ytinkmax=2.0)),
        D_loss = Scalar(vis, 'D_loss', opts=dict(
            showlegend=True, title='D loss', ytickmin=0, ytinkmax=2.0)),
        G_loss = Scalar(vis, 'GE_loss', opts=dict(
            showlegend=True, title='GE loss', ytickmin=0, ytinkmax=10)),
        AC_loss = Scalar(vis, 'AC_loss', opts=dict(
            showlegend=True, title='AC loss', ytickmin=0, ytinkmax=10)),
        info_loss = Scalar(vis, 'info_loss', opts=dict(
            showlegend=True, title='info_loss', ytickmin=0, ytinkmax=10)),
        acc = Scalar(vis, 'Accuracy', opts=dict(
            showlegend=True, title='Accuracy', ytickmin=0, ytinkmax=2.0)),
        ofake = Scalar(vis, 'Fake output', opts=dict(
            showlegend=True, title='Fake output', ytickmin=0, ytinkmax=2.0)),
        oreal = Scalar(vis, 'Real output', opts=dict(
            showlegend=True, title='Real output', ytickmin=0, ytinkmax=2.0)),
        inputs = Image3D(vis, 'inputs'),
        fake = Image3D(vis, 'fake'),
        valid = Image3D(vis, 'valid'),
        outputs = Image3D(vis, 'outputs'),
        outputs2 = Image3D(vis, 'outputs2'))

    x, y = Trainset(FG)      # x = image, y=target
    # x, y, train_idx, test_idx, ratio = fold_split(FG)
    if FG.gm:
        transform=Compose([ToFloatTensor()])
    else :
        transform=Compose([ToFloatTensor(), Normalize(0.5,0.5)])

    trainset = ADNIDataset(FG, x, y, transform=transform)
    trainloader = DataLoader(trainset, batch_size=FG.batch_size,
                             shuffle=True, pin_memory=True)

    D = dcDiscriminator(FG).to('cuda:{}'.format(FG.devices[0]))  # discriminator net D(x, z)
    G = dcGenerator(FG.z_dim, FG.c_code, FG.axis).to('cuda:{}'.format(FG.devices[0]))  # generator net (decoder) G(x|z)

    if FG.load_ckpt:
        D.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'D.pth')))
        G.load_state_dict(torch.load(os.path.join(FG.checkpoint_root, save_dir, 'G.pth')))
    if len(FG.devices) != 1:
        G = torch.nn.DataParallel(G, FG.devices)
        D = torch.nn.DataParallel(D, FG.devices)

    optimizerD = optim.Adam(D.parameters(), lr=FG.lrD, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrG, betas=(0.5, 0.999))
    schedularD=torch.optim.lr_scheduler.ExponentialLR(optimizerD, FG.lr_gamma)
    schedularG=torch.optim.lr_scheduler.ExponentialLR(optimizerG, FG.lr_gamma)

    optimizerinfo = optim.Adam(itertools.chain(G.parameters(), E.parameters()),
                               lr=FG.lr_adam, betas=(FG.beta1, 0.999))
    schedularinfo = ExponentialLR(optimizerinfo, FG.lr_gamma)

    BCEWithLogitsLoss = nn.BCEWithLogitsLoss().to('cuda:{}'.format(FG.devices[0]))
    BCE_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
    MSE_loss = nn.MSELoss().to('cuda:{}'.format(FG.devices[0]))

    d_code = torch.tensor([0,1])
    c_code = FG.c_code  # gaussian distribution (e.g. rotation, thickness)
    fixed_z = torch.rand((FG.batch_size, FG.z_dim)).type(torch.FloatTensor).cuda(device, non_blocking=True)
    fixed_c = torch.from_numpy(np.random.uniform(-1, 1, size=(FG.batch_size,\
                           c_code))).type(torch.FloatTensor).cuda(device, non_blocking=True)
    fixed_class = torch.from_numpy(np.random.uniform(0, 1, size=(FG.batch_size,\
                           1))).type(torch.FloatTensor).cuda(device, non_blocking=True)

    input = torch.FloatTensor(batchSize, 1, imageSize, imageSize, imageSize).cuda(device, non_blocking=True)
    noise = torch.FloatTensor(batchSize, z).cuda(device, non_blocking=True)
    fixed_noise = torch.FloatTensor(batchSize, z).normal_(0, 1).cuda(device, non_blocking=True)

    label = torch.FloatTensor(batchSize).cuda(device, non_blocking=True)
    real_label = 1
    fake_label = 0

    for epoch in range(FG.num_epochs):
        stats = logging.Statistics(['loss_D', 'loss_GE'])
        printers['lr']('D', epoch, optimizerD.param_groups[0]['lr'])
        printers['lr']('GE',  epoch, optimizerGE.param_groups[0]['lr'])
        printers['lr']('info', epoch, optimizerinfo.param_groups[0]['lr'])
        timer.tic()

        if (epoch+1)%100 == 0:
            schedularD.step()
            schedularGE.step()
            schedularinfo.step()
            #schedularAC.step()

        torch.set_grad_enabled(True)
        D.train(True)
        G.train(True)
        for i, data in enumerate(trainloader):
            x = data['image']
            y = data['target']
            batch_size = x.size(0)  # batch_size <= FG.batch_size
            D.zero_grad()
            G.zero_grad()
            print(x.shape)
            exit()
            x = x.unsqueeze(dim=1)
            inputs = (x*0.5)+0.5
            printers['inputs']('input', inputs)

            """#####################################
               ##       Update D network          ##
               #####################################"""
            """---------- train with real ----------"""
            D.zero_grad()
            real = x.cuda(device, non_blocking=True)
            batch_size = real.size(0)
            input.resize_as_(real).copy_(real)
            label.resize_(batch_size).fill_(real_label)

            inputv = input.cuda(device, non_blocking=True)
            labelv = label.cuda(device, non_blocking=True)
            output = D(inputv)
            lossD_real = criterion(output, labelv)
            lossD_real.backward()
            D_x = output.data.mean()

            """---------- train with  fake ----------"""
            noise.resize_(batch_size, z).normal_(0, 1)
            #print(noise.shape)
            noisev = noise.cuda(device, non_blocking=True)
            fake = G(noisev)
            labelv = label.fill_(fake_label).cuda(device, non_blocking=True)
            output = D(fake.detach())
            lossD_fake = criterion(output, labelv)
            lossD_fake.backward()
            D_G_z1 = output.data.mean()

            lossD = (lossD_real + lossD_fake)/2
            #lossD.backward()
            optimizerD.step()
            printers['noise']('noise', torch.clamp(fake/fake.max(), min=0, max=1))

            """#####################################
               ##       Update G network          ##
               #####################################"""
            G.zero_grad()
            labelv = label.fill_(real_label)
            output = D(fake)
            lossG = criterion(output, labelv)
            lossG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            FG.global_step += 1
            printers['D_fake_loss']('D_fake_loss', FG.global_step/len(trainloader),
                                    lossD_fake)
            printers['D_real_loss']('D_real_loss', FG.global_step/len(trainloader),
                                    lossD_real)
            printers['D_loss']('D_loss', FG.global_step/len(trainloader), lossD)
            printers['G_loss']('G_loss', FG.global_step/len(trainloader), lossG)
            pbar.update()
        pbar.close()

        fake = G(fixed_noise)

        printers['output']('fake_noise', torch.clamp(fake/fake.max(), min=0, max=1))
        if ((epoch+1) % 100 == 0):
            print(i, "step")
            fake = G(fixed_noise)
            save_printer = summary.Image3D(vis, 'fake_samples_epoch_'+str(epoch+1))
            save_printer('fake_samples_epoch_'+str(epoch+1), torch.clamp(fake/fake.max(), min=0, max=1))

        print("loss_D:",lossD.item(),", loss_G:",lossG.item(),", score_D:",D_x.item(),
              ", score_G1:",D_G_z1.item(),", score_G2:",D_G_z2.item())
        result_dict = {"loss_D":lossD,"loss_G":lossG,"score_D":D_x,"score_G1":D_G_z1,"score_G2":D_G_z2}
        # do checkpointing
        torch.save(G.state_dict(), '%s/G.pth' % (outf))
        torch.save(D.state_dict(), '%s/D.pth' % (outf))

        for step, data in enumerate(trainloader):



            """        G network          """
            z_fake = torch.rand(batch_size, FG.z_dim).type(torch.FloatTensor)
            c_fake = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size,\
                                   c_code))).type(torch.FloatTensor)

            class_fake = torch.from_numpy(np.random.uniform(0, 1, size=(batch_size,\
                                   1))).type(torch.FloatTensor)
            C = torch.cat([c_fake, class_fake],1)
            z_fake, c_fake, C = z_fake.cuda(device, non_blocking=True),\
                            c_fake.cuda(device, non_blocking=True),\
                            C.cuda(device, non_blocking=True)

            x_fake_0 = G(z_fake, c_fake, FG.axis)
            # x_fake_0 = G(z_fake, c_fake, d_code[0], FG.axis)
            # x_fake_1 = G(z_fake, c_fake, d_code[1], FG.axis)
            #x_fake = G(z_fake, C, FG.axis)
            printers['outputs']('x_0_fake', x_fake_0[0,:,:,:])
            # printers['outputs2']('x_1_fake', x_fake_1[0,:,:,:])

            """         E network         """
            #x_real = x.type(torch.FloatTensor).cuda(device, non_blocking=True)
            x_real_0 = x.cuda(device, non_blocking=True)
            z_real_0, c_real_0 = E(x_real_0)
            # z_real_1, c_real_1 = E(x_real[1])

            """         D network         """
            # print(x_fake.shape, x_real.shape)
            output_fake_0 = D(x_fake_0, z_fake)
            output_real_0 = D(x_real_0, z_real_0)
            # output_fake_1 = D(x_fake_1, z_fake_1)
            # output_real_1 = D(x_real_1, z_real_1)

            real_rate[step] = torch.mean(output_real_0)
            fake_rate[step] = torch.mean(output_fake_0)
            print('Real : ', real_rate[step].item(), ' Fake : ', fake_rate[step].item())

            loss_D_0 = -torch.mean(torch.log(output_real_0+EPS)+torch.log(1-output_fake_0+EPS))
            loss_GE_0 = -torch.mean(torch.log(1-output_real_0+EPS)+torch.log(output_fake_0+EPS))

            loss_D = loss_D_0
            loss_GE = loss_GE_0

            _, c_fake_E_0 = E(x_fake_0)
            c_fake_E_0 = c_fake_E_0[:, :FG.c_code]
            # _, c_fake_E_1 = E(x_fake_1)
            # c_fake_E_1 = c_fake_E_1[:, :FG.c_code]
            # y_class = y.type(torch.FloatTensor).cuda(device, non_blocking=True)
            loss_info = MSE_loss(c_fake_E_0, c_fake)

            loss_D.backward(retain_graph=True)
            optimizerD.step()
            loss_GE.backward(retain_graph=True)
            optimizerGE.step()
            loss_info.backward()
            optimizerinfo.step()

            if FG.wasserstein:
                for p in D.parameters():
                    p.data.clamp_(-FG.clamp, FG.clamp)

        printers['D_loss']('train', epoch+step/len(trainloader), loss_D)
        printers['G_loss']('train', epoch+step/len(trainloader), loss_GE)
        printers['info_loss']('train', epoch+step/len(trainloader), loss_info)
        printers['ofake']('train', epoch+step/len(trainloader), torch.mean(fake_rate))
        printers['oreal']('train', epoch+step/len(trainloader), torch.mean(real_rate))

        train_acc = train_scores.accuracy
        # printers['acc']('train', epoch+i/len(trainloader), train_acc)
        print("Epoch: [%2d] D_loss: %.4f, G_loss: %.4f, info_loss: %.4f" %
             ((epoch + 1), loss_D.item(), loss_GE.item(), loss_info.item()))

        if (epoch+1)%10 == 0:
            G.eval()
            D.eval()
            E.eval()
            """      G network     """
            C = torch.cat([fixed_c, fixed_class],1)
            valid_x_fake = G(fixed_z, fixed_c, FG.axis)
            # valid_x_fake = G(fixed_z, C, FG.axis)
            fake = (valid_x_fake*0.5)+0.5
            printers['fake']('fake', fake[0,:,:,:])
            if ((epoch+1) % 50 == 0):
                saver = Image3D(vis, 'output_'+str(epoch+1))
                saver('output_'+str(epoch+1), fake[0,:,:,:])

            valid_scores.clear()

        if ((epoch+1) % 10 == 0):
            with torch.no_grad():
                torch.save(D.state_dict(), '%s/D_%d.pth' % (save_dir, epoch+1))
                torch.save(G.state_dict(), '%s/G_%d.pth' % (save_dir, epoch+1))
                torch.save(E.state_dict(), '%s/E_%d.pth' % (save_dir, epoch+1))
        # torch.save(D.state_dict(), os.path.join(save_dir, 'D.pth'))
        # torch.save(G.state_dict(), os.path.join(save_dir, 'G.pth'))
        # torch.save(E.state_dict(), os.path.join(save_dir, 'E.pth'))
        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))
        vis.save([vis.env])
        time.sleep(0.5)
