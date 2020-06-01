from __future__ import print_function, division, absolute_import
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
from torch import optim


# project packages
from config import train_args, argument_report
from biGAN import Generator, Discriminator, Q, AP
from models import sliceGenerator, sliceDiscriminator
from dataset import ADNIDataset, Trainset, fold_split
from transforms import RandomCrop, ToWoldCoordinateSystem, ToTensor, Normalize
from utils import print_model_parameters, ScoreReport, SimpleTimer, loss_plot
from utils import save_checkpoint
from summary import Scalar, Image3D
import itertools
import numpy as np
import imageio

import itertools
import shutil
import torch.backends.cudnn as cudnn

#from utils import logging
from torch.autograd import Variable
from sklearn import metrics


if __name__ == '__main__':
    FG = train_args()
    vis = Visdom(port=FG.vis_port, env=str(FG.vis_env))
    vis.text(argument_report(FG, end='<br>'), win='config')

    # torch setting
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])
    timer = SimpleTimer()

    FG.save_dir = str(FG.vis_env)
    if not os.path.exists(FG.save_dir):
            os.makedirs(FG.save_dir)

    printers = dict(
        lr = Scalar(vis, 'lr', opts=dict(
            showlegend=True, title='lr', ytickmin=0, ytinkmax=2.0)),
        D_loss = Scalar(vis, 'D_loss', opts=dict(
            showlegend=True, title='D loss', ytickmin=0, ytinkmax=2.0)),
        G_loss = Scalar(vis, 'G_loss', opts=dict(
            showlegend=True, title='G loss', ytickmin=0, ytinkmax=10)),
        AP_loss = Scalar(vis, 'AP_loss', opts=dict(
            showlegend=True, title='AP_loss', ytickmin=0, ytinkmax=10)),
        info_loss = Scalar(vis, 'info_loss', opts=dict(
            showlegend=True, title='info_loss', ytickmin=0, ytinkmax=10)),
        acc = Scalar(vis, 'Accuracy', opts=dict(
            showlegend=True, title='Accuracy', ytickmin=0, ytinkmax=2.0)),
        inputs = Image3D(vis, 'inputs'),
        outputs = Image3D(vis, 'outputs'))


    # create train set,  x = image, y=target
    x, y, train_idx, test_idx, ratio = fold_split(FG)
    #transform=Compose([ToWoldCoordinateSystem(), Normalize((0.5, 0.9)), ToTensor()])
    transform=Compose([ToWoldCoordinateSystem(), ToTensor()])

    trainset = ADNIDataset(FG, x[train_idx], y[train_idx], transform=transform)
    testset = ADNIDataset(FG, x[test_idx], y[test_idx], transform=transform)

    trainloader = DataLoader(trainset, batch_size=FG.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=FG.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    D = Discriminator(FG).to('cuda:{}'.format(FG.devices[0]))  # discriminator D(x, z)
    G = Generator(FG).to('cuda:{}'.format(FG.devices[0]))      # generator(decoder) P(x|z)
    Q = Q(FG).to('cuda:{}'.format(FG.devices[0]))              # inference(encoder) Q(z|x)
    AP = AP(FG).to('cuda:{}'.format(FG.devices[0]))            # predict AD

    if FG.load_pkl == 'true':
        G.load_state_dict(torch.load(os.path.join(FG.save_dir, 'G.pth')))
        D.load_state_dict(torch.load(os.path.join(FG.save_dir, 'D.pth')))
        Q.load_state_dict(torch.load(os.path.join(FG.save_dir, 'Q.pth')))
        AP.load_state_dict(torch.load(os.path.join(FG.save_dir, 'AP.pth')))

    # if len(FG.devices) != 1:
    #     G = torch.nn.DataParallel(G, FG.devices)
    #     D = torch.nn.DataParallel(D, FG.devices)
    #     Q = torch.nn.DataParallel(Q, FG.devices)
    #     AP = torch.nn.DataParallel(AP, FG.devices)

    def matching(preds, true):
        total, match = 0, 0
        for p, t in zip(preds, true):
            if np.argmax(p) in np.nonzero(t)[0]:
                match += 1
            total += 1

        return match/ total

    BCE_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
    CE_loss = nn.CrossEntropyLoss().to('cuda:{}'.format(FG.devices[0]))
    MSE_loss = nn.MSELoss().to('cuda:{}'.format(FG.devices[0]))
    L1_loss = nn.L1Loss().to('cuda:{}'.format(FG.devices[0]))

    optimizerD = optim.Adam(D.parameters(), lr=FG.lrD, betas=(FG.beta1, FG.beta2))
    optimizerG = optim.Adam(itertools.chain(G.parameters(),AP.parameters(), Q.parameters()),
                            lr=FG.lrG, betas=(FG.beta1, FG.beta2))
    optimizerAP = optim.Adam(itertools.chain(Q.parameters(),AP.parameters()),
                             lr=FG.lr, betas=(FG.beta1, FG.beta2))
    optimizerinfo = optim.Adam(itertools.chain(D.parameters(),G.parameters()),
                               lr=FG.lrD, betas=(FG.beta1, FG.beta2))

    schedularD=torch.optim.lr_scheduler.ExponentialLR(optimizerD, FG.lr_gamma)
    schedularG=torch.optim.lr_scheduler.ExponentialLR(optimizerG, FG.lr_gamma)
    schedularAP=torch.optim.lr_scheduler.ExponentialLR(optimizerAP, FG.lr_gamma)
    schedularinfo=torch.optim.lr_scheduler.ExponentialLR(optimizerinfo, FG.lr_gamma)

    # fixed noise & condition
    d_code = FG.d_code  # categorical distribution (i.e. label=AD,MCI,CN) --> slice distridution
    c_code = FG.c_code  # gaussian distribution (e.g. rotation, thickness)
    z_dim = FG.z
    ################################ training bigan ################################
    accs = []
    start_time = time.time()
    EPS = 1e-12

    train_scores = ScoreReport()
    valid_scores = ScoreReport()
    max_acc = 0
    min_loss = 0

    for epoch in range(FG.num_epoch):
        timer.tic()
        D.zero_grad()
        G.zero_grad()
        Q.zero_grad() #Encoder
        AP.zero_grad()
        train_scores.clear()

        if (epoch+1)%100 == 0:
            schedularD.step()
            schedularG.step()
            schedularinfo.step()
        else:
            pass
        schedularAP.step()
        printers['lr']('D', epoch, optimizerD.param_groups[0]['lr'])
        printers['lr']('G',  epoch, optimizerG.param_groups[0]['lr'])
        printers['lr']('info', epoch, optimizerinfo.param_groups[0]['lr'])
        printers['lr']('AP', epoch, optimizerAP.param_groups[0]['lr'])
        for i, data in enumerate(trainloader):
            x = data['image']
            xi = data['image']
            y = data['target']
            batch_size = x.size(0)  # batch_size <= FG.batch_size

            # extract d_code slice
            d_G = d_code
            addi = torch.zeros((batch_size,1*len(d_G),xi.shape[3],xi.shape[4]))

            for k in range(batch_size):
                for j in range(len(d_G)):
                    addi[k, j] = xi[k,:,d_G[j],:,:]
            x = torch.zeros((batch_size*len(d_G),1,xi.shape[3],xi.shape[4]))
            x = addi

            # G network
            z_G = torch.randn(batch_size, FG.z).type(torch.FloatTensor)
            c_G = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size,\
                                   c_code))).type(torch.FloatTensor)
            d_G = torch.ones((batch_size,len(d_code))).type(torch.FloatTensor)
            #print(c_G.shape, d_G.shape)
            z_G, c_G, d_G = z_G.cuda(device, non_blocking=True),\
                            c_G.cuda(device, non_blocking=True),\
                            d_G.cuda(device, non_blocking=True)
            x_G = G(z_G, c_G, d_G, FG.axis)
            printers['outputs']('x_G', x_G[0,:,:,:])

            # Q network
            x_q = x.type(torch.FloatTensor).cuda(device, non_blocking=True)
            #ap_q  = y.type(torch.FloatTensor).cuda(device, non_blocking=True)
            ap_q  = torch.zeros((batch_size, 2)).type(torch.FloatTensor)
            for i in range(batch_size):
                if y[i] == 0:
                    ap_q[i][0] = 1
                else:
                    ap_q[i][1] = 1

            ap_q  = ap_q.cuda(device, non_blocking=True)
            z_q = Q(x_q)

            # D network
            #print(x_G.view(x_G.shape[0],-1).shape,AP(z_G).shape)
            #print("D fake:", x_G.shape, x_G.view(x_G.shape[0],-1).shape)
            D_fake, c_D = D(x_G, AP(z_G), z_G)
            D_real, _ = D(x_q, ap_q, z_q)

            loss_D = -torch.mean(torch.log(D_real+EPS)+torch.log(1-D_fake+EPS))
            loss_G = -torch.mean(torch.log(D_fake+EPS)+torch.log(1-D_real+EPS))
            loss_info = MSE_loss(c_D, c_G)

            loss_D.backward(retain_graph=True)
            optimizerD.step()
            loss_G.backward(retain_graph=True)
            optimizerG.step()

            loss_info.backward(retain_graph=True)
            optimizerinfo.step()

            Q.zero_grad()
            AP.zero_grad()

            # Q network
            z_AP = Q(x_q)
            # AP network
            ap_AP = AP(z_AP)
            # AP loss & back propagation
            #acc = matching(AP_AP.permute(0,2,1).data.cpu().numpy(), AP.permute(0,2,1).numpy())
            score = ap_AP
            loss_AP = BCE_loss(ap_AP, ap_q).mean()
            loss_AP.backward()
            optimizerAP.step()
            train_scores.update_true(y)
            train_scores.update_score(score)
        printers['D_loss']('train', epoch+i/len(trainloader), loss_D)
        printers['G_loss']('train', epoch+i/len(trainloader), loss_G)
        printers['info_loss']('train', epoch+i/len(trainloader), loss_info)
        printers['AP_loss']('train', epoch+i/len(trainloader), loss_AP)

        train_acc = train_scores.accuracy
        printers['acc']('train', epoch+i/len(trainloader), train_acc)
        print("Epoch: [%2d] D_loss: %.5f, G_loss: %.5f, info_loss: %.5f" %
                      ((epoch + 1), loss_D.item(), loss_G.item(), loss_info.item()))

        if epoch % (10) == 0:
            #valid_data, valid_AP, _ = getData(data['valid'],FG.batch_size, iteration)
            valid_scores.clear()
            valid_printer = Image3D(vis, 'valid_output')
            for j, data in enumerate(testloader):
                x = data['image']
                valid_y = data['target']

                # extract d_code slice
                d_G = d_code
                addi = torch.zeros((x.shape[0],1*len(d_G),x.shape[3],x.shape[4]))
                for i in range(x.shape[0]):
                    for k in range(len(d_G)):
                        addi[i, k] = x[i,:,d_G[k],:,:]
                valid_data = torch.zeros((x.shape[0]*len(d_G),1,x.shape[3],x.shape[4]))
                valid_data = addi

                # G network
                valid_z_G = torch.randn(x.shape[0], FG.z).type(torch.FloatTensor).cuda(device, non_blocking=True)
                valid_c_G = torch.randn(x.shape[0], FG.c_code).type(torch.FloatTensor).cuda(device, non_blocking=True)
                valid_d_G = torch.ones((x.shape[0],len(d_code))).type(torch.FloatTensor).cuda(device, non_blocking=True)
                valid_x_G = G(valid_z_G, valid_c_G, valid_d_G, FG.axis)
                valid_printer('valid_output', valid_x_G[0,:,:,:])
                if epoch % 50 == 0:
                    valid_printer_save = Image3D(vis, 'valid_output_'+str(epoch+1))
                    valid_printer_save('valid_output_'+str(epoch+1), valid_x_G[0,:,:,:])

                # Q network
                valid_x_q = valid_data.type(torch.FloatTensor).cuda(device, non_blocking=True)
                #valid_AP_q  = valid_y.type(torch.FloatTensor).cuda(device, non_blocking=True)
                ap_q  = torch.zeros((x.shape[0], 2)).type(torch.FloatTensor)
                for i in range(x.shape[0]):
                    if valid_y[i] == 0:
                        ap_q[i][0] = 1
                    else:
                        ap_q[i][1] = 1

                valid_AP_q  = ap_q.cuda(device, non_blocking=True)
                valid_z_q = Q(valid_x_q)

                # D network
                valid_D_fake, valid_c_D = D(valid_x_G, AP(valid_z_G), valid_z_G)
                valid_D_real, _ = D(valid_x_q, valid_AP_q, valid_z_q)
                # loss & back propagation
                valid_loss_D = -torch.mean(torch.log(valid_D_real+EPS)+torch.log(1-valid_D_fake+EPS))
                valid_loss_G = -torch.mean(torch.log(valid_D_fake+EPS)+torch.log(1-valid_D_real+EPS))
                valid_loss_info = MSE_loss(valid_c_D, valid_c_G)
                # Q network
                valid_z_AP = Q(valid_x_q)

                # AP network
                valid_ap_AP = AP(valid_z_AP)
                # AP loss & back propagation
                #valid_acc = matching(valid_AP_AP.permute(0,2,1).data.cpu().numpy(), valid_AP.permute(0,2,1).numpy())
                valid_loss_AP = BCE_loss(valid_ap_AP, valid_AP_q).mean()
                score = valid_ap_AP
                valid_scores.update_true(valid_y)
                valid_scores.update_score(score)

            printers['D_loss']('valid', epoch+i/len(testloader), valid_loss_D)
            printers['G_loss']('valid', epoch+i/len(testloader), valid_loss_G)
            printers['AP_loss']('valid', epoch+i/len(testloader), valid_loss_AP)
            printers['info_loss']('valid', epoch+i/len(testloader), valid_loss_info)
            valid_acc = valid_scores.accuracy
            printers['acc']('valid', epoch+i/len(testloader), valid_acc)

            if valid_acc > max_acc:
                min_loss = valid_loss_AP
                max_acc = valid_acc
                training_state = dict(
                    epoch=epoch, best_score=dict(loss=min_loss, acc=max_acc),
                    #state_dict=AP.module.state_dict(),
                    optimizer_state_dict=optimizerAP.state_dict())
                #save_checkpoint(FG, FG.model, training_state, is_best=True)
                fname = FG.vis_env
                save_checkpoint(FG, fname, training_state, is_best=True)

            # logging
            # if FG.print_every > 0 and iteration%FG.print_every == 0:
            #     logger.log('epoch {}/{}, step {}/{}:\t{}'.format(epoch, FG.num_epochs, iteration%(len(data['train']['input'])//batch_size), len(data['train']['input'])//batch_size, info))
            result_dict = {"D_loss":loss_D, "G_loss":loss_G,"AP_loss":loss_AP}

            # if ((epoch+1) % 10 == 0):
            #     j = 0
            #     valid_acc = []
            #     valid_roc = []
            #     while valid_epoch < 1:
            #         valid_x, valid_AP, valid_epoch =  getData(data['test'],FG.batch_size, j)
            #         j += 1
            #         valid_AP_G = AP(Q(valid_x.type(torch.FloatTensor).cuda(device, non_blocking=True)))
            #         valid_acc.append(matching(valid_AP_G.permute(0,2,1).data.cpu().numpy(),valid_AP.permute(0,2,1).numpy()))
            #         valid_roc.append(metrics.roc_auc_score( valid_AP.permute(0,2,1).numpy().flatten().astype(int), valid_AP_G.permute(0,2,1).data.cpu().numpy().flatten()))
            #     accs.append(np.mean(valid_acc))
            #     print(np.mean(valid_roc))
            #     print(accs[-1])
            #     PATIENCE = 20
            #     if (epoch > PATIENCE
            #         and max(accs[-PATIENCE:])
            #         < max(accs)):
            #         epoch = FG.num_epochs

        if ((epoch+1) % 10 == 0):
            with torch.no_grad():
                torch.save(G.state_dict(), '%s/G_%d.pth' % (FG.save_dir, epoch+1))
                torch.save(D.state_dict(), '%s/D_%d.pth' % (FG.save_dir, epoch+1))
                torch.save(Q.state_dict(), '%s/Q_%d.pth' % (FG.save_dir, epoch+1))
                torch.save(AP.state_dict(), '%s/AP_%d.pth' % (FG.save_dir, epoch+1))

        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))
