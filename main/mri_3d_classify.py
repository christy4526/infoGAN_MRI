from __future__ import print_function, division, absolute_import
import os

# visualizes
from tqdm import tqdm
from visdom import Visdom

# deep learning framework
import torch
from torchvision.transforms import Compose, Lambda
from torch.utils.data import DataLoader
from torch.nn.init import kaiming_normal_
from torch import optim

# project packages
from config import train_args, argument_report
from models import Generator, Discriminator, Plane
from dataset import ADNIDataset, Trainset, fold_split
from transforms import RandomCrop, ToWoldCoordinateSystem, ToTensor, Normalize
from utils import print_model_parameters, ScoreReport
from utils import save_checkpoint
from summary import Scalar, Image3D
import numpy as np
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    # option flags
    FG = train_args()
    report = argument_report(FG, end='<br>')

    # create summary and report the option
    #vis = Visdom(port=10001, env='info_classifier')
    vis = Visdom(port=10002, env=str(FG.vis_env))

    vis.text(argument_report(FG, end='<br>'), win='config')

    # torch setting - main device
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])

    # create save porder
    FG.save_dir = 'result_classify'
    if not os.path.exists(FG.save_dir):
            os.makedirs(FG.save_dir)

    # create train set
    x, y, train_idx, test_idx, ratio = fold_split(FG)
    # x = image, y=target
    trainset = ADNIDataset(FG, x[train_idx], y[train_idx],
                           transform=Compose([ToWoldCoordinateSystem(), RandomCrop(64,80,64),
                                              Normalize(0.2,0.8),ToTensor()]))
                                              #Normalize(0.5, 0.5)]))
    testset = ADNIDataset(FG, x[test_idx], y[test_idx],
                          transform=Compose([ToWoldCoordinateSystem(), RandomCrop(64,80,64),
                                             Normalize(0.2,0.8),ToTensor()]))
                                              #Normalize(0.5, 0.5)]))
    trainloader = DataLoader(trainset, batch_size=FG.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=4)
    testloader = DataLoader(testset, batch_size=len(FG.devices),
                             shuffle=False, pin_memory=True,
                             num_workers=4)

    # create model
    model = Plane(len(FG.labels))
    print_model_parameters(model)
    model.to(device)
    model = torch.nn.DataParallel(model, FG.devices)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=FG.lr, weight_decay=FG.l2_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, FG.lr_gamma)

    # create generator
    G = Generator(FG).to('cuda:{}'.format(FG.devices[0]))
    G = torch.nn.DataParallel(G, FG.devices)
    #G.apply(weights_init)
    #G.load_state_dict(torch.load(os.path.join(FG.save_dir, 'G.pth')))
    print_model_parameters(G)
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrD, betas=(0.5, 0.999))

    # create latent codes
    discrete_code = FG.d_code
    continuous_code = FG.c_code
    #sample_num = discrete_code ** 3 #9
    sample_num = 50
    z_dim = FG.z
    sample_c = torch.zeros((sample_num, continuous_code))
    sample_c = sample_c.cuda(device, non_blocking=True)

    # create visdom printers
    printers = dict(
        lr = Scalar(vis, 'learning_rate', opts=dict(
            showlegend=True, title='learning rate', ytickmin=0, ytinkmax=1)),
        loss = Scalar(vis, 'loss', opts=dict(
            showlegend=True, title='loss', ytickmin=0, ytinkmax=3)),
        acc = Scalar(vis, 'acc', opts=dict(
            showlegend=True, title='acc', ytickmin=0, ytinkmax=3)),
        accuracy = Scalar(vis, 'accuracy', opts=dict(
            showlegend=True, title='accuracy', ytickmin=0, ytinkmax=3)),
        predict0 = Scalar(vis, 'predict_0', opts=dict(
            showlegend=True, title='predict_0', ytickmin=0, ytinkmax=3)),
        predict1 = Scalar(vis, 'predict_1', opts=dict(
            showlegend=True, title='predict_1', ytickmin=0, ytinkmax=3)),
        predict2 = Scalar(vis, 'predict_2', opts=dict(
            showlegend=True, title='predict_2', ytickmin=0, ytinkmax=3)),
        input = Image3D(vis, 'input'),
        output_0 = Image3D(vis, 'output_0'),
        output_1 = Image3D(vis, 'output_1'),
        output_2 = Image3D(vis, 'output_2'))
    parent_pbar = tqdm(total=FG.num_epoch)
    ######################### training ###########################
    train_scores = ScoreReport()
    test_scores = ScoreReport()
    max_acc = 0
    min_loss = 0

    for epoch in range(FG.num_epoch):
        sample_z = torch.zeros((sample_num, z_dim))
        for k in range(discrete_code):
            sample_z[k*discrete_code] = torch.rand(1, z_dim)
            for j in range(1, discrete_code):
                sample_z[k*discrete_code+j] = sample_z[k*discrete_code]
        sample_z = sample_z.cuda(device, non_blocking=True)

        # train()
        model.train(True)
        torch.set_grad_enabled(True)

        train_scores.clear()
        scheduler.step()
        # print lr
        printers['lr']('lr',
                       epoch, optimizer.param_groups[0]['lr'])
        train_pbar = tqdm(total=len(trainloader), desc='Epoch Train {:>3}'.format(epoch))
        for i, data in enumerate(trainloader):
            #print(i, data)
            image = data['image']
            target = data['target']
            image = image.cuda(device, non_blocking=True)
            target = target.type(torch.LongTensor).cuda(device, non_blocking=True)
            #printers['output_0']('input',image)
            #exit()

            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # print train loss
            score = torch.nn.functional.softmax(output, 1)
            train_scores.update_true(target)
            train_scores.update_score(score)
            #printers['loss']('train_f{}'.format(FG.code), epoch+i/len(trainloader), loss)
            #print(len(train_scores.y_true), len(train_scores.y_pred))
            train_pbar.update()
        train_pbar.close()

        #print(train_scores.accuracy)
        # printers['acc']('train_f{}'.format(FG.code),
        #                 epoch, train_scores.accuracy)

        ############################ test #############################
        G.eval()
        model.eval()
        torch.set_grad_enabled(False)
        test_scores.clear()

        test_pbar = tqdm(total=len(testloader), desc='Epoch Train {:>3}'.format(epoch))
        for i, data in enumerate(testloader):
            image = data['image']
            target = data['target']
            image = image.cuda(device, non_blocking=True)
            target = target.type(torch.LongTensor).cuda(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)

            # print test loss
            score = torch.nn.functional.softmax(output, 1)
            test_scores.update_true(target)
            test_scores.update_score(score)
            test_scores.update_loss(loss)
            #printers['loss']('test_f{}'.format(FG.code), epoch+i/len(testloader), loss)

            test_acc = test_scores.accuracy
            test_loss = test_scores.loss
            test_pbar.update()
        test_pbar.close()

        #print(test_scores.accuracy)
        # printers['acc']('test_f{}'.format(FG.code),
        #                 epoch, test_scores.accuracy)
        if test_acc > max_acc:
            min_loss = test_loss
            max_acc = test_acc
            training_state = dict(
                epoch=epoch, best_score=dict(loss=min_loss, acc=max_acc),
                state_dict=model.module.state_dict(),
                optimizer_state_dict=optimizer.state_dict())
            save_checkpoint(FG, training_state, is_best=True)

        for i in range(FG.d_code-1):
            pred = [0,0,0]
            y_score, y_pred=[],[]

            temp_y = torch.zeros((sample_num, discrete_code))
            for j in range(sample_num):
                temp_y[j][2-2*i] = 1 # 1, 2 생성
            sample_y = temp_y
            sample_y = sample_y.cuda(device, non_blocking=True)

            gimage = G(sample_z, sample_c, sample_y)
            #printers['input']('input', torch.clamp(gimage/gimage.max(), min=0, max=1))
            target = torch.zeros(sample_num).type(torch.LongTensor)
            for j in range(sample_num):
                target[j] = i # target = 0,1
            target = target.cuda(device, non_blocking=True)

            output = model(gimage)
            loss = criterion(output, target)
            score=torch.nn.functional.softmax(output, 1)
            if  score.requires_grad:
                score=score.detach()
            score = score.cpu()
            if score.dim() == 1:
                score = score.unsqueeze(0)
            y_score += score.tolist()

            y_pred = [np.argmax(s) for s in y_score]

            for j in range(sample_num):
                if y_pred[j] == 0:
                    pred[0] += 1
                elif y_pred[j] == 1:
                    pred[1] += 1
                elif y_pred[j] == 2:
                    pred[2] += 1

            #test_acc = test_scores.accuracy
            #test_loss = test_scores.loss
            #printers['loss']('test_c{}'.format(i*2), epoch, test_loss)
            # accuracy : 0 class image를 0으로 predict

            if i == 0 :
                printers['output_0']('output_0', torch.clamp(gimage/gimage.max(), min=0, max=1))
                printers['predict0']('AD', epoch, pred[0]/sample_num)
                printers['predict0']('NC', epoch, pred[1]/sample_num)
            elif i == 1 :
                printers['output_1']('output_1', torch.clamp(gimage/gimage.max(), min=0, max=1))
                printers['predict1']('AD', epoch, pred[0]/sample_num)
                printers['predict1']('NC', epoch, pred[1]/sample_num)
            else :
                printers['output_2']('output_2', torch.clamp(gimage/gimage.max(), min=0, max=1))
                printers['predict2']('AD', epoch, pred[0]/sample_num)
                printers['predict2']('NC', epoch, pred[1]/sample_num)

        """
        # report

        """
        parent_pbar.update()
    parent_pbar.close()
