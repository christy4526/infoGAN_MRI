import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torch.nn.init import kaiming_normal_
from torch import optim

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models import sliceGenerator, Discriminator, Plane2D
from visdom import Visdom
from config import train_args, argument_report
from dataset import ADNIDataset, ADNIDataset2D, Trainset, fold_split
from transforms_2d import RandomCrop, ToWoldCoordinateSystem, ToTensor, FiveCrop, Normalize
import ori_utils, utils, time, os, pickle, itertools
from utils import ScoreReport, SimpleTimer, save_checkpoint
from summary import Image3D, Scalar


if __name__ == '__main__':
    FG = train_args()
    report = argument_report(FG, end='<br>')

    # create summary and report the option
    vis = Visdom(port=10002, env=str(FG.vis_env))
    vis.text(argument_report(FG, end='<br>'), win='config')

    # torch setting - main device
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])

    # create save porder
    FG.save_dir = 'result_2d_classify'
    if not os.path.exists(FG.save_dir):
            os.makedirs(FG.save_dir)

    # create train set
    x, y, train_idx, test_idx, ratio = fold_split(FG)
    # x = image, y=target
    if FG.isize == 157:
        if FG.axis == 0:
            RCrop=RandomCrop(160,128)
            FCrop=FiveCrop((188,156),(160,128))
        else:
            RCrop=RandomCrop(128,128)
            FCrop=FiveCrop((156,156),(128,128))
    if FG.isize == 79:
        if FG.axis == 0:
            RCrop=RandomCrop(80,64)
            FCrop=FiveCrop((95,79),(80,64))
        else:
            RCrop=RandomCrop(64,64)
            FCrop=FiveCrop((79,79),(64,64))
    trainset = ADNIDataset2D(FG, x[train_idx], y[train_idx],
                           transform=Compose([ToWoldCoordinateSystem(), RCrop,
                                              ToTensor()]))
    testset = ADNIDataset2D(FG, x[test_idx], y[test_idx],
                          transform=Compose([ToWoldCoordinateSystem(),FCrop,
                                             Lambda(lambda patches: torch.stack([
                                             ToTensor()(patch) for patch in patches]))]))

    trainloader = DataLoader(trainset, batch_size=FG.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    testloader = DataLoader(testset, num_workers=4, pin_memory=True)
    #testloader = DataLoader(testset, batch_size=FG.batch_size, num_workers=4, pin_memory=True)

    # create model
    #model = Plane(len(FG.labels))
    model = Plane2D(FG, len(FG.labels))
    #print_model_parameters(model)
    model.to(device)
    model = torch.nn.DataParallel(model, FG.devices)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    #criterion = torch.nn.CrossEntropyLoss(((1-ratio)*2).to(device))
    optimizer = torch.optim.Adam(model.parameters(),lr=FG.lr, weight_decay=FG.l2_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, FG.lr_gamma)

    # create generator
    G = sliceGenerator(FG).to('cuda:{}'.format(FG.devices[0]))
    G = torch.nn.DataParallel(G, FG.devices)
    #G.load_state_dict(torch.load(os.path.join(FG.save_dir, 'G.pth')))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrD, betas=(0.5, 0.999))

    # create latent codes
    discrete_code = FG.d_code
    continuous_code = FG.c_code
    #sample_num = discrete_code ** 3 #9
    sample_num = FG.batch_size
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
        predict = Scalar(vis, 'predict', opts=dict(
            showlegend=True, title='predict', ytickmin=0, ytinkmax=3)),
        train_input = Image3D(vis, 'train_input'),
        test_input = Image3D(vis, 'test_input'),
        output0 = Image3D(vis, 'output0'),
        output1 = Image3D(vis, 'output1'),
        output2 = Image3D(vis, 'output2'))

    train_scores = ScoreReport()
    test_scores = ScoreReport()
    max_acc = 0
    min_loss = 0

    for epoch in range(FG.num_epoch):
        model.train()
        torch.set_grad_enabled(True)
        train_scores.clear()
        scheduler.step()
        # print lr
        printers['lr']('lr_f{}'.format(FG.running_fold), epoch, optimizer.param_groups[0]['lr'])
        ############################ classification : train #############################
        train_pbar = tqdm(total=len(trainloader), desc='Epoch {:>3}'.format(epoch))
        for i, data in enumerate(trainloader):
            #print(i, data)
            image = data['image']
            target = data['target']
            images = image.cuda(device, non_blocking=True)
            target = target.type(torch.LongTensor).cuda(device, non_blocking=True)

            printers['train_input']('train_input', images[1,:,:,:])

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            score = torch.nn.functional.softmax(outputs, 1)
            train_scores.update_true(target)
            train_scores.update_score(score)
            printers['loss']('train_f{}'.format(FG.running_fold),
                             epoch+i/len(trainloader), loss)
            train_pbar.update()
        train_pbar.close()


        printers['acc']('train_f{}'.format(FG.running_fold), epoch, train_scores.accuracy)


        ############################ classification : test #############################
        model.eval()
        torch.set_grad_enabled(False)
        test_scores.clear()

        test_pbar = tqdm(total=len(testloader), desc='Validation {:>3}'.format(epoch))
        for i, data in enumerate(testloader):
            image = data['image']
            _, npatchs, c, w, h = image.shape
            image = image.reshape(-1, c, w, h)
            images = image.cuda(device, non_blocking=True)
            target = data['target']
            target_r = torch.Tensor(
                [target.item() for _ in range(npatchs)]).long()
#            target_r = torch.zeros((images.shape[0])).long()
#             for j in range(images.shape[0]):
#                 for k in range(npatchs):
#                     print(target)
#                     print(j*npatchs+k)
#                     target_r[j*npatchs+k] = target[j]
            target_r = target_r.cuda(device, non_blocking=True)
            #target = target.cuda(device, non_blocking=True)

            for j in range(images.shape[0]):
                printers['test_input']('test_input', images[j,:,:,:])
            output = model(images)
            loss = criterion(output, target_r)
            #loss = criterion(output, target)

            # print test loss
            score = torch.mean(
                torch.nn.functional.softmax(output, 1), 0)
            score = torch.nn.functional.softmax(output, 1)

            #test_scores.update_true(target)
            test_scores.update_true(target_r)
            test_scores.update_score(score)
            test_scores.update_loss(loss)
            test_pbar.update()
        test_pbar.close()
        test_acc = test_scores.accuracy
        test_loss = test_scores.loss

        printers['loss']('test_f{}'.format(FG.running_fold), epoch, test_loss)
        printers['acc']('test_f{}'.format(FG.running_fold), epoch, test_acc)
        #print('Test : epoch :%d correct:%d total:%d mean: %f'%(epoch,  correct, total, correct/total))

        fname = 'Plane2D_157_'+str(FG.axis)+'_4th'
        if FG.isize == 79:
            fname = 'Plane2D_79_'+str(FG.axis)

        if test_acc > max_acc:
            min_loss = test_loss
            max_acc = test_acc
            training_state = dict(
                epoch=epoch, best_score=dict(loss=min_loss, acc=max_acc),
                state_dict=model.module.state_dict(),
                optimizer_state_dict=optimizer.state_dict())
            #save_checkpoint(FG, FG.model, training_state, is_best=True)
            save_checkpoint(FG, fname, training_state, is_best=True)

        ############################ Generator : test #############################
        # G.eval()
        # for i in range(discrete_code):
        #     Gcorrect = 0
        #     Gtotal = 0
        #     y_score, y_pred=[],[]
        #     pred = np.zeros(discrete_code)
        #
        #     temp_y = torch.zeros((sample_num, discrete_code))
        #     for j in range(sample_num):
        #         temp_y[j][i] = 1
        #     sample_y = temp_y
        #     sample_y = sample_y.cuda(device, non_blocking=True)
        #
        #     target = torch.zeros(sample_num).type(torch.LongTensor)
        #     for j in range(sample_num):
        #         target[j] = i # target = 0,1
        #     target = target.cuda(device, non_blocking=True)
        #
        #     # sample_z = torch.zeros((sample_num, z_dim))
        #     # for k in range(discrete_code):
        #     #     sample_z[k] = torch.rand(1, z_dim)
        #     # sample_z = sample_z.cuda(device, non_blocking=True)
        #     #
        #     # gimage = G(sample_z, sample_c, sample_y, FG.axis)
        #
        #     gimages = torch.zeros((sample_num, 1, 95, 79))
        #     for j in range(FG.batch_size):
        #         sample_z = torch.zeros((sample_num, z_dim))
        #         for k in range(discrete_code):
        #             sample_z[k] = torch.rand(1, z_dim)
        #         sample_z = sample_z.cuda(device, non_blocking=True)
        #         gimage = G(sample_z, sample_c, sample_y, FG.axis)
        #         gimages[j] = torch.clamp(gimage[0,:,:,:], min=0,max=1)
        #
        #     for j in range(int(FG.batch_size/5)):
        #         checker = Image3D(vis, 'target_'+str(target[0].item())+'_gi_'+str(j))
        #         checker('target_'+str(target[0].item())+'_gi_'+str(j), gimages[j])
        #
        #     output = model(gimages)
        #     loss = criterion(output, target)
        #     score=torch.nn.functional.softmax(output, 1)
        #
        #     if  score.requires_grad:
        #         score=score.detach()
        #     score = score.cpu()
        #     if score.dim() == 1:
        #         score = score.unsqueeze(0)
        #     y_score += score.tolist()
        #
        #     y_pred = [np.argmax(s) for s in y_score]
        #
        #     for k in range(sample_num):
        #         if y_pred[k] == 0:
        #             pred[0] += 1
        #         elif y_pred[k] == 1:
        #             pred[1] += 1
        #
        #     test_acc = test_scores.accuracy
        #     test_loss = test_scores.loss
        #     printers['loss']('G_class{}'.format(i), epoch, test_loss)
        #
        #     for j in range(discrete_code):
        #         printers['predict']('{}class_to_{}'.format(i,j), epoch, pred[j]/sample_num)
        #     _, predicted = torch.max(output.data, 1)
        #     Gtotal += target.size(0)
        #     Gcorrect += (predicted == target).sum().item()
        #     printers['accuracy']('test_G_{}'.format(i), epoch+i/sample_num, Gcorrect/Gtotal)


    # Save the model and plot
    #torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.pkl')

    #########################################################################
