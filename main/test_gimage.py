from __future__ import print_function, division, absolute_import

# visualizes
from tqdm import tqdm
from visdom import Visdom

# deep learning framework
import os
import torch
from torch import optim
from torchvision.transforms import Compose, Lambda
from torch.utils.data import DataLoader

# project packages
from models import sliceGenerator, Discriminator, Plane2D
from config import train_args, argument_report
from dataset import ADNIDataset2D, Trainset, fold_split
from transforms_2d import RandomCrop, ToWoldCoordinateSystem, ToTensor, FiveCrop

from utils import print_model_parameters, ScoreReport
from utils import save_checkpoint, load_checkpoint
from summary import Image3D, Scalar

from sklearn import metrics
import numpy as np


#def main(rf):
if __name__ == '__main__':
    # get FLAGS and repot it
    FG = train_args()
    vis = Visdom(port=10002, env=str(FG.vis_env))

    # main device
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])


    G = sliceGenerator(FG).to('cuda:{}'.format(FG.devices[0]))
    G = torch.nn.DataParallel(G, FG.devices)
    if FG.isize == 79:
        G.load_state_dict(torch.load(os.path.join('result_2d_classify', 'G_79.pth')))
    else:
        G.load_state_dict(torch.load(os.path.join('result_2d_classify', 'G.pth')))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrD, betas=(0.5, 0.999))

    # make dataset
    #_, test, _ = kfold_split(FG)
    x, y, train_idx, test_idx, ratio = fold_split(FG)

    # original dataset
    testset = ADNIDataset2D(FG, x[test_idx], y[test_idx], transform=Compose([
                            #ToWoldCoordinateSystem(), ToTensor()]))
                            ToWoldCoordinateSystem(),RandomCrop(80,64),ToTensor()]))

    # dataloader
    testloader = DataLoader(testset, num_workers=4, pin_memory=True)

    G.eval()
    torch.set_grad_enabled(False)

    discrete_code = FG.d_code
    continuous_code = FG.c_code
    z_dim = FG.z
    sample_num = 1000

    #temp_c = torch.linspace(-2, 2, 5)
    temp_c = torch.linspace(-2, 2, 10)

    sample_z = torch.zeros((sample_num, z_dim))
    z2 = torch.rand(1, z_dim)
    for i in range(sample_num):
        sample_z[i] = torch.rand(1, z_dim)
        #sample_z[i] = z2
    sample_z = sample_z.cuda(device, non_blocking=True)

    sample_c = torch.zeros((sample_num, continuous_code))

    idx=0
    for i in range(len(temp_c)):
        for j in range(100):
            sample_c[i*100+j, 0] = temp_c[idx]
        idx+=1
    for i in range(sample_num):
        sample_c[i, 2] = temp_c[i%len(temp_c)]

    idx=0
    for i in range(len(temp_c)**2):
        for j in range(len(temp_c)):
            sample_c[i*len(temp_c)+j, 1] = temp_c[idx%len(temp_c)]
        idx+=1

    sample_c = sample_c.cuda(device, non_blocking=True)
    gi_c1 = G(sample_z, sample_c, FG.axis)
    for i in range(sample_num):
        name = 'target_'+str(i)+':'+str(sample_c[i, 0].item())+':'+str(sample_c[i, 1].item())+':'+str(sample_c[i, 2].item())
        save_printer1 = Image3D(vis, name)
        save_printer1(name, torch.clamp(gi_c1[i,:,:,:], min=0, max=1))
    exit()
    ############################################################################

    # create latent codes
    discrete_code = FG.d_code
    continuous_code = FG.c_code
    z_dim = FG.z
    sample_num = 400
    sample_c = torch.zeros((sample_num, continuous_code))
    sample_c = sample_c.cuda(device, non_blocking=True)

    sample_z = torch.zeros((400, z_dim))
    for k in range(400):
        sample_z[k] = torch.rand(1, z_dim)
    sample_z = sample_z.cuda(device, non_blocking=True)

    temp_c = torch.linspace(-2, 2, 20)
    sample_c1 = torch.zeros((sample_num, continuous_code))
    for j in range(sample_num):
        sample_c1[j%20, 0] = temp_c[j%20]
    sample_c1 = sample_c1.cuda(device, non_blocking=True)

    ############################ original : test #############################
    test_scores = ScoreReport()
    test_scores.clear()
    print('original image result')
    for sample in testloader:
        mri = sample['image']
        target = sample['target']

        mri = mri.cuda(device, non_blocking=True)
        target = target.to(device)

        output = model(mri)

        score = torch.nn.functional.softmax(output, 1)
        test_scores.update_true(target)
        test_scores.update_score(score)

    print(test_scores.accuracy)

    ############################ cont 1 ##################################
    vis = Visdom(port=10002, env='cont-code1-test')
    gtest_scores = ScoreReport()
    gtest_scores.clear()
    print('continuous_code1 result')
    for i in range(discrete_code):
        y_score, y_pred=[],[]
        pred = np.zeros(discrete_code)
        #set sample_y : set class(0 or 1)
        temp_y = torch.zeros((sample_num, discrete_code))
        for j in range(sample_num):
            temp_y[j][i] = 1
        sample_y = temp_y
        sample_y = sample_y.cuda(device, non_blocking=True)

        if i == 0:
            target = torch.zeros(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)
        elif i == 1:
            target = torch.ones(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)

        gi_c1 = G(sample_z, sample_c1, sample_y, FG.axis)
        for j in range(int(sample_num/20)):
            name = 'target_'+str(i)+'_cont1:'+str(temp_c[j].item())
            save_printer1 = Image3D(vis, name)
            save_printer1(name, torch.clamp(gi_c1[j,:,:,:], min=0, max=1))

        gimages = torch.zeros((sample_num, 1, 80, 64))
        for j in range(sample_num):
            gimages[j] = torch.clamp(RandomCrop(80,64)(gi_c1[j,0,:,:]), min=0,max=1)
        gimages = gimages.contiguous().cuda(device, non_blocking=True)

        # for j in range(20):
        #     checker = Image3D(vis, 'target_'+str(target[0].item())+'_gi_'+str(j))
        #     checker('target_'+str(target[0].item())+'_gi_'+str(j), gimages[j])

        output = model(gimages)
        loss = criterion(output, target)
        #loss = criterion(output, target_r)
        score=torch.nn.functional.softmax(output, 1)

        for j in range(sample_num):
            #print(score[j][0].item() , score[j][1].item() )
            if i == 0:
                if score[j][0].item() >= 0.90:
                    sprint = Image3D(vis, str(i)+'_to_0 : '+str(score[j][0].item()))
                    sprint(str(i)+'_to_0 : '+str(score[j][0].item()), gimages[j,:,:,:])
                if score[j][1].item() >= 0.5:
                    sprint = Image3D(vis, str(i)+'_to_1 : '+str(score[j][1]))
                    sprint(str(i)+'_to_1 : '+str(score[j][1].item()), gimages[j,:,:,:])
            if i == 1:
                if score[j][0].item() >= 0.5:
                    sprint = Image3D(vis, str(i)+'_to_0 : '+str(score[j][0].item()))
                    sprint(str(i)+'_to_0 : '+str(score[j][0].item()), gimages[j,:,:,:])
                if score[j][1].item() >= 0.90:
                    sprint = Image3D(vis, str(i)+'_to_1 : '+str(score[j][1]))
                    sprint(str(i)+'_to_1 : '+str(score[j][1].item()), gimages[j,:,:,:])

        if  score.requires_grad:
            score=score.detach()
        score = score.cpu()
        if score.dim() == 1:
            score = score.unsqueeze(0)
        y_score += score.tolist()

        y_pred = [np.argmax(s) for s in y_score]

        for k in range(sample_num):
            if y_pred[k] == 0:
                pred[0] += 1
            elif y_pred[k] == 1:
                pred[1] += 1

        gtest_acc = gtest_scores.accuracy
        #gtest_loss = gtest_scores.loss
        #printers['loss']('G_class{}'.format(i), epoch, test_loss)

        for j in range(discrete_code):
            #printers['predict']('{}_class_to_{}'.format(i,j), epoch, pred[j]/sample_num)
            print('{}_class_to_{} : {}'.format(i,j, pred[j]/sample_num))
    ############################ cont 2 ##################################
    vis = Visdom(port=10002, env='cont-code2-test')
    gtest_scores = ScoreReport()
    gtest_scores.clear()
    print('continuous_code2 result')
    for i in range(discrete_code):
        y_score, y_pred=[],[]
        pred = np.zeros(discrete_code)
        #set sample_y : set class(0 or 1)
        temp_y = torch.zeros((sample_num, discrete_code))
        for j in range(sample_num):
            temp_y[j][i] = 1
        sample_y = temp_y
        sample_y = sample_y.cuda(device, non_blocking=True)

        if i == 0:
            target = torch.zeros(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)
        elif i == 1:
            target = torch.ones(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)

        sample_c2 = torch.zeros((sample_num, continuous_code))
        for j in range(sample_num):
            for k in range(sample_num):
                sample_c2[k, 0] = temp_c[j%20]
                sample_c2[k, 1] = temp_c[k%20]
            sample_c2 = sample_c2.cuda(device, non_blocking=True)

        gi_c2 = G(sample_z, sample_c2, sample_y, FG.axis)
        for k in range(int(sample_num/20)):
            name = 'target_'+str(i)+'_cont1:'+str(temp_c[k].item())+\
                    '_cont2:'+str(temp_c[k].item())
            save_printer2 = Image3D(vis, name)
            save_printer2(name, torch.clamp(gi_c2[k,:,:,:], min=0, max=1))

        gimages = torch.zeros((sample_num, 1, 80, 64))
        for j in range(sample_num):
            gimages[j] = torch.clamp(RandomCrop(80,64)(gi_c2[j,0,:,:]), min=0,max=1)
        gimages = gimages.contiguous().cuda(device, non_blocking=True)

        # for j in range(20):
        #     checker = Image3D(vis, 'target_'+str(target[0].item())+'_gi_'+str(j))
        #     checker('target_'+str(target[0].item())+'_gi_'+str(j), gimages[j])

        output = model(gimages)
        loss = criterion(output, target)
        #loss = criterion(output, target_r)
        score=torch.nn.functional.softmax(output, 1)

        for j in range(sample_num):
            #print(score[j][0].item() , score[j][1].item() )
            if i == 0:
                if score[j][0].item() >= 0.90:
                    sprint = Image3D(vis, str(i)+'_to_0 : '+str(score[j][0].item()))
                    sprint(str(i)+'_to_0 : '+str(score[j][0].item()), gimages[j,:,:,:])
                if score[j][1].item() >= 0.5:
                    sprint = Image3D(vis, str(i)+'_to_1 : '+str(score[j][1]))
                    sprint(str(i)+'_to_1 : '+str(score[j][1].item()), gimages[j,:,:,:])
            if i == 1:
                if score[j][0].item() >= 0.5:
                    sprint = Image3D(vis, str(i)+'_to_0 : '+str(score[j][0].item()))
                    sprint(str(i)+'_to_0 : '+str(score[j][0].item()), gimages[j,:,:,:])
                if score[j][1].item() >= 0.90:
                    sprint = Image3D(vis, str(i)+'_to_1 : '+str(score[j][1]))
                    sprint(str(i)+'_to_1 : '+str(score[j][1].item()), gimages[j,:,:,:])

        if  score.requires_grad:
            score=score.detach()
        score = score.cpu()
        if score.dim() == 1:
            score = score.unsqueeze(0)
        y_score += score.tolist()

        y_pred = [np.argmax(s) for s in y_score]

        for k in range(sample_num):
            if y_pred[k] == 0:
                pred[0] += 1
            elif y_pred[k] == 1:
                pred[1] += 1

        gtest_acc = gtest_scores.accuracy
        #gtest_loss = gtest_scores.loss
        #printers['loss']('G_class{}'.format(i), epoch, test_loss)

        for j in range(discrete_code):
            #printers['predict']('{}_class_to_{}'.format(i,j), epoch, pred[j]/sample_num)
            print('{}_class_to_{} : {}'.format(i,j, pred[j]/sample_num))

    ############################ Generator : test #############################
    vis = Visdom(port=10002, env='disc-code-test')
    gtest_scores = ScoreReport()
    gtest_scores.clear()

    print( 'discrete_code result')
    for i in range(discrete_code):
        y_score, y_pred=[],[]
        pred = np.zeros(discrete_code)
        #set sample_y : set class(0 or 1)
        temp_y = torch.zeros((sample_num, discrete_code))
        for j in range(sample_num):
            temp_y[j][i] = 1
        sample_y = temp_y
        sample_y = sample_y.cuda(device, non_blocking=True)

        if i == 0:
            target = torch.zeros(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)
        elif i == 1:
            target = torch.ones(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)

        gi = G(sample_z, sample_c, sample_y, FG.axis)

        gimages = torch.zeros((sample_num, 1, 80, 64))
        for j in range(sample_num):
            gimages[j] = torch.clamp(RandomCrop(80,64)(gi[j,0,:,:]), min=0,max=1)
        gimages = gimages.contiguous().cuda(device, non_blocking=True)

        for j in range(5):
            checker = Image3D(vis, 'target_'+str(target[0].item())+'_gi_'+str(j))
            checker('target_'+str(target[0].item())+'_gi_'+str(j), gimages[j])

        output = model(gimages)
        loss = criterion(output, target)
        #loss = criterion(output, target_r)
        score=torch.nn.functional.softmax(output, 1)


        for j in range(sample_num):
            #print(score[j][0].item() , score[j][1].item() )
            if i == 0:
                if score[j][0].item() >= 0.90:
                    sprint = Image3D(vis, str(i)+'_to_0 : '+str(score[j][0].item()))
                    sprint(str(i)+'_to_0 : '+str(score[j][0].item()), gimages[j,:,:,:])
                if score[j][1].item() >= 0.5:
                    sprint = Image3D(vis, str(i)+'_to_1 : '+str(score[j][1]))
                    sprint(str(i)+'_to_1 : '+str(score[j][1].item()), gimages[j,:,:,:])
            if i == 1:
                if score[j][0].item() >= 0.5:
                    sprint = Image3D(vis, str(i)+'_to_0 : '+str(score[j][0].item()))
                    sprint(str(i)+'_to_0 : '+str(score[j][0].item()), gimages[j,:,:,:])
                if score[j][1].item() >= 0.93:
                    sprint = Image3D(vis, str(i)+'_to_1 : '+str(score[j][1]))
                    sprint(str(i)+'_to_1 : '+str(score[j][1].item()), gimages[j,:,:,:])

        if  score.requires_grad:
            score=score.detach()
        score = score.cpu()
        if score.dim() == 1:
            score = score.unsqueeze(0)
        y_score += score.tolist()

        y_pred = [np.argmax(s) for s in y_score]

        for k in range(sample_num):
            if y_pred[k] == 0:
                pred[0] += 1
            elif y_pred[k] == 1:
                pred[1] += 1

        gtest_acc = gtest_scores.accuracy
        #gtest_loss = gtest_scores.loss
        #printers['loss']('G_class{}'.format(i), epoch, test_loss)

        for j in range(discrete_code):
            #printers['predict']('{}_class_to_{}'.format(i,j), epoch, pred[j]/sample_num)
            print('{}_class_to_{} : {}'.format(i,j, pred[j]/sample_num))

    #return test_scores.y_true, test_scores.y_score


# if __name__ == '__main__':
#     total = ScoreReport()
#     ac = []
#     for i in range(5):
#         yt, ys = main(i)
#         total.update_score(torch.Tensor(ys))
#         total.update_true(torch.Tensor(yt))

#     print(total.accuracy)
