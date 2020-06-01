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
from transforms_2d import RandomCrop, ToWoldCoordinateSystem, ToTensor, FiveCrop, Normalize

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
    vis.text(argument_report(FG, end='<br>'), win='config')
    #FG.running_fold = rf

    # main device
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])

    # make model
    model = Plane2D(FG, len(FG.labels))
    #print('Copying model to GPU')
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=FG.devices)
    #fname = 'Plane2D_79'
    fname = 'plane-5crop'
    fname = 'Plane2D_79_'+str(FG.axis)
    if FG.isize == 157:
        fname = 'Plane2D_157_'+str(FG.axis)
    load_checkpoint(FG.checkpoint_root, FG.running_fold,
                    fname, model.module, epoch=None, is_best=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    G = sliceGenerator(FG).to('cuda:{}'.format(FG.devices[0]))
    G = torch.nn.DataParallel(G, FG.devices)
    if FG.isize == 79:
        G.load_state_dict(torch.load(os.path.join('result_2d_classify', 'G_79.pth')))
    else:
        G.load_state_dict(torch.load(os.path.join('result_2d_classify', 'G.pth')))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrD, betas=(0.5, 0.999))

    # make dataset
    x, y, train_idx, test_idx, ratio = fold_split(FG)
    # original dataset
    testset = ADNIDataset2D(FG, x[test_idx], y[test_idx], transform=Compose([
                            #ToWoldCoordinateSystem(), ToTensor()]))
                            ToWoldCoordinateSystem(),RandomCrop(80,64),
                            ToTensor()]))
    # dataloader
    testloader = DataLoader(testset, num_workers=4, pin_memory=True)

    model.eval()
    G.eval()
    torch.set_grad_enabled(False)
    ############################ original : test #############################
    test_scores = ScoreReport()
    test_scores.clear()
    print('original image result')
    for sample in testloader:
        mri = sample['image']
        target = sample['target']
        mri = mri.cuda(device, non_blocking=True)
        target = target.to(device)

        name = 'ori-target'
        save_printer1 = Image3D(vis, name)
        save_printer1(name, torch.clamp(mri[0,:,:,:], min=0, max=1))

        output = model(mri)

        score = torch.nn.functional.softmax(output, 1)
        test_scores.update_true(target)
        test_scores.update_score(score)

    print('accuracy:', test_scores.accuracy)

    ############################ cont test ##################################
    # create latent codes
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


    # idx=0
    # for i in range(len(temp_c)):
    #     for j in range(25):
    #         sample_c[i*25+j, 0] = temp_c[idx]
    #     idx+=1
    # for i in range(sample_num):
    #     sample_c[i, 2] = temp_c[i%len(temp_c)]
    #
    # idx=0
    # for i in range(len(temp_c)**2):
    #     for j in range(5):
    #         sample_c[i*5+j, 1] = temp_c[idx%len(temp_c)]
    #     idx+=1
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
        name = 'target_'+str(sample_c[i, 0].item())+':'+str(sample_c[i, 1].item())+':'+str(sample_c[i, 2].item())
        save_printer1 = Image3D(vis, name)
        save_printer1(name, torch.clamp(gi_c1[i,:,:,:], min=0, max=1))
    exit()



    """######################################################################"""
    meani_ad = torch.zeros((5,12500,1,95,79))
    meani_nc = torch.zeros((5,12500,1,95,79))
    aidx=[0,0,0,0,0]
    nidx=[0,0,0,0,0]

    for t in range(5):
        sample_c[:, 0] = temp_c[t]
        for t2 in range(5):
            sample_c[:, 1] = temp_c[t2]
            for t3 in range(5):
                sample_c[:, 2] = temp_c[t3]
                sample_c = sample_c.cuda(device, non_blocking=True)
                gi_c1 = G(sample_z, sample_c, FG.axis)
                gtest_scores = ScoreReport()
                gtest_scores.clear()

                #print('Result:'+str(sample_c[0, 0].item())+':'+str(sample_c[0, 1].item())+':'+str(sample_c[0, 2].item()))
                # for i in range(sample_num):
                #     name = 'target_'+str(i)+':'+str(sample_c[i, 0].item())+':'+str(sample_c[i, 1].item())+':'+str(sample_c[i, 2].item())
                #     save_printer1 = Image3D(vis, name)
                #     save_printer1(name, torch.clamp(gi_c1[i,:,:,:], min=0, max=1))

                # for i in range(5):
                #     name = 'target_'+str(i)+':'+str(sample_c[i, 0].item())+':'+str(sample_c[i, 1].item())+':'+str(sample_c[i, 2].item())
                #     save_printer1 = Image3D(vis, name)
                #     save_printer1(name, torch.clamp(gi_c1[i,:,:,:], min=0, max=1))

                name = 'target_'+str(0)+':'+str(sample_c[0, 0].item())+':'+str(sample_c[0, 1].item())+':'+str(sample_c[0, 2].item())
                save_printer1 = Image3D(vis, name)
                save_printer1(name, torch.clamp(gi_c1[0,:,:,:], min=0, max=1))

                if i == 0:
                    target = torch.zeros(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)
                elif i == 1:
                    target = torch.ones(sample_num).type(torch.LongTensor).cuda(device, non_blocking=True)

                output = model(gi_c1)
                score=torch.nn.functional.softmax(output, 1)

                y_score, y_pred=[],[]
                pred = np.zeros(discrete_code)
                confidence = torch.zeros((5,2))

                r = torch.linspace(0.9, 0.5, 5)
                # for j in range(sample_num):
                #     if score[j][0].item() >= 0.85:
                #         name = 'c0_'+str(j)+'_'+str(score[j][0].item())
                #         sprint = Image3D(vis, name)
                #         sprint(name, gi_c1[j,:,:,:])
                #
                #     elif score[j][1].item() >= 0.850:
                #         name = 'c1_'+str(j)+'_'+str(score[j][1].item())
                #         sprint = Image3D(vis, name)
                #         sprint(name, gi_c1[j,:,:,:])


                # for j in range(sample_num):
                #     print(sample_c[j, 0].item(),sample_c[j, 1].item(),sample_c[j, 2].item(),
                #         score[j][0].item(),score[j][1].item())


                for j in range(sample_num):
                    if score[j][0].item() >= 0.90:
                        confidence[0,0] += 1
                        meani_ad[0,aidx[0],:,:,:] = gi_c1[j,:,:,:]
                        aidx[0] += 1
                    elif score[j][0].item() >= 0.8:
                        confidence[1,0] += 1
                        meani_ad[1,aidx[1],:,:,:] = gi_c1[j,:,:,:]
                        aidx[1] += 1
                    elif score[j][0].item() >= 0.7:
                        confidence[2,0] += 1
                        meani_ad[2,aidx[2],:,:,:] = gi_c1[j,:,:,:]
                        aidx[2] += 1
                    elif score[j][0].item() >= 0.6:
                        confidence[3,0] += 1
                        meani_ad[3,aidx[3],:,:,:] = gi_c1[j,:,:,:]
                        aidx[3] += 1
                    elif score[j][0].item() >= 0.5:
                        confidence[4,0] += 1
                        meani_ad[4,aidx[4],:,:,:] = gi_c1[j,:,:,:]
                        aidx[4] += 1

                    if score[j][1].item() >= 0.9:
                        confidence[0,1] += 1
                        meani_nc[0,nidx[0],:,:,:] = gi_c1[j,:,:,:]
                        nidx[0] += 1
                    elif score[j][1].item() >= 0.8:
                        confidence[1,1] += 1
                        meani_nc[1,nidx[1],:,:,:] = gi_c1[j,:,:,:]
                        nidx[1] += 1
                    elif score[j][1].item() >= 0.7:
                        confidence[2,1] += 1
                        meani_nc[2,nidx[2],:,:,:] = gi_c1[j,:,:,:]
                        nidx[2] += 1
                    elif score[j][1].item() >= 0.6:
                        confidence[3,1] += 1
                        meani_nc[3,nidx[3],:,:,:] = gi_c1[j,:,:,:]
                        nidx[3] += 1
                    elif score[j][1].item() >= 0.5:
                        confidence[4,1] += 1
                        meani_nc[4,nidx[4],:,:,:] = gi_c1[j,:,:,:]
                        nidx[4] += 1

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
                # for j in range(discrete_code):
                #     #print('c{}_class_to_{} : {}'.format(i,j, pred[j]/sample_num))
                #     print('confidence_c{} 0.9:{}, 0.8:{}, 0.7:{}, 0.6:{}, 0.5:{}'.format(j,
                #         confidence[0,j], confidence[1,j], confidence[2,j], confidence[3,j], confidence[4,j]))
    vis = Visdom(port=10002, env=str(FG.vis_env))
    #print(meani_ad[0,:aidx[0]].shape)
    for i in range(5):
        #ad_mean = torch.mean(meani_ad[i,:aidx[i]], dim=0)
        #cn_mean = torch.mean(meani_nc[i,:nidx[i]], dim=0)
        print(meani_ad[i,:aidx[i]].shape)
        printer = Image3D(vis, 'mean_ad'+str(i))
        printer('AD'+str(i), torch.mean(meani_ad[i,:aidx[i]], dim=0), nimages=1)
        print(meani_nc[i,:nidx[i]].shape)
        #printer = Image3D(vis, 'mean_nc'+str(i))
        #printer('CN'+str(i), torch.mean(meani_nc[i,:nidx[i]], dim=0), nimages=1)

    exit()
