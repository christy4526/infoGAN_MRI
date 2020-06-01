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
from models import ResGenerator, ResDiscriminator, Generator, Discriminator, Plane
from models import sliceGenerator, sliceDiscriminator
from models import ResGenerator2D, ResDiscriminator2D, bigiGenerator, bigiDiscriminator
from dataset import ADNIDataset, Trainset
from transforms import RandomCrop, ToWoldCoordinateSystem, ToTensor, Normalize
from utils import print_model_parameters, ScoreReport, SimpleTimer, loss_plot
from utils import save_checkpoint #, save_images
from summary import Scalar, Image3D
import itertools
import numpy as np
import imageio


#def create_merge_images(epoch, snum, G, z, c, y, save_dir, name):
#    G.eval()
#    iframe_dim = int(np.floor(np.sqrt(snum)))
#    ###### style by class #####
#    samples = G(z, c, y)
#    #samples = samples/samples.max()
#    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 4, 1)
#    #samples = (samples+1)/2
#    samples = samples[:iframe_dim*iframe_dim, :, :, :, :]

#    save_images(samples, [iframe_dim, iframe_dim, iframe_dim],
#               save_dir+'/'+name+'_w,h_epoch_'+str(epoch)+'.png', 'wh')
#    save_images(samples, [iframe_dim, iframe_dim, iframe_dim],
#                save_dir+'/'+name+'_h,d_epoch_'+str(epoch)+'.png', 'hd')
#    save_images(samples, [iframe_dim, iframe_dim, iframe_dim],
#                save_dir+'/'+name+'_d,w_epoch_'+str(epoch)+'.png', 'dw')

def save_images(images, size, image_path):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        #print('idx : ', idx)
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return imageio.imwrite(image_path, img)



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
            showlegend=True, title='lr', ytickmin=0, ytinkmax=1.0)),
        D_loss = Scalar(vis, 'D_loss', opts=dict(
            showlegend=True, title='D loss', ytickmin=0, ytinkmax=2.0)),
        G_loss = Scalar(vis, 'G_loss', opts=dict(
            showlegend=True, title='G loss', ytickmin=0, ytinkmax=10)),
        info_loss = Scalar(vis, 'info_loss', opts=dict(
            showlegend=True, title='info loss', ytickmin=0, ytinkmax=10)),
        input = Image3D(vis, 'input'),
        output = Image3D(vis, 'output'),
        cont_output = Image3D(vis, 'cont_output'),
        disc_output = Image3D(vis, 'disc_output'))

    # create train set
    x, y = Trainset(FG)      # x = image, y=target
    if FG.gm == 'true':
        transform=Compose([ToWoldCoordinateSystem(), ToTensor()])
    else :
        transform=Compose([ToWoldCoordinateSystem(), Normalize(0.2,0.9), ToTensor()])
    trainset = ADNIDataset(FG, x, y, transform=transform)
    trainloader = DataLoader(trainset, batch_size=FG.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=4)

    for _, data in enumerate(trainloader):
        data = data['image']
        break
    if '3d' in FG.model:
        G = Generator(FG).to('cuda:{}'.format(FG.devices[0]))
        D = Discriminator(FG).to('cuda:{}'.format(FG.devices[0]))
    elif '2d' in FG.model:
        G = sliceGenerator(FG, output_dim=data.shape[1]).to('cuda:{}'.format(FG.devices[0]))
        D = sliceDiscriminator(FG, input_dim=data.shape[1]).to('cuda:{}'.format(FG.devices[0]))
    elif 'resnet' in FG.model:
        G = ResGenerator(FG).to('cuda:{}'.format(FG.devices[0]))
        D = ResDiscriminator(FG).to('cuda:{}'.format(FG.devices[0]))
    elif 'mix' in FG.model:
        G = ResGenerator(FG).to('cuda:{}'.format(FG.devices[0]))
        D = Discriminator(FG).to('cuda:{}'.format(FG.devices[0]))
    elif 'bigi' in FG.model:
        G = bigiGenerator(FG).to('cuda:{}'.format(FG.devices[0]))
        D = bigiDiscriminator(FG).to('cuda:{}'.format(FG.devices[0]))
    elif '2dmix-1' in FG.model:
        G = ResGenerator2D(FG, output_dim=data.shape[1]).to('cuda:{}'.format(FG.devices[0]))
        D = bigiDiscriminator(FG, input_dim=data.shape[1]).to('cuda:{}'.format(FG.devices[0]))
    elif '2dmix-2' in FG.model:
        G = ResGenerator2D(FG, output_dim=data.shape[1]).to('cuda:{}'.format(FG.devices[0]))
        D = sliceDiscriminator(FG, input_dim=data.shape[1]).to('cuda:{}'.format(FG.devices[0]))
    else:
        raise NotImplementedError(FG.model)

    # print(G, D)
    #print_model_parameters(G)
    #print_model_parameters(D)

    if len(FG.devices) != 1:
        G = torch.nn.DataParallel(G, FG.devices)
        D = torch.nn.DataParallel(D, FG.devices)

    # if FG.load_pkl == 'true':
    #     G.load_state_dict(torch.load(os.path.join(FG.save_dir, 'G.pth')))
    #     D.load_state_dict(torch.load(os.path.join(FG.save_dir, 'D.pth')))

    BCE_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
    CE_loss = nn.CrossEntropyLoss().to('cuda:{}'.format(FG.devices[0]))
    MSE_loss = nn.MSELoss().to('cuda:{}'.format(FG.devices[0]))
    # setup optimizer
    optimizerD = optim.Adam(D.parameters(), lr=FG.lrD, betas=(FG.beta1, FG.beta2))
    optimizerG = optim.Adam(G.parameters(), lr=FG.lrG, betas=(FG.beta1, FG.beta2))
    optimizerInfo = optim.Adam(itertools.chain(G.parameters(), D.parameters()),
                               lr=FG.lrD, betas=(FG.beta1, FG.beta2))
    schedularD=torch.optim.lr_scheduler.ExponentialLR(optimizerD, FG.lr_gamma)
    schedularG=torch.optim.lr_scheduler.ExponentialLR(optimizerG, FG.lr_gamma)
    schedularInfo=torch.optim.lr_scheduler.ExponentialLR(optimizerInfo, FG.lr_gamma)

    # fixed noise & condition
    discrete_code = FG.d_code  # categorical distribution (i.e. label=AD,MCI,CN)
    continuous_code = FG.c_code  # gaussian distribution (e.g. rotation, thickness)
    sample_num = discrete_code ** 2
    z_dim = FG.z

    sample_z = torch.zeros((sample_num, z_dim))
    for i in range(discrete_code):
        sample_z[i*discrete_code] = torch.rand(1, z_dim)
        for j in range(1, discrete_code):
            sample_z[i*discrete_code+j] = sample_z[i*discrete_code]

    temp = torch.zeros((discrete_code, 1))
    for i in range(discrete_code):
        temp[i, 0] = i # tensor([[ 0.],[ 1.],[ 2.]]) : label

    temp_y = torch.zeros((sample_num, 1))
    for i in range(discrete_code): #0~2
        temp_y[i*discrete_code: (i+1)*discrete_code] = temp

    sample_y = torch.zeros((sample_num, discrete_code)).scatter_(1,temp_y.type(torch.LongTensor), 1)
    sample_c = torch.zeros((sample_num, continuous_code))

    ### manipulating two continuous code
    sample_z2 = torch.zeros((sample_num, z_dim))
    z2 = torch.rand(1, z_dim)
    print(z2.shape)
    for i in range(sample_num):
        sample_z2[i] = z2

    sample_y2 = torch.zeros((sample_num, discrete_code))  # torch.Size([9, 3])
    sample_y2[:, 0] = 1
    temp_c = torch.linspace(-1, 1, discrete_code)
    sample_c2 = torch.zeros((sample_num, continuous_code))
    for i in range(discrete_code):
        for j in range(discrete_code):
            sample_c2[i*discrete_code+j, 0] = temp_c[i]
            sample_c2[i*discrete_code+j, 1] = temp_c[j]


    sample_z = sample_z.cuda(device, non_blocking=True)
    sample_y = sample_y.cuda(device, non_blocking=True)
    sample_c = sample_c.cuda(device, non_blocking=True)
    sample_z2 = sample_z2.cuda(device, non_blocking=True)
    sample_y2 = sample_y2.cuda(device, non_blocking=True)
    sample_c2 = sample_c2.cuda(device, non_blocking=True)
    y_real = torch.ones(FG.batch_size, 1).cuda(device, non_blocking=True)
    y_fake = torch.zeros(FG.batch_size, 1).cuda(device, non_blocking=True)

    D.train()
    #D_loss, G_loss, info_loss, disc_loss, cont_loss = 0,0,0,0,0
    train_hist = {}
    train_hist['D_loss'] = []
    train_hist['G_loss'] = []
    train_hist['info_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []
    epoch_start_time = time.time()

    print('--------- Start training! ---------')
    start_time = time.time()
    for epoch in range(FG.num_epoch):
        timer.tic()
        if (epoch+1)%100 == 0:
            schedularD.step()
            schedularG.step()
            schedularInfo.step()

        printers['lr']('D_lr', epoch, optimizerD.param_groups[0]['lr'])
        printers['lr']('G_lr',  epoch, optimizerG.param_groups[0]['lr'])
        printers['lr']('info_lr', epoch, optimizerInfo.param_groups[0]['lr'])
        torch.set_grad_enabled(True)
        G.train(True)
        for i, data in enumerate(trainloader):
            z = torch.rand((FG.batch_size, z_dim))
            x = data['image']
            y = data['target']

            if FG.SUPERVISED == True:
                # y_disc = torch.zeros((FG.batch_size, discrete_code)).scatter_(1,\
                #                         y.type(torch.LongTensor).unsqueeze(1), 1)
                y_disc = torch.zeros((FG.batch_size, discrete_code)).scatter_(1,\
                                    torch.zeros(y.shape).type(torch.FloatTensor).unsqueeze(1), 1)
            else:
                y_disc = torch.from_numpy(
                    np.random.multinomial(1, discrete_code*[float(1.0/discrete_code)],
                                          size=[FG.batch_size])).type(torch.FloatTensor)
            #y_cont = torch.from_numpy(np.random.uniform(-1, 1, size=(FG.batch_size, 2))).type(torch.FloatTensor)
            y_cont = torch.from_numpy(np.random.uniform(-2, 2, size=(FG.batch_size, continuous_code))).type(torch.FloatTensor)

            # addi = torch.zeros((xi.shape[0]*5,1,156,156))
            # si=[s for s in range(92, 96)]
            # #print('x:', x.shape)
            # if FG.isize == 157:
            #     for i in range(xi.shape[0]):
            #         for j in range(len(si)):
            #             #print('addi:',addi[i*len(si)+j].shape)
            #             #print('x:',x[i,:,:,si[j],:].shape)
            #             addi[i*len(si)+j] = xi[i,:,:,si[j],:]
            #     x = torch.zeros((x.shape[0]*5,1,156,156))
            #     x = addi
            x, z, y_disc, y_cont = x.cuda(device, non_blocking=True),\
                                   z.cuda(device, non_blocking=True),\
                                   y_disc.cuda(device, non_blocking=True),\
                                   y_cont.cuda(device, non_blocking=True)

            """#############;,### Update D network ################"""
            optimizerD.zero_grad()
            #print("input min : %.5f, max: %.5f" % (x.min(), x.max()))

            #D_real, _, _ = D(x)
            #D_real, _ = D(x, FG.axis)
            D_real, _ = D(x)
            if D_real.shape[0] == 1:
                 break
            batch_size = D_real.size(0)
            y_real.resize_(batch_size).fill_(1)
            D_real_loss = BCE_loss(D_real, y_real)

            #fake = G(z, y_cont, y_disc, FG.axis)'
            #fake = G(z, y_cont, FG.axis)
            fake = G(z, y_cont)
            if fake.shape[0] == 1:
                 break

            #D_fake, _, _ = D(fake)
            #D_fake, _ = D(fake, FG.axis)
            D_fake, _ = D(fake)
            batch_size = fake.size(0)
            y_fake.resize_(batch_size).fill_(0)
            D_fake_loss = BCE_loss(D_fake, y_fake)

            D_loss = D_real_loss + D_fake_loss
            train_hist['D_loss'].append(D_loss.item())

            D_loss.backward(retain_graph=True)
            optimizerD.step()


            """################ Update G network ################"""
            #G.zero_grad()
            optimizerG.zero_grad()

            #fake = G(z, y_cont, y_disc, FG.axis)
            #fake = G(z, y_cont, FG.axis)
            fake = G(z, y_cont)
            fake = torch.clamp(fake/fake.max(), min=0, max=1)

            #printers['output']('output', fake)
            for j in range(5):
                printer = Image3D(vis, 'output'+str(j))
                #printer('output'+str(j), fake[j,:,:,:])
                printer('output'+str(j), fake)
            if ((epoch+1) % 20 == 0):
                saver = Image3D(vis, 'output_'+str(epoch+1))
                #saver('output_'+str(epoch+1),fake[0,:,:,:])
                saver('output_'+str(epoch+1),fake)

            #D_fake, D_cont, D_disc = D(fake)
            #D_fake, D_cont = D(fake, FG.axis)
            D_fake, D_cont = D(fake)
            batch_size = D_fake.size(0)
            y_real.resize_(batch_size).fill_(1)
            G_loss = BCE_loss(D_fake, y_real)
            train_hist['G_loss'].append(G_loss.item())

            G_loss.backward(retain_graph=True)
            optimizerG.step()

            # information loss
            #disc_loss = CE_loss(D_disc, torch.max(y_disc, 1)[1])
            cont_loss = MSE_loss(D_cont, y_cont)
            #info_loss = disc_loss + cont_loss
            info_loss = cont_loss
            train_hist['info_loss'].append(info_loss.item())

            info_loss.backward()
            optimizerInfo.step()

            printers['D_loss']('D_fake_loss', epoch+i/len(trainloader),
                                    D_fake_loss)
            printers['D_loss']('D_real_loss', epoch+i/len(trainloader),
                                    D_real_loss)
            printers['D_loss']('D_loss', epoch+i/len(trainloader), D_loss)
            printers['G_loss']('G_loss', epoch+i/len(trainloader), G_loss)

            #printers['info_loss']('disc_loss', epoch+i/len(trainloader), disc_loss)
            #printers['info_loss']('cont_loss', epoch+i/len(trainloader), cont_loss)
            printers['info_loss']('info_loss', epoch+i/len(trainloader), info_loss)

            """
            fake = G(sample_z, sample_c, sample_y)
            if fake.shape[0] == 1:
                break
            printers['cont_output']('cont_output', torch.clamp(fake/fake.max(), min=0, max=1))

            fake = G(sample_z2, sample_c2, sample_y2)
            if fake.shape[0] == 1:
                break
            printers['disc_output']('disc_output', torch.clamp(fake/fake.max(), min=0, max=1))
            """
            print("Epoch: [%2d] D_loss: %.5f, G_loss: %.5f, info_loss: %.5f" %
                 ((epoch + 1), D_loss.item(), G_loss.item(), info_loss.item()))
            result_dict = {"D_loss":D_loss, "G_loss":G_loss,"info_loss":info_loss,
                          }#"disc_loss":disc_loss,"cont_loss":cont_loss}


        # if ((epoch+1) % 10 == 0):
        #     with torch.no_grad():
        #         G.eval()
        #         iframe_dim = int(np.floor(np.sqrt(sample_num)))
        #         ###### style by class #####
        #         #samples = G(sample_z, sample_c, sample_y, FG.axis)
        #         #samples = G(sample_z, sample_c, FG.axis)
        #         samples = G(sample_z, sample_c)
        #
        #         samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        #         samples = (samples+1)/2
        #         samples = samples[:iframe_dim*iframe_dim, :, :, :]
        #
        #         for k in range(samples.shape[3]):
        #             save_images(samples, [iframe_dim, iframe_dim],
        #                             FG.save_dir+'/disc_'+str(k+1)+'_epoch_'+str(epoch+1)+'.png')
        #
        #         # manipulating two continous codes
        #         #samples = G(sample_z2, sample_c2, sample_y2, FG.axis)
        #         #samples = G(sample_z2, sample_c2, FG.axis)
        #         samples = G(sample_z2, sample_c2)
        #         samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        #         samples = (samples+1)/2
        #         samples = samples[:iframe_dim*iframe_dim, :, :, :]
        #         save_images(samples, [iframe_dim, iframe_dim],
        #                    FG.save_dir+'/cont_w,h_epoch_'+str(epoch+1)+'.png')

        if ((epoch+1) % 10 == 0):
            with torch.no_grad():
                torch.save(G.state_dict(), '%s/G_%d.pth' % (FG.save_dir, epoch+1))
                torch.save(D.state_dict(), '%s/D_%d.pth' % (FG.save_dir, epoch+1))



        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        timer.toc()
        print('Time elapse {}h {}m {}s'.format(*timer.total()))

    train_hist['total_time'].append(time.time() - start_time)
    print("Avg one epoch time: %.2f, total %d epochs time: %.2f"
          %(np.mean(train_hist['per_epoch_time']), epoch, train_hist['total_time'][0]))
    print("Training finish!... save training results")
