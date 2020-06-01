from __future__ import print_function, division, absolute_import

import ori_utils, torch, time, os, pickle, itertools, argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from summary import Scalar, Image3D
from utils import SimpleTimer
from visdom import Visdom
from dataset import ADNIDataset, Trainset
from transforms_2d import RandomCrop, ToWoldCoordinateSystem, ToTensor, NineCrop, Resize, Flip, NineCropFlip


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
                        help='The name of dataset')
    parser.add_argument('--devices', type=int, nargs='+', default=(9,8,7,6))
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--num_epoch', type=int, default=10000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='result')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--vis_env', type=str, default='infoGAN')
    parser.add_argument('--subject_ids_path', type=str,
                        default=os.path.join('data', 'subject_ids.pkl'))
    parser.add_argument('--diagnosis_path', type=str,
                        default=os.path.join('data', 'diagnosis.pkl'))
    parser.add_argument('--labels', type=str, nargs='+', default=('AD','CN'))
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--running_fold', type=int, default=0)
    parser.add_argument('--z', type=int, default=64)

    return check_args(parser.parse_args())


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.len_discrete_code + self.len_continuous_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128*8*8),
            nn.BatchNorm1d(128*8*8),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            #nn.Tanh(),
            nn.Sigmoid(),
        )
        ori_utils.initialize_weights(self)

    def forward(self, input, cont_code, dist_code):
        x = torch.cat([input, cont_code, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv(x)

        return x

    

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            # nn.Sigmoid(),
        )
        ori_utils.initialize_weights(self)

    def forward(self, input):
        b = torch.ones(1).cuda()
        #print(input.shape)
        if input.shape[0] == 1:
            return b, b, b
        else :
            #print('input', input.shape)
            x = self.conv(input)
            #print('x', x.shape)
            x = x.view(-1, 128*8*8)
            x = self.fc(x)
            a = F.sigmoid(x[:, self.output_dim])
            b = x[:, self.output_dim:self.output_dim + self.len_continuous_code]
            c = x[:, self.output_dim + self.len_continuous_code:]

            return a, b, c


class infoGAN(object):
    def __init__(self, FG, SUPERVISED=True):
        # parameters
        self.num_epoch = FG.num_epoch
        self.batch_size = FG.batch_size
        self.save_dir = FG.save_dir
        self.result_dir = FG.result_dir
        self.dataset = 'MRI'
        self.log_dir = FG.log_dir
        self.model_name = 'infoGAN'
        self.input_size = FG.input_size
        self.z_dim = FG.z
        self.SUPERVISED = SUPERVISED        # if it is true, label info is directly used for code
        self.len_discrete_code = 10         # categorical distribution (i.e. label)
        self.len_continuous_code = 2        # gaussian distribution (e.g. rotation, thickness)
        self.sample_num = self.len_discrete_code ** 2
        
        # torch setting
        self.device = torch.device('cuda:{}'.format(FG.devices[0]))
        torch.cuda.set_device(FG.devices[0])
        timer = SimpleTimer()

        # load dataset
        x, y = Trainset(FG)      # x = image, y=target
        trainset = ADNIDataset(FG, x, y, cropping=NineCrop((40,40,40),(32,32,32)),
                               transform=Compose([Lambda(lambda patches: torch.stack([ToTensor()(patch) for patch in patches]))]))     
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=4)
        #self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        #data = self.trainloader
        for _, data in enumerate(self.trainloader):
            data = data['image']
            break

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1],
                           input_size=self.input_size, len_discrete_code=self.len_discrete_code,
                           len_continuous_code=self.len_continuous_code).to('cuda:{}'.format(FG.devices[0]))
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size,
                               len_discrete_code=self.len_discrete_code, len_continuous_code=self.len_continuous_code).to('cuda:{}'.format(FG.devices[0]))
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=FG.lrG, betas=(FG.beta1, FG.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=FG.lrD, betas=(FG.beta1, FG.beta2))
        self.info_optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr=FG.lrD, betas=(FG.beta1, FG.beta2))

        if len(FG.devices) != 1:
            self.G = torch.nn.DataParallel(self.G, FG.devices)
            self.D = torch.nn.DataParallel(self.D, FG.devices)
        self.BCE_loss = nn.BCELoss().to('cuda:{}'.format(FG.devices[0]))
        self.CE_loss = nn.CrossEntropyLoss().to('cuda:{}'.format(FG.devices[0]))
        self.MSE_loss = nn.MSELoss().to('cuda:{}'.format(FG.devices[0]))

        print('---------- Networks architecture -------------')
        ori_utils.print_network(self.G)
        ori_utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.len_discrete_code):
            self.sample_z[i * self.len_discrete_code] = torch.rand(1, self.z_dim)
            for j in range(1, self.len_discrete_code):
                self.sample_z[i * self.len_discrete_code + j] = self.sample_z[i * self.len_discrete_code]

        temp = torch.zeros((self.len_discrete_code, 1))
        for i in range(self.len_discrete_code):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.len_discrete_code):
            temp_y[i * self.len_discrete_code: (i + 1) * self.len_discrete_code] = temp

        self.sample_y = torch.zeros((self.sample_num, self.len_discrete_code)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        self.sample_c = torch.zeros((self.sample_num, self.len_continuous_code))

        # manipulating two continuous code
        #self.sample_z2 = torch.rand((1, self.z_dim)).expand(self.sample_num, self.z_dim)
        self.sample_z2 = torch.zeros((self.sample_num, self.z_dim))
        z2 = torch.rand(1, self.z_dim)
        for i in range(self.sample_num):
            self.sample_z2[i] = z2
        
        self.sample_y2 = torch.zeros(self.sample_num, self.len_discrete_code)
        self.sample_y2[:, 0] = 1

        temp_c = torch.linspace(-1, 1, 10)
        self.sample_c2 = torch.zeros((self.sample_num, 2))
        for i in range(self.len_discrete_code):
            for j in range(self.len_discrete_code):
                self.sample_c2[i*self.len_discrete_code+j, 0] = temp_c[i]
                self.sample_c2[i*self.len_discrete_code+j, 1] = temp_c[j]

        self.sample_z = self.sample_z.cuda(self.device, non_blocking=True)
        self.sample_y = self.sample_y.cuda(self.device, non_blocking=True) 
        self.sample_c = self.sample_c.cuda(self.device, non_blocking=True)
        self.sample_z2 = self.sample_z2.cuda(self.device, non_blocking=True)
        self.sample_y2 = self.sample_y2.cuda(self.device, non_blocking=True)
        self.sample_c2 = self.sample_c2.cuda(self.device, non_blocking=True)


        vis = Visdom(port=10002, env=str(FG.vis_env))

        self.printers = dict(
            D_loss = Scalar(vis, 'D_loss', opts=dict(
                showlegend=True, title='D loss', ytickmin=0, ytinkmax=2.0)),
            G_loss = Scalar(vis, 'G_loss', opts=dict(
                showlegend=True, title='G loss', ytickmin=0, ytinkmax=10)),
            info_loss = Scalar(vis, 'info_loss', opts=dict(
                showlegend=True, title='info loss', ytickmin=0, ytinkmax=10)),
            input = Image3D(vis, 'input'),
            input_fi = Image3D(vis, 'input_fi'),
            output = Image3D(vis, 'output'),
            output2 = Image3D(vis, 'output2'))

        self.timer = SimpleTimer()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['info_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real, self.y_fake = torch.ones(self.batch_size, 1).cuda(self.device, non_blocking=True), torch.zeros(self.batch_size, 1).cuda(self.device, non_blocking=True)

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.num_epoch):
            self.timer.tic()
            torch.set_grad_enabled(True)
            self.G.train()
            epoch_start_time = time.time()
            for iter, data in enumerate(self.trainloader):
                image = data['image']
                y = data['target']
#                 if iter == len(self.trainloader) // self.batch_size:
#                     break
                z = torch.rand((self.batch_size, self.z_dim))
                if self.SUPERVISED == True:
                    y_disc = torch.zeros((self.batch_size, self.len_discrete_code)).scatter_(1, y.type(torch.LongTensor).unsqueeze(1), 1)
                else:
                    y_disc = torch.from_numpy(
                        np.random.multinomial(1, self.len_discrete_code * [float(1.0 / self.len_discrete_code)],
                                              size=[self.batch_size])).type(torch.FloatTensor)

                y_cont = torch.from_numpy(np.random.uniform(-1, 1, size=(self.batch_size, 2))).type(torch.FloatTensor)

                z, y_disc, y_cont = z.cuda(self.device, non_blocking=True), \
                                    y_disc.cuda(self.device, non_blocking=True),\
                                    y_cont.cuda(self.device, non_blocking=True)
                ci = (np.array(image.shape)/2).astype(int)
                x = image[:,:,0,:,:,ci[5]].contiguous().cuda(self.device, non_blocking=True)
                flipped = []
                for cim in image:
                    fi = np.flip(cim, 0)
                    #fi = torch.from_numpy(fi.copy()).float()
                    flipped += [fi]
                    fi = np.flip(cim, 1)
                    flipped += [fi]
                
                for i in range(len(flipped)):
                    flipped[i] = torch.from_numpy(flipped[i].copy()).float()
                flipped = torch.stack(flipped)
                flipped = flipped[:,:,0,:,:,ci[5]].contiguous().cuda(self.device, non_blocking=True)

                """################ Update D network ################"""
                self.D_optimizer.zero_grad()
                self.printers['input']('input', x[1,:,:,:])
                
                D_real, _, _ = self.D(x)
                if D_real.shape[0] == 1:
                     break
                batch_size = D_real.size(0)
                self.y_real.resize_(batch_size).fill_(1)
                D_real_loss = self.BCE_loss(D_real, self.y_real)
                #print(D_real.shape, self.y_real.shape)
                
                self.printers['input_fi']('input_fi', flipped[1,:,:,:])
                
                D_real, _, _ = self.D(flipped)
                if D_real.shape[0] == 1:
                     break
                batch_size = D_real.size(0)
                self.y_real.resize_(batch_size).fill_(1)
                D_real_loss = self.BCE_loss(D_real, self.y_real)

                fake = self.G(z, y_cont, y_disc)
            
                D_fake, _, _ = self.D(fake)
                batch_size = D_fake.size(0)
                self.y_fake.resize_(batch_size).fill_(0)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward(retain_graph=True)
                self.D_optimizer.step()

                """################ Update G network ################"""
                self.G_optimizer.zero_grad()

                fake = self.G(z, y_cont, y_disc)
                #print(fake.shape)
                
                D_fake, D_cont, D_disc = self.D(fake)
                batch_size = D_fake.size(0)
                #print(D_fake.shape, fake.shape)
                #self.y_real.resize_(batch_size).fill_(1)
                y_real = torch.ones(batch_size).cuda(self.device, non_blocking=True)
                #print(D_fake.shape, self.y_real.shape)
                G_loss = self.BCE_loss(D_fake, y_real)
                
                self.train_hist['G_loss'].append(G_loss.item())
                
                #with torch.enable_grad():
                G_loss.backward(retain_graph=True)
                self.G_optimizer.step()

                # information loss
                disc_loss = self.CE_loss(D_disc, torch.max(y_disc, 1)[1])
                cont_loss = self.MSE_loss(D_cont, y_cont)
                info_loss = disc_loss + cont_loss
                self.train_hist['info_loss'].append(info_loss.item())

                with torch.enable_grad():
                    info_loss.backward()
                self.info_optimizer.step()

                self.printers['D_loss']('D_fake_loss', epoch+iter/len(self.trainloader),
                                        D_fake_loss)
                self.printers['D_loss']('D_real_loss', epoch+iter/len(self.trainloader),
                                        D_real_loss)
                self.printers['D_loss']('D_loss', epoch+iter/len(self.trainloader), D_loss)

                self.printers['G_loss']('G_loss', epoch+iter/len(self.trainloader), G_loss)

                self.printers['info_loss']('disc_loss', epoch+iter/len(self.trainloader), disc_loss)
                self.printers['info_loss']('cont_loss', epoch+iter/len(self.trainloader), cont_loss)
                self.printers['info_loss']('info_loss', epoch+iter/len(self.trainloader), info_loss)

                fake = fake[1,:,:,:]
                self.printers['output']('ori_output', fake)
                self.printers['output2']('output2', fake)


                if ((iter + 1) % 10) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %
                          ((epoch + 1), (iter + 1), len(self.trainloader) // self.batch_size, D_loss.item(), G_loss.item(), info_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
#             if ((epoch) % 10) == 0:
#                 with torch.no_grad():
#                     self.visualize_results((epoch+1))
#                 self.loss_plot(self.train_hist,
#                        os.path.join(self.save_dir, self.dataset, self.model_name),
#                        self.model_name)
            self.timer.toc()
            print('Time elapse {}h {}m {}s'.format(*self.timer.total()))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f"
              %(np.mean(self.train_hist['per_epoch_time']), self.epoch,
              self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        ori_utils.generate_animation(self.result_dir+'/'+self.dataset+'/'
                                 +self.model_name+'/'+self.model_name,
                                 self.num_epoch)
        ori_utils.generate_animation(self.result_dir+'/'+self.dataset+'/'
                                 +self.model_name+'/'+self.model_name+'_cont',
                                 self.num_epoch)
        self.loss_plot(self.train_hist,
                       os.path.join(self.save_dir, self.dataset, self.model_name),
                       self.model_name)


    def visualize_results(self, epoch):
        self.G.eval()

        if not os.path.exists(self.result_dir+'/'+self.dataset+'/'+self.model_name):
            os.makedirs(self.result_dir+'/'+self.dataset+'/'+self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        """ style by class """
        samples = self.G(self.sample_z, self.sample_c, self.sample_y)       
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        
        #samples.shape = 100,32,32,9
        samples = (samples + 1) / 2
        ori_utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :,:],
                              [image_frame_dim, image_frame_dim],
                              self.save_dir+'/info-2D/disc_epoch%03d'%epoch+'.png')

        """ manipulating two continous codes """
        samples = self.G(self.sample_z2, self.sample_c2, self.sample_y)

        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        ori_utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                              [image_frame_dim, image_frame_dim],
                              self.save_dir+'/info-2D/cont_epoch%03d'%epoch+'.png')


    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir,self.model_name
                                                     + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir,self.model_name
                                                     + '_D.pkl'))

        with open(os.path.join(save_dir,self.model_name+'_history.pkl'),'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name
                                                       + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name
                                                       + '_D.pkl')))

    def loss_plot(self, hist, path='Train_hist.png', model_name=''):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        y3 = hist['info_loss']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')
        plt.plot(x, y3, label='info_loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(path, model_name + '_loss.png')

        plt.savefig(path)




"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.num_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


if __name__ == '__main__':
    # parse arguments
    FG = parse_args()
    if FG is None:
        exit()

    if FG.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    gan = infoGAN(FG, SUPERVISED=False)


    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    gan.visualize_results(FG.num_epoch)
    print(" [*] Testing finished!")
