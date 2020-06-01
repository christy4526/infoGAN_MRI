import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from visdom import Visdom
from config import train_args, argument_report
import o_utils, utils, time, os, pickle, itertools
from utils import ScoreReport, SimpleTimer
from summary import Image3D, Scalar


class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code
        self.len_continuous_code = len_continuous_code

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.len_discrete_code + self.len_continuous_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        o_utils.initialize_weights(self)

    def forward(self, input, cont_code, dist_code):
        x = torch.cat([input, cont_code, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code
        self.len_continuous_code = len_continuous_code

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            # nn.Sigmoid(),
        )
        o_utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        a = F.sigmoid(x[:, self.output_dim])
        b = x[:, self.output_dim:self.output_dim + self.len_continuous_code]
        c = x[:, self.output_dim + self.len_continuous_code:]

        return a, b, c


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    FG = train_args()
    vis = Visdom(port=FG.vis_port, env=str(FG.vis_env))
    vis = Visdom(port=10002, env=str(FG.vis_env))

    # torch setting
    device = torch.device('cuda:{}'.format(FG.devices[0]))
    torch.cuda.set_device(FG.devices[0])
    timer = SimpleTimer()

    # Hyperparameters
    num_epochs = 100
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    #batch_size = FG.batch_size
    save_dir = FG.save_dir
    input_size = 28
    z_dim = 62
    SUPERVISED = 'True'
    len_discrete_code = 10
    len_continuous_code = 2
    #sample_num = len_discrete_code ** 2
    sample_num = 100

    DATA_PATH = 'data'
    MODEL_STORE_PATH = 'mnist_classify_'

    FG.save_dir = str(FG.vis_env)
    if not os.path.exists(FG.save_dir):
            os.makedirs(FG.save_dir)


    printers = dict(
        lr = Scalar(vis, 'learning_rate', opts=dict(
            showlegend=True, title='learning rate', ytickmin=0, ytinkmax=1)),
        loss = Scalar(vis, 'loss', opts=dict(
            showlegend=True, title='loss', ytickmin=0, ytinkmax=3)),
        loss_G = Scalar(vis, 'loss_G', opts=dict(
            showlegend=True, title='loss_G', ytickmin=0, ytinkmax=3)),
        acc = Scalar(vis, 'acc', opts=dict(
            showlegend=True, title='acc', ytickmin=0, ytinkmax=3)),
        accuracy = Scalar(vis, 'accuracy', opts=dict(
            showlegend=True, title='accuracy', ytickmin=0, ytinkmax=3)),
        predict = Scalar(vis, 'predict', opts=dict(
            showlegend=True, title='predict', ytickmin=0, ytinkmax=3)),
        output0 = Image3D(vis, 'output0'),
        output1 = Image3D(vis, 'output1'),
        output2 = Image3D(vis, 'output2'),
        output3 = Image3D(vis, 'output3'),
        output4 = Image3D(vis, 'output4'),
        output5 = Image3D(vis, 'output5'),
        output6 = Image3D(vis, 'output6'),
        output7 = Image3D(vis, 'output7'),
        output8 = Image3D(vis, 'output8'),
        output9 = Image3D(vis, 'output9')
    )

    # transforms to apply to the data
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvNet().to('cuda:{}'.format(FG.devices[0]))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ############# set generator ################
    G = generator(input_dim=z_dim, output_dim=1, input_size=input_size, len_discrete_code=len_discrete_code,
                  len_continuous_code=len_continuous_code).to('cuda:{}'.format(FG.devices[0]))
    #G.load_state_dict(torch.load(os.path.join(FG.save_dir, 'infoGAN_G.pkl')))

    sample_c = torch.zeros((sample_num, len_continuous_code))
    sample_c = sample_c.cuda(device, non_blocking=True)

    ######################### training ###########################
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    train_scores = ScoreReport()
    test_scores = ScoreReport()
    for epoch in range(num_epochs):
        sample_z = torch.zeros((sample_num, z_dim))
        for i in range(len_discrete_code):
            sample_z[i * len_discrete_code] = torch.rand(1, z_dim)
            for j in range(1, len_discrete_code):
                sample_z[i * len_discrete_code + j] = sample_z[i * len_discrete_code]
        sample_z = sample_z.cuda(device, non_blocking=True)

        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            images = images.cuda(device, non_blocking=True)
            print(images.shape)
            exit()
            labels = labels.cuda(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)
            printers['loss']('train', epoch+i/len(train_loader), loss)

            score = torch.nn.functional.softmax(outputs, 1)
            train_scores.update_true(labels)
            train_scores.update_score(score)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))
        printers['acc']('train', epoch, train_scores.accuracy)

        ######################### testing ###########################
        # Test the model
        model.eval()
        G.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.cuda(device, non_blocking=True)
                labels = labels.cuda(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                score = torch.nn.functional.softmax(outputs, 1)

                test_scores.update_true(labels)
                test_scores.update_score(score)
                test_scores.update_loss(loss)

            print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
            printers['acc']('test_ori', epoch, test_scores.accuracy)

        for i in range(len_discrete_code):
            y_score, y_pred=[],[]
            pred = np.zeros(10)

            temp_y = torch.zeros((sample_num, len_discrete_code))
            for j in range(sample_num):
                temp_y[j][i] = 1
            sample_y = temp_y
            sample_y = sample_y.cuda(device, non_blocking=True)

            gimage = G(sample_z, sample_c, sample_y)
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

            for k in range(sample_num):
                if y_pred[k] == k%10:
                    #print( k ,k%10)
                    pred[k%10] += 1

            test_acc = test_scores.accuracy
            test_loss = test_scores.loss
            printers['loss_G']('G_c{}'.format(i*2), epoch, test_loss)
            gimage = gimage[1,:,:,:]
            if i == 0 :
                printers['output0']('output_0', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 1 :
                printers['output1']('output_1', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 2 :
                printers['output2']('output_2', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 3 :
                printers['output3']('output_3', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 4 :
                printers['output4']('output_4', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 5 :
                printers['output5']('output_5', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 6 :
                printers['output6']('output_6', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 7 :
                printers['output7']('output_7', torch.clamp(gimage/gimage.max(), min=0, max=1))
            elif i == 8 :
                printers['output8']('output_8', torch.clamp(gimage/gimage.max(), min=0, max=1))
            else :
                printers['output9']('output_9', torch.clamp(gimage/gimage.max(), min=0, max=1))

            for j in range(len_discrete_code):
                printers['predict']('class_{}'.format(j), epoch, pred[j]/sample_num)

        printers['accuracy']('test_G', epoch+i/sample_num, test_acc)



    # Save the model and plot
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.pkl')

    #########################################################################
