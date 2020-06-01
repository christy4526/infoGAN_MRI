from __future__ import print_function, division, absolute_import
import os

from visdom import Visdom
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Lambda
from torch import optim
# project packages
from summary import Scalar, Image3D
import itertools
import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt

if __name__ == '__main__':
    vis = Visdom(port=10002, env='result-plot')

    plot = Scalar(vis, 'c1', opts=dict(
        showlegend=True, title='c1', ytickmin=0, ytinkmax=1.0))

    plots = dict(
        c1 = Scalar(vis, 'c1', opts=dict(
            showlegend=True, title='c1')),
        c2 = Scalar(vis, 'c2', opts=dict(
            showlegend=True, title='c2')),
        c3 = Scalar(vis, 'c3', opts=dict(
            showlegend=True, title='c3')))
    #data = np.genfromtxt('sort-result0.txt')
    data = np.genfromtxt('sort-result0.txt', names=('n','c1', 'c2','c3','AD','NC'))
    print(data['n'],data['c1'],data['c2'],data['c3'],data['AD'],data['NC'])
    #exit()
    #plots['c1']('c1', data['c1'], data['n'], )
    for i in range(12500):
        #print('AD', data[i][1], data[i][4])
        #print('NC', data[i][1], data[i][5])

        print('AD', data['c1'][i], data['AD'][i])
        print('NC', data['c1'][i], data['NC'][i])

        # plots['c1']('AD', data[i][1], data[i][4])
        # plots['c1']('NC', data[i][1], data[i][5])
        # plots['c2']('c2', data[i][0], data[i][2])
        # plots['c3']('c3', data[i][0], data[i][3])

        # plots['c1']('AD', data['c1'][i], data['AD'][i])
        # plots['c1']('NC', data['c1'][i], data['NC'][i])
        # plots['c2']('c2', data['n'][i], data['c2'][i])
        # plots['c3']('c3', data['n'][i], data['c3'][i])
