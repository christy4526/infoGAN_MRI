from __future__ import absolute_import, division, print_function

import torch.nn as nn

class ConvT3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, dilation=1,
                 batch_norm=True, activation=None):
        super(ConvT3d, self).__init__()
        self.convT = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                              stride, padding, output_padding, groups, not batch_norm, dilation)
        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else None

        self.activation = activation

    def forward(self, x):
        x = self.convT(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 batch_norm=True, activation=None):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=not batch_norm)
        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else None

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class SplitConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0,
                 weights_initializer=nn.init.xavier_normal,
                 biases_initializer=None,
                 batch_norm=True,
                 activation=nn.ReLU):
        super(SplitConv3d, self).__init__()

        if (out_channels-in_channels) % 3 != 0:
            raise ArithmeticError
        cremental = int((out_channels-in_channels)/3)

        self.conv = nn.Sequential(
            Conv3d(in_channels, in_channels+cremental,
                   (kernel_size, 1, 1), (stride, 1, 1), (padding, 0, 0),
                   weights_initializer=weights_initializer,
                   batch_norm=batch_norm,
                   activation=activation),
            Conv3d(in_channels+cremental, in_channels+2*cremental,
                   (1, kernel_size, 1), (1, stride, 1), (0, padding, 0),
                   weights_initializer=weights_initializer,
                   batch_norm=batch_norm,
                   activation=activation),
            Conv3d(in_channels+2*cremental, out_channels,
                   (1, 1, kernel_size), (1, 1, stride), (0, 0, padding),
                   weights_initializer=weights_initializer,
                   batch_norm=batch_norm,
                   activation=activation))

    def forward(self, x):
        return self.conv(x)


class ConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 weights_initializer=nn.init.xavier_normal,
                 biases_initializer=None,
                 batch_norm=True,
                 activation=nn.ReLU):
        super(ConvTranspose3d, self).__init__()
        self.convT = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride, padding, output_padding, groups,
                                        bias, dilation)
        self.batch_norm = nn.BatchNorm3d(out_channels) if batch_norm else None

        self.activation = activation
        if self.activation is not None:
            self.activation = activation(inplace=True)

        if weights_initializer is not None:
            weights_initializer(self.convT.weight)
        else:
            # n = np.prod(self.convT.kernel_size) * self.convT.out_channels
            # self.convT.weight.data.normal_(0, np.sqrt(2./n))
            if isinstance(self.activation, nn.SELU):
                fan_in = np.prod(self.convT.kernel_size[1:])
                self.convT.weight.data.normal_(0, np.sqrt(1./fan_in))
            else:
                pass

        if bias:
            if biases_initializer is not None:
                biases_initializer(self.convT.bias)
            else:
                self.convT.bias.data.zero_()

        if self.batch_norm is not None:
            self.batch_norm.weight.data.fill_(1)
            self.batch_norm.bias.data.zero_()

    def forward(self, x):
        x = self.convT(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DropoutLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 p=0.5, activation=nn.ReLU):
        super(DropoutLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(p=p)

        self.activation = activation
        if self.activation is not None:
            self.activation = activation(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Merge(nn.Module):
    def __init__(self, mode='concat'):
        super(Merge, self).__init__()
        assert mode in ('concat', 'add')

        self.mode = mode

        if mode == 'concat':
            self.fn = lambda x: torch.cat(x, dim=1)
        elif mode == 'add':
            self.fn = lambda x: x[0] if len(x) == 1 else x[0]+self.fn(x[1:])
        else:
            raise ValueError

    def forward(self, x):
        return self.fn(x)


class Split(nn.Module):
    def __init__(self, indices):
        super(Split, self).__init__()
        self.indices = indices

    def forward(self, x):
        splited = []
        remain = x
        for idx in self.indices:
            splited += [remain[:, :idx, ...]]
            remain = remain[:, idx:, ...]
        splited += [remain]
        return splited
