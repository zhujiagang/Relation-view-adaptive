# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init


class unit_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 use_local_bn=False,
                 kernel_size=1,
                 stride=1,
                 mask_learning=False):
        super(unit_gcn, self).__init__()

        # ==========================================
        # number of nodes
        self.V = A.size()[-1]

        # the adjacency matrixes of the graph
        # self.A = Variable(
        #     A.clone(), requires_grad=False).view(-1, self.V, self.V)

        # number of input channels
        self.in_channels = in_channels

        # number of output channels
        self.out_channels = out_channels

        # if true, use mask matrix to reweight the adjacency matrix
        self.mask_learning = mask_learning

        # number of adjacency matrix (number of partitions)
        # self.num_A = self.A.size()[0]

        # if true, each node have specific parameters of batch normalizaion layer.
        # if false, all nodes share parameters.
        self.use_local_bn = use_local_bn
        # ==========================================

        # self.conv_list = nn.ModuleList([
        #     nn.Conv2d(
        #         self.in_channels,
        #         self.out_channels,
        #         kernel_size=(kernel_size, 1),
        #         padding=(int((kernel_size - 1) / 2), 0),
        #         stride=(stride, 1)) for i in range(self.num_A)
        # ])

        kernel_size = 2
        self.conv_dilation = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size= (1, kernel_size),
                dilation = (1, i+1),
                padding=(0, int((kernel_size - 1) / 2)),
                stride=stride) for i in range(self.V - 1)
        ])

        # if mask_learning:
        #     self.mask = nn.Parameter(torch.ones(self.A.size()))
        if use_local_bn:
            self.bn = nn.BatchNorm1d(self.out_channels * self.V)
        else:
            self.bn = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU()

        # initialize
        # for conv in self.conv_list:
        #     conv_init(conv)

        for conv in self.conv_dilation:
            conv_init(conv)

    def forward(self, x):
        N, C, T, V = x.size()

        for i in range(V - 1):
            x = torch.cat((x, x[:, :, :, i].contiguous().view(N, C, T, -1)), 3) ### 6977M 2:02
            if i == 0:
                xx = self.conv_dilation[i](x) ### 2400 64 24
            else:
                xx = xx + self.conv_dilation[i](x)

        xx = self.bn(xx)
        # nonliner
        xx = self.relu(xx)
        return xx