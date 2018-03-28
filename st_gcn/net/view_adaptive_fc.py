# The based unit of graph convolutional networks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net import conv_init
import random, math
import numpy as np

def conv_init_zeros(module):
    module.weight.data.zero_()
    module.bias.data.zero_()

class view_adaptive_fc(nn.Module):
    def __init__(self):
        super(view_adaptive_fc, self).__init__()

        self.trans_fc = nn.Sequential(
            nn.Conv2d(
                3*25,
                3*25,
                kernel_size= (9, 1),
                padding=(4, 0),
                stride=1),
            nn.BatchNorm2d(3 * 25),
            nn.ReLU(),
            nn.Conv2d(
                    3*25,
                    3,
                    kernel_size= (1, 1),
                    padding=(0, 0),
                    stride=1)
        )
        self.rotate_fc = nn.Sequential(
            nn.Conv2d(
                3*25,
                3*25,
                kernel_size= (9, 1),
                padding=(4, 0),
                stride=1),
            nn.BatchNorm2d(3 * 25),
            nn.ReLU(),
            nn.Conv2d(
                    3*25,
                    3,
                    kernel_size= (1, 1),
                    padding=(0, 0),
                    stride=1)
        )
        cnt = 0
        for conv in self.trans_fc:
            if isinstance(conv, nn.Conv2d):
                cnt += 1
                if cnt == 1:
                    conv_init(conv)
                else:
                    conv_init_zeros(conv)
        cnt = 0
        for conv in self.rotate_fc:
            if isinstance(conv, nn.Conv2d):
                cnt += 1
                if cnt == 1:
                    conv_init(conv)
                else:
                    conv_init_zeros(conv)


    def forward(self, x):
        N, C, T, V, M = x.size()
        xx = x[:,:,0].unsqueeze(2).mean(dim=3).unsqueeze(3).repeat(1, 1, T, V, 1)
        x = x - xx ###back to origin

        x = x.permute(0, 1, 3, 2, 4).contiguous()
        xx = x.view(N, C * V, T, M)

        trans = self.trans_fc(xx).permute(0, 2, 3, 1).contiguous().view(-1, 3).unsqueeze(1).repeat(1, V, 1)
        rotate = self.rotate_fc(xx).permute(0, 2, 3, 1).contiguous() #xx[:, :3]# * math.pi/180.0

        rotate.clamp(max=math.pi, min=-math.pi)

        cos_ = torch.cos(rotate)
        sin_ = torch.sin(rotate)

        cos_x = cos_[:, :, :, 0].unsqueeze(3)
        cos_y = cos_[:, :, :, 1].unsqueeze(3)
        cos_z = cos_[:, :, :, 2].unsqueeze(3)

        sin_x = sin_[:, :, :, 0].unsqueeze(3)
        sin_y = sin_[:, :, :, 1].unsqueeze(3)
        sin_z = sin_[:, :, :, 2].unsqueeze(3)

        zer = Variable(torch.zeros(cos_x.size()).cuda(), requires_grad=False)
        one_s = Variable(torch.ones(cos_x.size()).cuda(), requires_grad=False)

        Rx = [one_s, zer, zer, zer, cos_x, sin_x, zer, -sin_x, cos_x]
        Rx = torch.cat(Rx, 3).view(N*T*2, 3, 3)#.permute(0, 2, 3, 1).contiguous()

        Ry = [cos_y, zer, -sin_y, zer, one_s, zer, sin_y, zer, cos_y]
        Ry = torch.cat(Ry, 3).view(N*T*2, 3, 3)

        Rz = [cos_z, sin_z, zer, -sin_z, cos_z, zer, zer, zer, one_s]
        Rz = torch.cat(Rz, 3).view(N*T*2, 3, 3)

        temp = torch.matmul(Ry, Rx)
        temp = torch.matmul(Rz, temp)

        temp_x = x.permute(0, 3, 4, 2, 1).contiguous()
        temp_x = temp_x.view(-1, V, C)
        temp_x = temp_x - trans

        transform_x = torch.matmul(temp_x, temp)
        transform_x = transform_x.view(N, T, M, V, C).permute(0, 4, 1, 3, 2) ###N, C, T, V, M

        return transform_x


def rand_view_transform(self, X, angle1=-10, angle2=10, s1=0.9, s2=1.1):
    # skeleton data X, tensor3
    # genearte rand matrix
    random.random()
    agx = random.randint(angle1, angle2)
    agy = random.randint(angle1, angle2)
    s = random.uniform(s1, s2)
    agx = math.radians(agx)
    agy = math.radians(agy)
    Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
    Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
    Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
    # X0 = np.reshape(X,(-1,3))*Ry*Rx*Ss
    X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
    X = np.reshape(X0, X.shape)
    X = X.astype(np.float32)
    return X