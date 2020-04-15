import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import ChebConv

class temporal_conv_layer(nn.Module):
    def __init__(self, c_in, c_out, dia = 1,act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.act = act
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, c_out, (2, 1), 1, dilation = dia, padding = (0,0))


    def forward(self, x):
        return torch.relu(self.conv(x))


class spatio_conv_layer_A(nn.Module):
    def __init__(self, c, Lk):
        super(spatio_conv_layer_A, self).__init__()
        self.g = Lk
        print('c :',c)
        self.gc = GraphConv(c, c, activation=F.relu)
        # self.gc = ChebConv(c, c, 3)

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x.transpose(0, 3)
        x = x.transpose(1, 3)
        output = self.gc(self.g, x)
        output = output.transpose(1, 3)
        output = output.transpose(0, 3)
        return torch.relu(output)

class fully_conv_layer(nn.Module):
    def __init__(self, c):
        super(fully_conv_layer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class output_layer(nn.Module):
    def __init__(self, c, T, n):
        super(output_layer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = fully_conv_layer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class STGCN_WAVE(nn.Module):
    def __init__(self, c, T, n, Lk, p):
        super(STGCN_WAVE, self).__init__()
        self.tlayer1 = temporal_conv_layer(c[0], c[1], dia = 1)
        self.ln1 = nn.LayerNorm([n, c[1]])
        self.tlayer2 = temporal_conv_layer(c[1], c[2], dia = 2)
        self.slayer1 = spatio_conv_layer_A(c[2], Lk)

        self.tlayer3 = temporal_conv_layer(c[2], c[3], dia = 4)
        self.ln2 = nn.LayerNorm([n, c[3]])
        self.tlayer4 = temporal_conv_layer(c[3], c[4], dia = 8)
        self.slayer2 = spatio_conv_layer_A(c[4], Lk)
        self.tlayer5 = temporal_conv_layer(c[4], c[5], dia = 16)
        # self.ln3 = nn.LayerNorm([n, c[6]])
        # self.tlayer6 = temporal_conv_layer(c[5], c[6])
        # self.slayer3 = spatio_conv_layer_A(ks, c[6], Lk)
        # self.tlayer7 = temporal_conv_layer(c[6], c[7])
        
        print('T :',T)
        self.output = output_layer(c[5], T - 31, n)

    def forward(self, x):
        x = self.tlayer1(x)
        x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  
        x = self.tlayer2(x)
        x = self.slayer1(x)
        x = self.tlayer3(x)
        x = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  
        x = self.tlayer4(x)
        x = self.slayer2(x)
        x = self.tlayer5(x)
        # x = self.ln3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  
        # x = self.tlayer6(x)
        # x = self.slayer3(x)
        # x = self.tlayer7(x)
        # print('final x shape:',x.shape)
        return self.output(x)
