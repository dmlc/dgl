import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import ChebConv

class TemporalConvLayer(nn.Module):
    def __init__(self, c_in, c_out, dia = 1,act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.act = act
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, c_out, (2, 1), 1, dilation = dia, padding = (0,0))


    def forward(self, x):
        return torch.relu(self.conv(x))


class SpatioConvLayer(nn.Module):
    def __init__(self, c, Lk):
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        # print('c :',c)
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

class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)

class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = FullyConvLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class STGCN_WAVE(nn.Module):
    def __init__(self, c, T, n, Lk, p, num_layers):
        super(STGCN_WAVE, self).__init__()
        self.num_layers = num_layers
        print('T :',T)
        self.layers = []
        cnt = 0
        diapower = 0
        for i in range(num_layers):
            group = i // 4 + 1
            id = (i + 1) % 4
            start = group * 2 - 2
            if (i + 1) % 4 == 0:
                self.layers.append(SpatioConvLayer(c[group * 2], Lk))
            
            if ((i + 1) % 4 == 1) or ((i + 1) % 4 == 3):
                if i == 0:
                    print('x :',start + id // 2,'y :',start + id // 2 + 1)
                self.layers.append(TemporalConvLayer(c[start + id // 2], c[start + id // 2 + 1], dia = 2**diapower))
                diapower += 1
                cnt += 1
            if (i + 1) % 4 == 2:
                self.layers.append(nn.LayerNorm([n,c[2*group - 1]]))
        # print('diapower :',diapower)
        # print('cnt :',cnt, "2**(diapower) :",2**(diapower))
        self.output = OutputLayer(c[cnt], T + 1 - 2**(diapower), n)
        for layer in self.layers:
            layer = layer.cuda()
    def forward(self, x):
        for i in range(self.num_layers):
            # print('i :', i)
            if (i + 1) % 4 == 2:
                x = self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  
            else:
                x = self.layers[i](x)
        return self.output(x)
