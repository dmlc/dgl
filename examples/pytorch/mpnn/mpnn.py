import torch
import torch.nn as nn
from torch.nn import Parameter
import dgl.function as fn

class NNConvLayer(nn.Module):
    def __init__(self,
                 g,
                 in_channels,
                 out_channels,
                 edge_net,
                 aggr='add',
                 root_weight=True,
                 bias=True):
        super(NNConvLayer, self).__init__()
        
        self.g = g
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_net = edge_net
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    
    def reset_parameters(self):
        if self.root is not None:
            nn.init.xavier_normal_(self.root.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.zero_()
        for m in edge_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.414)
                
    def message(self, edges):
        return {'m' : torch.matmul(edges.src['h'].unsqueeze(1),edges.data['w']).squeeze(1)}
    
    def reduce(self, nodes):
        if self.aggr == 'add':
            return {'aggr_out' : torch.sum(nodes.mailbox['m'], 1)}
        elif self.aggr == 'mean':
            return {'aggr_out' : torch.mean(nodes.mailbox['m'], 1)}
        else:
            raise AssertionError()
            
    def apply_node_func(self, nodes):
        aggr_out = nodes.data['aggr_out']        
        if self.root is not None:
            aggr_out = torch.mm(nodes.data['h'], self.root) + aggr_out
        
        if self.bias is not None:
            aggr_out = aggr_out + self.bias   
        
        return {'h': aggr_out}

    def forward(self, h, e):
        h = h.unsqueeze(-1) if h.dim() == 1 else h
        e = e.unsqueeze(-1) if e.dim() == 1 else e
        
        self.g.ndata['h'] = h
        self.g.edata['w'] = self.edge_net(e).view(-1, self.in_channels, self.out_channels)
        
        if self.aggr == 'add':
            g.update_all(self.message, self.reduce,self.apply_node_func)
        elif self.aggr == 'mean':
            
            g.update_all(self.message, self.reduce, self.apply_node_func)

        return self.g.ndata.pop('h')
