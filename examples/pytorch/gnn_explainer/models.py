import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class dummy_layer(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(dummy_layer, self).__init__()
        self.layer = nn.Linear(in_dim * 2, out_dim, bias=True)

    def forward(self, graph, feat, eweight=None):
        graph.ndata['h'] = feat

        if eweight == None:
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        else:
            graph.edata['ew'] = eweight
            graph.update_all(fn.u_mul_e('h', 'ew', 'm'), fn.mean('m', 'h'))

        graph.ndata['h'] = self.layer(th.cat([graph.ndata['h'], feat], dim=-1))

        output = graph.ndata['h']
        return output


class dummy_gnn_model(nn.Module):

    """
    A dummy gnn model, which is same as graph sage, but could adopt edge mask in forward

    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim):
        super(dummy_gnn_model, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        
        self.in_layer = dummy_layer(self.in_dim, self.hid_dim)
        self.hid_layer = dummy_layer(self.hid_dim, self.hid_dim)
        self.out_layer = dummy_layer(self.hid_dim, self.out_dim)

    def forward(self, graph, feat, eweight=None):
        h = self.in_layer(graph, feat, eweight)
        h = F.relu(h)
        h = self.hid_layer(graph, h, eweight)
        h = F.relu(h)
        h = self.out_layer(graph, h, eweight)
        return h
        