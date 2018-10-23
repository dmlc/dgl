import torch
import torch.nn as nn
import dgl.function as fn

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None, self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.get_n_repr()['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.get_n_repr()['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.set_n_repr({'h': node_repr})


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, activation=None):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from basis
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        def msg_func(src, edge):
            # FIXME: normalizer
            w = weight[edge['type']]
            msg = torch.bmm(src['h'].unsqueeze(1), w).squeeze()
            return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None, activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias, activation, self_loop=self_loop, dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        def msg_func(src, edge):
            lambda x: self.linears[idx](x.unsqueeze(2)).squeeze()
            weight = self.weight[edge['type']].view(-1, self.submat_in, self.submat_out)
            node = src['h'].view(-1, 1, self.submat_in)
            msgs = torch.bmm(node, weight).view(-1, self.out_feat) * edge['norm']
            # FIXME: normalizer
            return {'msg': msgs}

        # FIXME: featureless case?
        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)
