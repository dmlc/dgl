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

    # define how propagation for each children is done and merged back to parent
    def propagate(self, parent, children):
        raise NotImplementedError

    def forward(self, parent, children):
        if self.self_loop:
            loop_message = torch.mm(parent.get_n_repr(), self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(parent, children)

        # apply bias and activation
        node_repr = parent.get_n_repr()
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        parent.set_n_repr(node_repr)


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,featureless=False, bias=None, activation=None):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        self.featureless = featureless

        # add weights
        self.weight = nn.Parameter(torch.Tensor(self.in_feat, self.num_bases, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, parent, children):
        if self.num_bases < self.num_rels:
            # generate all weights from basis
            weights = torch.matmul(self.w_comp, self.weight).view(self.num_rels, self.in_feat, self.out_feat)
        else:
            weights = self.weights

        def msg_func(src, edge):
            # FIXME: normalizer
            return {'msg': torch.bmm(src['h'], weights[edge['type']])}

        # FIXME: featureless case?
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

    def propagate(self, parent, children):
        def msg_func(src, edge):
            lambda x: self.linears[idx](x.unsqueeze(2)).squeeze()
            weight = self.weight[edge['type']].view(-1, self.submat_in, self.submat_out)
            node = src['h'].view(-1, 1, self.submat_in)
            msgs = torch.bmm(node, weight).view(-1, self.out_feat)
            # FIXME: normalizer
            return {'msg': msgs}

        # FIXME: featureless case?
        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)
