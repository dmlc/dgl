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
    def __init__(self, in_feat, out_feat, num_gs, num_rels, num_bases=-1,featureless=False, bias=None, activation=None):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = min(num_rels, num_gs)
        self.num_bases = num_bases
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        self.featureless = featureless

        # add weights
        self.weights = nn.ParameterList()
        for _ in range(self.num_bases):
            self.weights.append(nn.Parameter(torch.Tensor(self.in_feat, self.out_feat)))
            nn.init.xavier_uniform_(self.weights[-1], gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, parent, children):
        if self.num_bases < self.num_rels:
            weights = torch.stack(list(self.weights)).permute(1, 0, 2)
            weights = torch.matmul(self.w_comp, weights).view(-1, self.out_feat)
            weights = torch.split(weights, self.in_feat, dim=0)
        else:
            weights = self.weights

        for idx, g in enumerate(children):
            if self.featureless:
                # hack to avoid materize node features to avoid memory issues
                g.set_n_repr(weights[idx][g.parent_nid])
                g.update_all(fn.src_mul_edge(), fn.sum(), None, batchable=True)
            else:
                # update subgraph node repr
                g.copy_from(parent)
                g.update_all(fn.src_mul_edge(), fn.sum(),
                             lambda node: torch.mm(node, weights[idx]),
                             batchable=True)
        # end for

        # merge node repr
        parent.merge(children, node_reduce_func='sum', edge_reduce_func=None)

        for g in children:
            g.pop_n_repr()


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_gs, num_rels, num_bases, bias=None, activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias, activation, self_loop=self_loop, dropout=dropout)
        self.num_rels = min(num_rels, num_gs)
        self.num_bases = num_bases
        assert self.num_bases > 0

        # use graph convolution to implement block decomposition model, assuming in_feat and out_feat are divisible by num_bases
        self.linears = nn.ModuleList()
        for _ in range(self.num_rels):
            l = nn.Conv1d(in_feat, out_feat, kernel_size=1, groups=self.num_bases, bias=False)
            nn.init.xavier_uniform_(l.weight, gain=nn.init.calculate_gain('relu'))
            self.linears.append(l)

    def propagate(self, parent, children):
        for idx, g in enumerate(children):
            g.copy_from(parent)
            g.update_all(fn.src_mul_edge(), fn.sum(), lambda x: self.linears[idx](x.unsqueeze(2)).squeeze(), batchable=True)

        # merge node repr
        parent.merge(children, node_reduce_func='sum', edge_reduce_func=None)

        for g in children:
            g.pop_n_repr()
