import torch
import torch.nn as nn
import dgl.function as fn

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None, self_loop=None, dropout=0.0):
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
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.self_loop:
            node_repr = node_repr + loop_message

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
    def __init__(self, in_feat, out_feat, num_gs, num_rels, num_bases=-1,featureless=False, bias=None, activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias, activation, self_loop=self_loop, dropout=dropout)
        self.num_rels = min(num_rels, num_gs)
        self.num_bases = num_bases
        self.out_feat = out_feat
        assert self.num_bases > 0

        # assuming in_feat and out_feat are divisible by num_bases
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))


    def propagate(self, parent, children):
        for idx, g in enumerate(children):
            g.copy_from(parent)
            """
            XXX: pytorch matmul broadcast duplicates weight tensor..., so this impl
                 wastes more than 2GB memory
            def node_update(node, node):
                node = node.view(-1, self.num_bases, 1, self.submat_in)
                weight = self.weight[idx].view(self.num_bases, self.submat_in, self.submat_out)
                return torch.matmul(node, weight).view(-1, self.out_feat)
            """

            # (lingfan): following hack saves memory
            def node_update(node):
                # num_bases x num_nodes x submat_in
                node = node.view(-1, self.num_bases, self.submat_in).transpose(0, 1)
                # num_bases x submat_in x submat_out
                weight = self.weight[idx].view(self.num_bases, self.submat_in, self.submat_out)
                out = []
                for i in range(self.num_bases):
                    out.append(torch.mm(node[i], weight[i]))
                out = torch.stack(out) # num_bases x num_nodes x submat_out
                return out.transpose(0, 1).contiguous().view(-1, self.out_feat)

            g.update_all(fn.src_mul_edge(), fn.sum(), node_update, batchable=True)
        # end for

        # merge node repr
        parent.merge(children, node_reduce_func='sum', edge_reduce_func=None)

        for g in children:
            g.pop_n_repr()
