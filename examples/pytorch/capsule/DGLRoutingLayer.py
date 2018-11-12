import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl


class DGLRoutingLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, f_size, u_hat, has_batch_dim=False, device='cpu'):
        super(DGLRoutingLayer, self).__init__()
        self.has_batch_dim = has_batch_dim
        self.g = init_graph(in_nodes, out_nodes, f_size, u_hat, device=device, has_batch_dim=True)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_indx = list(range(in_nodes))
        self.out_indx = list(range(in_nodes, in_nodes + out_nodes))
        self.device = device

    def forward(self, routing_num=1):
        for r in range(routing_num):
            # step 1 (line 4): normalize over out edges
            in_edges = self.g.edata['b'].view(self.in_nodes, self.out_nodes)
            self.g.edata['c'] = F.softmax(in_edges, dim=1).view(-1, 1)

            def cap_message(edges):
                if self.has_batch_dim:
                    return {'m': edges.data['c'].unsqueeze(1) * edges.data['u_hat']}
                else:
                    return {'m': edges.data['c'] * edges.data['u_hat']}
            self.g.register_message_func(cap_message)

            # step 2 (line 5)
            def cap_reduce(nodes):
                return {'s': th.sum(nodes.mailbox['m'], dim=1)}
            self.g.register_reduce_func(cap_reduce)

            # Execute step 1 & 2
            self.g.update_all()

            # step 3 (line 6)
            if self.has_batch_dim:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=2)
            else:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=1)

            # step 4 (line 7)
            v = th.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
            if self.has_batch_dim:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).mean(dim=1).sum(dim=1, keepdim=True)
            else:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)

    def end(self):
        del self.g
        # del self.g.edata['u_hat']
        # del self.g.ndata['v']
        # del self.g.ndata['s']
        # del self.g.edata['b']


def squash(s, dim=1):
    sq = th.sum(s ** 2, dim=dim, keepdim=True)
    s_norm = th.sqrt(sq)
    s = (sq / (1.0 + sq)) * (s / s_norm)
    return s


def init_graph(in_nodes, out_nodes, f_size, u_hat, device='cpu', has_batch_dim=False):
    g = dgl.DGLGraph()
    all_nodes = in_nodes + out_nodes
    g.add_nodes(all_nodes)
    in_indx = list(range(in_nodes))
    out_indx = list(range(in_nodes, in_nodes + out_nodes))
    # add edges use edge broadcasting
    for u in in_indx:
        g.add_edges(u, out_indx)

    # init states
    if has_batch_dim:
        batch_size = u_hat.size(1)
        g.ndata['v'] = th.zeros(all_nodes, batch_size, f_size).to(device)
    else:
        g.ndata['v'] = th.zeros(all_nodes, f_size).to(device)
    g.edata['u_hat'] = u_hat.to(device)
    g.edata['b'] = th.zeros(in_nodes * out_nodes, 1).to(device)
    return g
