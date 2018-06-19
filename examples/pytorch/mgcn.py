"""Molecular GCN model proposed by Kearnes et al. (2016).
We use the description from "Neural Message Passing for Quantum Chemistry" Sec.2.
The model has an edge representation e_vw that is updated during message passing.
The message function is:
    - M(h_v, h_w, e_vw) = e_vw
The update function is:
    - U_v(h_v, m_v) = Affine(Affine(h_v) || m_v)
The edge update function is:
    - U_e(e_vw, h_v, h_w) = Affine(ReLU(W_e || e_vw) || Affine(h_v || h_w))
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F

import dgl

class NodeUpdateModule(nn.Module):
    def __init__(self, hv_dims):
        self.net1 = nn.Sequential(
                nn.Linear(hv_dims),
                nn.ReLU()
                )
        self.net2 = nn.Sequential(
                nn.Linear(hv_dims),
                nn.ReLU()
                )
    def forward(self, node, msgs):
        m = T.stack(msgs).mean(0)
        new_h = self.net2(T.cat(self.net1(node['hv']), m))
        return {'hv' : new_h}

class MessageModule(nn.Module):
    def __init__(self):
        pass
    def forward(self, src, dst, edge):
        return edge['he']

class EdgeUpdateModule(nn.Module):
    def __init__(self, he_dims):
        self.net1 = nn.Sequential(
                nn.Linear(he_dims),
                nn.ReLU()
                )
        self.net2 = nn.Sequential(
                nn.Linear(he_dims),
                nn.ReLU()
                )
        self.net3 = nn.Sequential(
                nn.Linear(he_dims),
                nn.ReLU()
                )
    def forward(self, src, dst, edge):
        new_he = self.net1(src['hv']) + self.net2(dst['hv']) + self.net3(edge['he'])
        return {'he' : new_he}

# TODO: we don't need this one anymore
class EdgeModule(nn.Module):
    def __init__(self, he_dims):
        # use a flag to trigger either message module or edge update module.
        self.is_msg = True
        self.msg_mod = MessageModule()
        self.upd_mod = EdgeUpdateModule()
    def forward(self, src, dst, edge):
        if self.is_msg:
            self.is_msg = not self.is_msg
            return self.msg_mod(src, dst, edge)
        else:
            self.is_msg = not self.is_msg
            return self.upd_mod(src, dst, edge)

def train(g):
    # TODO(minjie): finish the complete training algorithm.
    g = dgl.DGLGraph(g)
    g.register_message_func(MessageModule())
    g.register_edge_func(EdgeUpdateModule())
    g.register_update_func(NodeUpdateModule())
    # TODO(minjie): init hv and he
    num_iter = 10
    for i in range(num_iter):
        # The first call triggers message function and update all the nodes.
        g.update_all()
        # The second sendall updates all the edge features.
        # g.send_all()
