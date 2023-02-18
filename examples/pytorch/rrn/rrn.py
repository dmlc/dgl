"""
Recurrent Relational Network(RRN) module

References:
- Recurrent Relational Networks
- Paper: https://arxiv.org/abs/1711.08028
- Original Code: https://github.com/rasmusbergpalm/recurrent-relational-networks
"""

import dgl.function as fn
import torch
from torch import nn


class RRNLayer(nn.Module):
    def __init__(self, msg_layer, node_update_func, edge_drop):
        super(RRNLayer, self).__init__()
        self.msg_layer = msg_layer
        self.node_update_func = node_update_func
        self.edge_dropout = nn.Dropout(edge_drop)

    def forward(self, g):
        g.apply_edges(self.get_msg)
        g.edata["e"] = self.edge_dropout(g.edata["e"])
        g.update_all(
            message_func=fn.copy_e("e", "msg"), reduce_func=fn.sum("msg", "m")
        )
        g.apply_nodes(self.node_update)

    def get_msg(self, edges):
        e = torch.cat([edges.src["h"], edges.dst["h"]], -1)
        e = self.msg_layer(e)
        return {"e": e}

    def node_update(self, nodes):
        return self.node_update_func(nodes)


class RRN(nn.Module):
    def __init__(self, msg_layer, node_update_func, num_steps, edge_drop):
        super(RRN, self).__init__()
        self.num_steps = num_steps
        self.rrn_layer = RRNLayer(msg_layer, node_update_func, edge_drop)

    def forward(self, g, get_all_outputs=True):
        outputs = []
        for _ in range(self.num_steps):
            self.rrn_layer(g)
            if get_all_outputs:
                outputs.append(g.ndata["h"])
        if get_all_outputs:
            outputs = torch.stack(outputs, 0)  # num_steps x n_nodes x h_dim
        else:
            outputs = g.ndata["h"]  # n_nodes x h_dim
        return outputs
