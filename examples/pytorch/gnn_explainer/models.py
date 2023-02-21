import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim * 2, out_dim, bias=True)

    def forward(self, graph, feat, eweight=None):
        with graph.local_scope():
            graph.ndata["h"] = feat

            if eweight is None:
                graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "h"))
            else:
                graph.edata["ew"] = eweight
                graph.update_all(fn.u_mul_e("h", "ew", "m"), fn.mean("m", "h"))

            h = self.layer(th.cat([graph.ndata["h"], feat], dim=-1))

            return h


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=40):
        super().__init__()
        self.in_layer = Layer(in_dim, hid_dim)
        self.hid_layer = Layer(hid_dim, hid_dim)
        self.out_layer = Layer(hid_dim, out_dim)

    def forward(self, graph, feat, eweight=None):
        h = self.in_layer(graph, feat.float(), eweight)
        h = F.relu(h)
        h = self.hid_layer(graph, h, eweight)
        h = F.relu(h)
        h = self.out_layer(graph, h, eweight)
        return h
