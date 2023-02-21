import dgl
import dgl.function as fn
import torch.nn as nn
from modules.initializers import GlorotOrthogonal


class OutputBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_radial,
        num_dense,
        num_targets,
        activation=None,
        output_init=nn.init.zeros_,
    ):
        super(OutputBlock, self).__init__()

        self.activation = activation
        self.output_init = output_init
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_layers = nn.ModuleList(
            [nn.Linear(emb_size, emb_size) for _ in range(num_dense)]
        )
        self.dense_final = nn.Linear(emb_size, num_targets, bias=False)
        self.reset_params()

    def reset_params(self):
        GlorotOrthogonal(self.dense_rbf.weight)
        for layer in self.dense_layers:
            GlorotOrthogonal(layer.weight)
        self.output_init(self.dense_final.weight)

    def forward(self, g):
        with g.local_scope():
            g.edata["tmp"] = g.edata["m"] * self.dense_rbf(g.edata["rbf"])
            g.update_all(fn.copy_e("tmp", "x"), fn.sum("x", "t"))

            for layer in self.dense_layers:
                g.ndata["t"] = layer(g.ndata["t"])
                if self.activation is not None:
                    g.ndata["t"] = self.activation(g.ndata["t"])
            g.ndata["t"] = self.dense_final(g.ndata["t"])
            return dgl.readout_nodes(g, "t")
