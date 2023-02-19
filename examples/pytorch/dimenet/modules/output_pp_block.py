import dgl
import dgl.function as fn
import torch.nn as nn
from modules.initializers import GlorotOrthogonal


class OutputPPBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        out_emb_size,
        num_radial,
        num_dense,
        num_targets,
        activation=None,
        output_init=nn.init.zeros_,
        extensive=True,
    ):
        super(OutputPPBlock, self).__init__()

        self.activation = activation
        self.output_init = output_init
        self.extensive = extensive
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.up_projection = nn.Linear(emb_size, out_emb_size, bias=False)
        self.dense_layers = nn.ModuleList(
            [nn.Linear(out_emb_size, out_emb_size) for _ in range(num_dense)]
        )
        self.dense_final = nn.Linear(out_emb_size, num_targets, bias=False)
        self.reset_params()

    def reset_params(self):
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.up_projection.weight)
        for layer in self.dense_layers:
            GlorotOrthogonal(layer.weight)
        self.output_init(self.dense_final.weight)

    def forward(self, g):
        with g.local_scope():
            g.edata["tmp"] = g.edata["m"] * self.dense_rbf(g.edata["rbf"])
            g_reverse = dgl.reverse(g, copy_edata=True)
            g_reverse.update_all(fn.copy_e("tmp", "x"), fn.sum("x", "t"))
            g.ndata["t"] = self.up_projection(g_reverse.ndata["t"])

            for layer in self.dense_layers:
                g.ndata["t"] = layer(g.ndata["t"])
                if self.activation is not None:
                    g.ndata["t"] = self.activation(g.ndata["t"])
            g.ndata["t"] = self.dense_final(g.ndata["t"])
            return dgl.readout_nodes(
                g, "t", op="sum" if self.extensive else "mean"
            )
