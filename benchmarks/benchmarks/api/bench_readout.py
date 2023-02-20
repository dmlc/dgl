import time

import dgl

import torch

from .. import utils


@utils.benchmark("time")
@utils.parametrize("batch_size", [4, 256, 1024])
@utils.parametrize("feat_size", [16, 128, 512])
@utils.parametrize("readout_op", ["sum", "max", "min", "mean"])
@utils.parametrize("type", ["edge", "node"])
def track_time(batch_size, feat_size, readout_op, type):
    device = utils.get_bench_device()
    ds = dgl.data.QM7bDataset()
    # prepare graph
    graphs = ds[0:batch_size][0]

    g = dgl.batch(graphs).to(device)
    if type == "node":
        g.ndata["h"] = torch.randn((g.num_nodes(), feat_size), device=device)
        for i in range(10):
            out = dgl.readout_nodes(g, "h", op=readout_op)
        with utils.Timer() as t:
            for i in range(50):
                out = dgl.readout_nodes(g, "h", op=readout_op)
    elif type == "edge":
        g.edata["h"] = torch.randn((g.num_edges(), feat_size), device=device)
        for i in range(10):
            out = dgl.readout_edges(g, "h", op=readout_op)
        with utils.Timer() as t:
            for i in range(50):
                out = dgl.readout_edges(g, "h", op=readout_op)
    else:
        raise Exception("Unknown type")

    return t.elapsed_secs / 50
