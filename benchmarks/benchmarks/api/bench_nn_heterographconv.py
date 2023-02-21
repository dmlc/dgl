import time

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import HeteroGraphConv, SAGEConv

from .. import utils


@utils.benchmark("time")
@utils.parametrize("feat_dim", [4, 32, 256])
@utils.parametrize("num_relations", [5, 50, 200])
def track_time(feat_dim, num_relations):
    device = utils.get_bench_device()
    dd = {}
    nn_dict = {}
    candidate_edges = [
        dgl.data.CoraGraphDataset(verbose=False)[0].edges(),
        dgl.data.PubmedGraphDataset(verbose=False)[0].edges(),
        dgl.data.CiteseerGraphDataset(verbose=False)[0].edges(),
    ]
    for i in range(num_relations):
        dd[("n1", "e_{}".format(i), "n2")] = candidate_edges[
            i % len(candidate_edges)
        ]
        nn_dict["e_{}".format(i)] = SAGEConv(
            feat_dim, feat_dim, "mean", activation=F.relu
        )

    # dry run
    feat_dict = {}
    graph = dgl.heterograph(dd)
    for i in range(num_relations):
        etype = "e_{}".format(i)
        feat_dict[etype] = torch.randn(
            (graph[etype].num_nodes(), feat_dim), device=device
        )

    conv = HeteroGraphConv(nn_dict).to(device)

    # dry run
    for i in range(3):
        conv(graph, feat_dict)
    # timing
    with utils.Timer() as t:
        for i in range(50):
            conv(graph, feat_dict)

    return t.elapsed_secs / 50
