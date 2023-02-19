import time

import dgl
import dgl.function as fn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

from .. import utils


@utils.benchmark("time")
@utils.parametrize("graph_name", ["pubmed", "ogbn-arxiv"])
@utils.parametrize("feat_dim", [4, 32, 256])
@utils.parametrize("aggr_type", ["mean", "gcn", "pool"])
def track_time(graph_name, feat_dim, aggr_type):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name).to(device)

    feat = torch.randn((graph.num_nodes(), feat_dim), device=device)
    model = SAGEConv(
        feat_dim, feat_dim, aggr_type, activation=F.relu, bias=False
    ).to(device)

    # dry run
    for i in range(3):
        model(graph, feat)
    # timing
    with utils.Timer() as t:
        for i in range(50):
            model(graph, feat)

    return t.elapsed_secs / 50
