import dgl.sparse as dglsp

import torch
import torch.nn as nn

from utils import (
    inverse_graph_convolution,
    lazy_random_walk,
    symmetric_normalize_adjacency,
)


class GGCM(nn.Module):
    def __init__(self):
        super(GGCM, self).__init__()

    def get_embedding(self, graph, args):
        # get the learned node embeddings
        beta = 1.0
        beta_neg = 1.0
        layer_num, alpha = args.layer_num, args.alpha
        device = args.device
        features = graph.ndata["feat"]
        orig_feats = features.clone()
        temp_sum = torch.zeros_like(features)

        node_num = features.shape[0]
        I_N = dglsp.identity((node_num, node_num))
        A_hat = symmetric_normalize_adjacency(graph)

        # the inverser random adj
        edge_num = int(args.negative_rate * graph.num_edges() / node_num)
        # need n*k odd, for networkx
        edge_num = ((edge_num + 1) // 2) * 2

        for _ in range(layer_num):
            # inverse graph convlution (IGC), lazy version
            neg_A_hat = inverse_graph_convolution(edge_num, node_num, I_N).to(
                device
            )
            inv_lazy_A = lazy_random_walk(neg_A_hat, beta_neg, I_N).to(device)
            inv_features = dglsp.spmm(inv_lazy_A, features)

            # lazy graph convolution (LGC)
            lazy_A = lazy_random_walk(A_hat, beta, I_N).to(device)
            features = dglsp.spmm(lazy_A, features)

            # add for multi-scale version
            temp_sum += (features + inv_features) / 2.0
            beta *= args.decline
            beta_neg *= args.decline_neg
        embeds = alpha * orig_feats + (1 - alpha) * (
            temp_sum / (layer_num * 1.0)
        )
        return embeds
