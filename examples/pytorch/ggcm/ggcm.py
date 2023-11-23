import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import LinearNeuralNetwork, lazy_random_walk, inverse_graph_convolution


class GGCM(nn.Module):
    def __init__(self, graph, args):
        super(GGCM, self).__init__()
        self.linear_nn = LinearNeuralNetwork(nfeat=graph.ndata["feat"].shape[1],
                               nclass=F.one_hot(graph.ndata["label"]).shape[1],
                               bias=True).to(args.device)
        
        self.label = graph.ndata["label"]
        self.test_mask = graph.ndata["test_mask"]
        self.train_mask = graph.ndata["train_mask"]
        self.val_mask = graph.ndata["val_mask"]

    def forward(self, x):
        return self.linear_nn(x)
    
    def update_embedds(features, A_hat, avg_edge_num, args):
        beta = 1.0
        beta_neg = 1.0
        K = args.degree
        X = features.clone()
        temp_sum = torch.zeros_like(features)
        I_N = dglsp.identity((features.shape[0], features.shape[0]))

        for _ in range(K):
            # lazy graph convolution (LGC)
            lazy_A = lazy_random_walk(A_hat, beta, I_N).to(args.device)

            # inverse graph convlution (IGC), lazy version
            neg_A_hat = inverse_graph_convolution(
                avg_edge_num, features.shape[0], args.device).to(args.device)
            inv_lazy_A = lazy_random_walk(neg_A_hat, beta_neg, I_N).to(args.device)
            inv_features = dglsp.spmm(inv_lazy_A, features)
            features = dglsp.spmm(lazy_A, features)

            # add for multi-scale version
            temp_sum += (features + inv_features) / 2.0
            beta *= args.decline
            beta_neg *= args.decline_neg

        embedds = args.alpha * X + (1 - args.alpha) * (temp_sum / (K * 1.0))
        return embedds
