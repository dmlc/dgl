"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to tkipf/relation-gcn
* edge directions are reversed (kipf did not transpose adj before spmv)
* l2norm applied to all weights
* do not report filtered metrics
* mini-batch fashion evaluation
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from dgl.data import load_data
from functools import partial

from layers import RGCNBlockLayer as RGCNLayer
from model import BaseRGCN

from utils import Dataset, negative_sampling, evaluate


class RGCN(BaseRGCN):
    def create_features(self):
        # feature is embedding
        features = nn.Parameter(torch.Tensor(len(self.g), self.h_dim))
        nn.init.xavier_uniform_(features, gain=nn.init.calculate_gain('relu'))
        return features

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, len(self.subgraphs), self.num_rels, self.num_bases, activation=F.relu)


class LinkPredict(nn.Module):
    def __init__(self, g, h_dim, relations, num_bases=-1, num_hidden_layers=1, dropout=0, use_cuda=False):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(g, h_dim, h_dim, relations, num_bases, num_hidden_layers, dropout, use_cuda)
        self.w_relation = nn.Parameter(torch.Tensor(relations.shape[1] // 2, h_dim))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        return torch.sum(s * r * o, dim=1)

    def forward(self, triplets, training=True):
        embedding = self.rgcn.forward()
        return self.calc_score(embedding, triplets)

    def evaluate(self):
        # get embedding and relation weight without grad
        embedding = self.rgcn.forward()
        return embedding.detach(), self.w_relation.detach()


def main(args):
    # load graph data
    data = load_data(args)
    num_nodes = data.num_nodes
    edges = data.edges
    relations = data.relations
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    data = None

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create graph
    g = DGLGraph()
    g.add_nodes_from(np.arange(num_nodes))
    g.add_edges_from(edges)


    # create model
    model = LinkPredict(g,
                        args.n_hidden,
                        relations,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda)

    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    if use_cuda:
        model.cuda()
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    dataset = Dataset(train_data)
    collate_fn = partial(negative_sampling, num_entity=num_nodes, negative_rate=args.negative_sample)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    for epoch in range(args.n_epochs):
        print("Epoch {:03d}".format(epoch))
        for batch_idx, (data, labels) in enumerate(dataloader):
            if use_cuda:
                torch.cuda.empty_cache()
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)
            if use_cuda:
                data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            t0 = time.time()
            logits = model.forward(data, training=True)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            t1 = time.time()
            loss.backward()
            t2 = time.time()
            optimizer.step()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            print("Batch {:03d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".format(batch_idx, forward_time[-1], backward_time[-1]))

        evaluate(model, valid_data, num_nodes, hits=[1, 3, 10], eval_bz=args.eval_batch_size)

    print("test set:")
    evaluate(model, test_data, num_nodes, hits=[1, 3, 10], eval_bz=args.eval_batch_size)


    print("Mean iteration forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean iteration backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

    forward_time = np.sum(np.reshape(forward_time, (args.n_epochs, -1)), axis=1)
    backward_time = np.sum(np.reshape(backward_time, (args.n_epochs, -1)), axis=1)

    print("Mean epoch forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean epoch backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--batch-size", type=int, default=30000,
            help="number of possible samples to process in a batch")
    parser.add_argument("--eval-batch-size", type=int, default=128,
            help="batch size when evaluating")

    args = parser.parse_args()
    print(args)
    main(args)

