"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to tkipf/relation-gcn
* mini-batch fashion evaluation
* hack impl for weight basis multiply
* sample for each graph batch?
* report filtered metrics (todo)
* early stopping (by save model with best validation mrr)
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

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, activation=act, self_loop=True, dropout=self.dropout)


class LinkPredict(nn.Module):
    def __init__(self, g, h_dim, relations, num_bases=-1, num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(g, h_dim, h_dim, relations, num_bases, num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(relations.shape[1] // 2, h_dim))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        return torch.sum(s * r * o, dim=1)

    def forward(self):
        return self.rgcn.forward()

    def evaluate(self):
        # get embedding and relation weight without grad
        embedding = self.forward()
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, triplets, labels):
        embedding = self.forward()
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss


def main(args):
    # load graph data
    data = load_data(args)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

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
                        use_cuda=use_cuda,
                        reg_param=args.regularization)

    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    if use_cuda:
        model.cuda()
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    # build adj list and calculate degrees
    adj_list = [[] for _ in entities]
    for i,triplet in enumerate(train_data):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]

    best_mrr = 0
    model_state_file = 'model_state.pth'

    forward_time = []
    backward_time = []
    epoch = 0
    while True:
        epoch += 1
        if use_cuda:
            torch.cuda.empty_cache()
        model.train()
        print("Epoch {:03d}".format(epoch))

        # generate graph and data
        g, node_id, edge_type, data, labels = generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample)
        node_id, edge_type = torch.from_numpy(node_id), torch.from_numpy(edge_type)
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes)).tousertensor()
        if use_cuda:
            node_id, edge_type, deg = node_id.cuda(), edge_type.cuda(), deg.cuda()
            data, labels = data.cuda(), labels.cuda()
        g.set_n_repr({'id': node_id, 'deg': deg})
        g.set_e_repr({'type': edge_type})

        optimizer.zero_grad()
        t0 = time.time()
        loss = model.get_loss(g, data, labels)
        t1 = time.time()
        loss.backward()
        t2 = time.time()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Batch {:03d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".format(batch_idx, forward_time[-1], backward_time[-1]))

        if epoch % args.evaluate_every == 0:
            if use_cuda:
                torch.cuda.empty_cache()
            model.eval()
            mrr = evaluate(model, valid_data, num_nodes, hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            # save best model
            if mrr < best_mrr:
                if epoch > args.n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)

    print("training done")
    print("Mean iteration forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean iteration backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

    forward_time = np.sum(np.reshape(forward_time, (args.n_epochs, -1)), axis=1)
    backward_time = np.sum(np.reshape(backward_time, (args.n_epochs, -1)), axis=1)

    print("Mean epoch forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean epoch backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

    print("\nstart testing:")
    if use_cuda:
        torch.cuda.empty_cache()
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Using best epoch: {}".format(checkpoint['epoch']))
    evaluate(model, test_data, num_nodes, hits=[1, 3, 10], eval_bz=args.eval_batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=100,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, deafult=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=2000,
            help="perform evalution every n epochs")

    args = parser.parse_args()
    print(args)
    main(args)

