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
import random
from dgl.contrib.data import load_data

from layers import RGCNBlockLayer as RGCNLayer
from model import BaseRGCN

import utils

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = self.embedding(node_id)

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=act, self_loop=True, dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g):
        return self.rgcn.forward(g)

    def evaluate(self, g):
        # get embedding and relation weight without grad
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets, labels):
        embedding = self.forward(g)
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss


def main(args):
    # load graph data
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization)

    # build test graph
    valid_data = torch.LongTensor(valid_data)

    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_data = torch.LongTensor(test_data)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel).view(-1, 1)
    test_norm = torch.from_numpy(test_norm).view(-1, 1)

    test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
    test_graph.edata['type'] = test_rel

    if use_cuda:
        model.cuda()
        """
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()
        test_node_id = test_node_id.cuda()
        test_rel = test_rel.cuda()
        test_deg = test_deg.cuda()
        """

    # optimizer
    # Can't use adam due to pytorch memory issue
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("start training...")

    # build adj list and calculate degrees
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(train_data):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]

    best_mrr = 0
    model_state_file = 'model_state.pth'

    epoch_time = []
    epoch = 0
    while True:
        epoch += 1
        print("Epoch {:03d}".format(epoch))

        model.train()
        if use_cuda:
            torch.cuda.empty_cache()

        # generate graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample)
        print("Done edge sampling")
        node_id = torch.from_numpy(node_id).view(-1, 1)
        edge_type = torch.from_numpy(edge_type).view(-1, 1)
        node_norm = torch.from_numpy(node_norm).view(-1, 1)
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, node_norm = edge_type.cuda(), node_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
        g.ndata.update({'id': node_id, 'norm': node_norm})
        g.edata['type'] = edge_type

        t0 = time.time()
        loss = model.get_loss(g, data, labels)
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        t1 = time.time()

        epoch_time.append(t1 - t0)
        print("Train Epoch Time(s) {:.4f} | Loss {:.4f}".format(
                epoch_time[-1], loss.item()))

        optimizer.zero_grad()
        for p in model.parameters():
            p.grad = None
        if use_cuda:
            torch.cuda.empty_cache()

        if epoch % args.evaluate_every == 0:
            # save parameter
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       'model_state_latest.pth')

            # create new model
            test_model = LinkPredict(num_nodes,
                                args.n_hidden,
                                num_rels,
                                num_bases=args.n_bases,
                                num_hidden_layers=args.n_layers,
                                dropout=args.dropout,
                                use_cuda=False,
                                reg_param=args.regularization)
            checkpoint = torch.load('model_state_latest.pth')
            test_model.load_state_dict(checkpoint['state_dict'])
            test_model.eval()
            print("start eval")
            mrr = utils.evaluate(test_graph, test_model, valid_data, num_nodes,
                                 hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            # save best model
            if mrr < best_mrr:
                if epoch > args.n_epochs:
                    # break
                    pass # do nothing
            else:
                best_mrr = mrr
                torch.save({'state_dict': test_model.state_dict(), 'epoch': epoch},
                           model_state_file)
        """

    print("training done")
    print("Mean iteration forward time: {:4f}".format(
            np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean iteration backward time: {:4f}".format(
            np.mean(backward_time[len(backward_time) // 4:])))

    forward_time = np.sum(np.reshape(
            forward_time, (args.n_epochs, -1)), axis=1)
    backward_time = np.sum(np.reshape(
            backward_time, (args.n_epochs, -1)), axis=1)

    print("Mean epoch forward time: {:4f}".format(
            np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean epoch backward time: {:4f}".format(
            np.mean(backward_time[len(backward_time) // 4:])))

    print("\nstart testing:")
    if use_cuda:
        torch.cuda.empty_cache()
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Using best epoch: {}".format(checkpoint['epoch']))
    utils.evaluate(test_graph, model, test_data, num_nodes, hits=[1, 3, 10],
                   eval_bz=args.eval_batch_size)
        """


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
    parser.add_argument("-e", "--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evalution every n epochs")

    args = parser.parse_args()
    print(args)
    main(args)

