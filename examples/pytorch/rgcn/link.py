"""
Differences compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.data.knowledge_graph import load_data

from link_utils import build_test_graph, get_adj_and_degrees, sample_subgraph, calc_mrr
from model import RGCN

class LinkPredict(nn.Module):
    def __init__(self, in_dim, num_rels, h_dim=500, num_bases=100, dropout=0.2, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, regularizer="bdd",
                         num_bases=num_bases, dropout=dropout, self_loop=True, link_pred=True)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def main(args):
    data = load_data('FB15k-237')
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    # create model
    model = LinkPredict(num_nodes,
                        num_rels,
                        num_bases=100,
                        reg_param=0.01)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = build_test_graph(
        num_nodes, num_rels, train_data)
    test_node_id = torch.arange(0, num_nodes).view(-1, 1)
    test_graph.edata[dgl.ETYPE] = torch.from_numpy(test_rel)
    test_graph.edata['norm'] = node_norm_to_edge(test_graph, test_norm)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model_state_file = 'model_state.pth'

    best_mrr = 0
    for epoch in range(6000):
        model.train()

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = sample_subgraph(
            train_data, num_rels, adj_list, degrees, args.edge_sampler,
            sample_size=30000, split_size=0.5, negative_rate=10)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        g.edata[dgl.ETYPE] = torch.from_numpy(edge_type)
        g.edata['norm'] = node_norm_to_edge(g, node_norm)
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.num_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)

        embed = model(g, node_id)
        loss = model.get_loss(embed, data, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients
        optimizer.step()

        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f}".format(epoch, loss.item(), best_mrr))

        # validation
        if epoch % 500 == 0:
            # perform validation on CPU because full graph is too large
            model = model.cpu()
            model.eval()
            print("start eval")
            embed = model(test_graph, test_node_id)
            mrr = calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                           valid_data, test_data, eval_bz=500, eval_p=args.eval_protocol)
            # save best model
            if best_mrr < mrr:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

            if use_cuda:
                model = model.cuda()

    print("Start testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    model = model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_graph, test_node_id)
    calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
             test_data, hits=[1, 3, 10], batch_size=500, eval_p=args.eval_protocol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for link prediction')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--eval-protocol", type=str, default='filtered',
                        choices=['filtered', 'raw'],
                        help="Whether to use 'filtered' or 'raw' MRR for evaluation")
    parser.add_argument("--edge-sampler", type=str, default='uniform',
                        choices=['uniform', 'neighbor'],
                        help="Type of edge sampler: 'uniform' or 'neighbor'"
                             "The original implementation uses neighbor sampler.")

    args = parser.parse_args()
    print(args)
    main(args)
