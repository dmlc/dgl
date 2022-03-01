"""
Differences compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR.
"""

import argparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.data.knowledge_graph import FB15k237Dataset
from dgl.dataloading import GraphDataLoader

from link_utils import preprocess, SubgraphIterator, calc_mrr
from model import RGCN

class LinkPredict(nn.Module):
    def __init__(self, in_dim, num_rels, h_dim=500, num_bases=100, dropout=0.2, reg_param=0.01):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, regularizer="bdd",
                         num_bases=num_bases, dropout=dropout, self_loop=True)
        self.dropout = nn.Dropout(dropout)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(th.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = th.sum(s * r * o, dim=1)
        return score

    def forward(self, g, nids):
        return self.dropout(self.rgcn(g, nids=nids))

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def main(args):
    data = FB15k237Dataset(reverse=False)
    graph = data[0]
    num_nodes = graph.num_nodes()
    num_rels = data.num_rels

    train_g, test_g = preprocess(graph, num_rels)
    test_nids = th.arange(0, num_nodes)
    test_mask = graph.edata['test_mask']
    subg_iter = SubgraphIterator(train_g, num_rels, args.edge_sampler)
    dataloader = GraphDataLoader(subg_iter, batch_size=1, collate_fn=lambda x: x[0])

    # Prepare data for metric computation
    src, dst = graph.edges()
    triplets = th.stack([src, graph.edata['etype'], dst], dim=1)

    model = LinkPredict(num_nodes, num_rels)
    optimizer = th.optim.Adam(model.parameters(), lr=1e-2)

    if args.gpu >= 0 and th.cuda.is_available():
        device = th.device(args.gpu)
    else:
        device = th.device('cpu')
    model = model.to(device)

    best_mrr = 0
    model_state_file = 'model_state.pth'
    for epoch, batch_data in enumerate(dataloader):
        model.train()

        g, train_nids, edges, labels = batch_data
        g = g.to(device)
        train_nids = train_nids.to(device)
        edges = edges.to(device)
        labels = labels.to(device)

        embed = model(g, train_nids)
        loss = model.get_loss(embed, edges, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients
        optimizer.step()

        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f}".format(epoch, loss.item(), best_mrr))

        if (epoch + 1) % 500 == 0:
            # perform validation on CPU because full graph is too large
            model = model.cpu()
            model.eval()
            print("start eval")
            embed = model(test_g, test_nids)
            mrr = calc_mrr(embed, model.w_relation, test_mask, triplets,
                           batch_size=500, eval_p=args.eval_protocol)
            # save best model
            if best_mrr < mrr:
                best_mrr = mrr
                th.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

            model = model.to(device)

    print("Start testing:")
    # use best model checkpoint
    checkpoint = th.load(model_state_file)
    model = model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_g, test_nids)
    calc_mrr(embed, model.w_relation, test_mask, triplets,
             batch_size=500, eval_p=args.eval_protocol)

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
