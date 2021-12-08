"""
Differences compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import torch
import torch.nn.functional as F
import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

from model import RGCN

def main(args):
    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    else:
        dataset = AMDataset()

    # Load hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('labels')

    # Calculate normalization weight for each edge,
    # 1. / d, d is the degree of the destination node
    for cetype in hg.canonical_etypes:
        _, v, eid = hg.edges(form='all', etype=cetype)
        _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        hg.edges[cetype].data['norm'] = norm.unsqueeze(1)

    # get target category id
    category_id = hg.ntypes.index(category)

    g = dgl.to_homogeneous(hg, edata=['norm'])
    num_nodes = g.num_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    # Since the nodes are featureless, learn node embeddings from scratch
    # This requires passing the node IDs to the model.
    feats = torch.arange(num_nodes)

    # create model
    model = RGCN(num_nodes,
                 args.n_hidden,
                 num_classes,
                 num_rels,
                 num_bases=args.n_bases)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        feats = feats.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        labels = labels.cuda()
        model.cuda()
        g = g.to('cuda:%d' % args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    model.train()
    for epoch in range(50):
        logits = model(g, feats, edge_type, edge_norm)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        print("Epoch {:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
            epoch, train_acc, loss.item()))
    print()

    model.eval()
    with torch.no_grad():
        logits = model(g, feats, edge_type, edge_norm)
    logits = logits[target_idx]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['aifb', 'mutag', 'bgs', 'am'],
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=5e-4,
                        help="l2 norm coef")

    args = parser.parse_args()
    main(args)
