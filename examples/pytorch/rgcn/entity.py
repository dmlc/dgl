"""
Differences compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import torch
import torch.nn.functional as F

from entity_utils import load_data
from model import RGCN

def main(args):
    _, g, num_rels, num_classes, labels, train_idx, test_idx, target_idx = load_data(
        args.dataset, get_norm=True)

    num_nodes = g.num_nodes()

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
        labels = labels.cuda()
        model.cuda()
        g = g.to('cuda:%d' % args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=args.l2norm)

    # training loop
    model.train()
    for epoch in range(50):
        logits = model(g, feats)
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
        logits = model(g, feats)
    logits = logits[target_idx]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification')
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
    print(args)
    main(args)
