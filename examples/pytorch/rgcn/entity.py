"""
Differences compared to tkipf/relation-gcn
* weight decay applied to all weights
"""
import argparse
import torch as th
import torch.nn.functional as F

from torchmetrics.functional import accuracy

from entity_utils import load_data
from model import RGCN

def main(args):
    g, num_rels, num_classes, labels, train_idx, test_idx, target_idx = load_data(
        args.dataset, get_norm=True)

    model = RGCN(g.num_nodes(),
                 args.n_hidden,
                 num_classes,
                 num_rels,
                 num_bases=args.n_bases)

    if args.gpu >= 0 and th.cuda.is_available():
        device = th.device(args.gpu)
    else:
        device = th.device('cpu')
    labels = labels.to(device)
    model = model.to(device)
    g = g.int().to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=args.wd)

    model.train()
    for epoch in range(100):
        logits = model(g)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx].argmax(dim=1), labels[train_idx]).item()
        print("Epoch {:05d} | Train Accuracy: {:.4f} | Train Loss: {:.4f}".format(
            epoch, train_acc, loss.item()))
    print()

    model.eval()
    with th.no_grad():
        logits = model(g)
    logits = logits[target_idx]
    test_acc = accuracy(logits[test_idx].argmax(dim=1), labels[test_idx]).item()
    print("Test Accuracy: {:.4f}".format(test_acc))

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
    parser.add_argument("--wd", type=float, default=5e-4,
                        help="weight decay")

    args = parser.parse_args()
    print(args)
    main(args)
