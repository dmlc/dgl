"""Infering Relational Data with Graph Convolutional Networks
"""
import argparse
import torch as th
from functools import partial
import torch.nn.functional as F

from dgl.data.rdf import AIFB, MUTAG, BGS, AM
from entity_classify import EntityClassify

def main(args):
    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFB()
    elif args.dataset == 'mutag':
        dataset = MUTAG()
    elif args.dataset == 'bgs':
        dataset = BGS()
    elif args.dataset == 'am':
        dataset = AM()
    else:
        raise ValueError()

    g = dataset.graph
    category = dataset.predict_category
    num_classes = dataset.num_classes
    test_idx = dataset.test_idx
    labels = dataset.labels
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        labels = labels.cuda()
        test_idx = test_idx.cuda()

    # create model
    model = EntityClassify(g,
                           args.n_hidden,
                           num_classes,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           use_self_loop=args.use_self_loop)
    # training loop
    model.load_state_dict(th.load(args.model_path))
    if use_cuda:
        model.cuda()

    print("start testing...")
    model.eval()
    logits = model.forward()[category_id]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str,
            help='path of the model to load from')
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")

    args = parser.parse_args()
    print(args)
    main(args)