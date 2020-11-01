#-*- coding:utf-8 -*-


# The training codes of the dummy model


import argparse
import dgl

import torch as th
import torch.nn as nn

from dgl.data import AIFBDataset
from models import dummy_gnn_model
from gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4, gen_syn5


def main(args):
    # check dataset
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'syn1':
        g, role_ids, name = gen_syn1()
    elif args.dataset == 'syn2':
        g, role_ids, name = gen_syn2()
    elif args.dataset == 'syn3':
        g, role_ids, name = gen_syn3()
    elif args.dataset == 'syn4':
        g, role_ids, name = gen_syn4()
    elif args.dataset == 'syn5':
        g, role_ids, name = gen_syn5()
    else:
        raise ValueError()

    graph = dgl.from_networkx(g)

    num_classes = max(role_ids) + 1
    feat_dim = 10
    print(num_classes)

    # in this experiment fix dimensions of input, hidden, and output layers
    dummy_model = dummy_gnn_model(feat_dim, 40, num_classes)

    # set up feature of nodes to all 1
    n_feats = th.randn(graph.number_of_nodes(), feat_dim)

    # define loss funciton and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optim = th.optim.Adam(dummy_model.parameters(), lr=0.001)

    # train and output
    for epoch in range(500):

        dummy_model.train()

        logits = dummy_model(graph, n_feats)
        loss = loss_fn(logits, th.tensor(role_ids, dtype=th.long))
        pred = th.sum(logits.argmax(dim=1) == th.tensor(role_ids, dtype=th.long)).item() / len(role_ids)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print('Acc: {:.4f}; Loss: {:.6f}'.format(pred, loss.item()))

    print(logits[3])

    # save model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dummy model training')
    parser.add_argument('--dataset', type=str, default='syn1', help='dataset used for training the model')

    args = parser.parse_args()
    print(args)

    main(args)