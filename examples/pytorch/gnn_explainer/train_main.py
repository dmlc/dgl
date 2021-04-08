# The training codes of the dummy model


import os
import argparse
import dgl

import torch as th
import torch.nn as nn

from dgl import save_graphs
from models import dummy_gnn_model
from gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4, gen_syn5
import numpy as np

def main(args):
    # load dataset
    if args.dataset == 'syn1':
        g, labels, name = gen_syn1()
    elif args.dataset == 'syn2':
        g, labels, name = gen_syn2()
    elif args.dataset == 'syn3':
        g, labels, name = gen_syn3()
    elif args.dataset == 'syn4':
        g, labels, name = gen_syn4()
    elif args.dataset == 'syn5':
        g, labels, name = gen_syn5()
    else:
        raise NotImplementedError
    
    #Transform to dgl graph. 
    graph = dgl.from_networkx(g) 
    labels = th.tensor(labels, dtype=th.long)
    graph.ndata['label'] = labels
    graph.ndata['feat'] = th.randn(graph.number_of_nodes(), args.feat_dim)
    hid_dim = th.tensor(args.hidden_dim, dtype=th.long)
    label_dict = {'hid_dim':hid_dim}

    # save graph for later use
    save_graphs(filename='./'+args.dataset+'.bin', g_list=[graph], labels=label_dict)

    num_classes = max(graph.ndata['label']).item() + 1
    n_feats = graph.ndata['feat']

    #create model
    dummy_model = dummy_gnn_model(args.feat_dim, args.hidden_dim, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optim = th.optim.Adam(dummy_model.parameters(), lr=args.lr, weight_decay=args.wd)

    # train and output
    for epoch in range(args.epochs):

        dummy_model.train()

        logits = dummy_model(graph, n_feats)
        loss = loss_fn(logits, labels)
        acc = th.sum(logits.argmax(dim=1) == labels).item() / len(labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        print('In Epoch: {:03d}; Acc: {:.4f}; Loss: {:.6f}'.format(epoch, acc, loss.item()))

    # save model
    model_stat_dict = dummy_model.state_dict()
    model_path = os.path.join('./', 'dummy_model_{}.pth'.format(args.dataset))
    th.save(model_stat_dict, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dummy model training')
    parser.add_argument('--dataset', type=str, default='syn1', help='The dataset used for training the model.')
    parser.add_argument('--feat_dim', type=int, default=10, help='The feature dimension.')
    parser.add_argument('--hidden_dim', type=int, default=40, help='The hidden dimension.')
    parser.add_argument('--epochs', type=int, default=500, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate.')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay.')

    args = parser.parse_args()
    print(args)

    main(args)
    
