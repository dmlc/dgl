#-*- coding:utf-8 -*-


# The major idea of the overall GNN model explanation


import argparse
import os
import dgl

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AIFBDataset
from dgl.sampling import sample_neighbors
from models import dummy_gnn_model
from gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4, gen_syn5
from NodeExplainerModule import NodeExplainerModule
from utils_graph import extract_subgraph, visualize_sub_graph


def main(args):
    # load an exisitng model or ask for training a model
    if os.path.exists(args.model_path):
        model_stat_dict = th.load(args.model_path)
        dataset = args.model_path[-8:-4]
    else:
        raise FileExistsError('No Saved Model file. Please train a GNN model first...')

    # create dataset of the model trained
    if dataset == 'aifb':
        dataset = AIFBDataset()
    elif dataset == 'syn1':
        # g, labels, name = gen_syn1(nb_shapes=8, width_basis=30)
        g, labels, name = gen_syn1()
    elif dataset == 'syn2':
        g, labels, name = gen_syn2()
    elif dataset == 'syn3':
        g, labels, name = gen_syn3()
    elif dataset == 'syn4':
        g, labels, name = gen_syn4()
    elif dataset == 'syn5':
        g, labels, name = gen_syn5()
    else:
        raise ValueError()

    # generate dataset and related variables
    graph = dgl.from_networkx(g)
    graph = dgl.to_bidirected(graph)

    num_classes = max(labels) + 1
    feat_dim = 10

    # feed saved stat dictionary into a model
    dummy_model = dummy_gnn_model(feat_dim, 40, num_classes)
    dummy_model.load_state_dict(model_stat_dict)

    # set a node to be explained and extract its subgraph
    # For the test purpose, chose a node with label 1, which specifies the two nodes in the middle of a house
    l_1 = [i for i, e in enumerate(labels) if e==1]
    print(l_1)

    # here for test purpose, just pick the first one
    n_idx = th.tensor([l_1[-1]])

    sub_graph, new_n_idx = extract_subgraph(graph, n_idx)
    print(new_n_idx)
    src = sub_graph.edges()[0]
    dst = sub_graph.edges()[1]

    num_edges = sub_graph.number_of_edges()
    n_feats = th.randn(sub_graph.number_of_nodes(), feat_dim)

    # create an explainer
    explainer = NodeExplainerModule(model=dummy_model,
                                    num_edges=num_edges,
                                    node_feat_dim=feat_dim)

    # define optimizer
    optim = th.optim.Adam(explainer.parameters(), lr=0.01)

    # train the explainer for a given node
    dummy_model.eval()
    model_logits = dummy_model(sub_graph, n_feats)
    model_predict = F.one_hot(th.argmax(model_logits, dim=-1), num_classes)

    for epoch in range(500):
        explainer.train()
        exp_logits = explainer(sub_graph, n_idx[0], n_feats)
        loss = explainer._loss(exp_logits[new_n_idx], model_predict[new_n_idx])

        optim.zero_grad()
        loss.backward()
        optim.step()

        # print("In epoch: {:03d}, Loss: {:.6f}, Pred: {}".format(epoch, loss.item(), model_predict[new_n_idx]))

    # visualize the importance of edges
    edge_weights = explainer.edge_mask.sigmoid().detach()

    total_t = th.cat([th.stack([src,dst], dim=-1), edge_weights], dim=-1)
    print(total_t)

    # draw_sub_graph(sub_graph)
    visualize_sub_graph(sub_graph, edge_weights.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of GNN explainer in DGL')
    parser.add_argument('--model_path', type=str, default='./dummy_model_4_syn1.pth',
                        help='The path of saved model to be explained.')
    # parser.add_argument('--dataset', type=str, default='aifb', help='dataset used for training the model')


    args = parser.parse_args()
    print(args)

    main(args)