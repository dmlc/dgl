#-*- coding:utf-8 -*-


# The major idea of the overall GNN model explanation


import argparse
import os
import dgl

import torch as th
from dgl.data import AIFBDataset
from dgl.sampling import sample_neighbors
from models import dummy_gnn_model
from gengraph import gen_syn1, gen_syn2, gen_syn3, gen_syn4, gen_syn5
from NodeExplainerModule import NodeExplainerModule



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
        g, role_ids, name = gen_syn1()
    elif dataset == 'syn2':
        g, role_ids, name = gen_syn2()
    elif dataset == 'syn3':
        g, role_ids, name = gen_syn3()
    elif dataset == 'syn4':
        g, role_ids, name = gen_syn4()
    elif dataset == 'syn5':
        g, role_ids, name = gen_syn5()
    else:
        raise ValueError()

    # generate dataset and related variables
    graph = dgl.from_networkx(g)
    graph = dgl.to_bidirected(graph)

    num_classes = max(role_ids) + 1
    feat_dim = 10

    # feed saved stat dictionary into a model
    dummy_model = dummy_gnn_model(feat_dim, 40, num_classes)
    dummy_model.load_state_dict(model_stat_dict)

    # set a node to be explained and extract its subgraph
    # here for test purpose, just pick the first one
    n_idx = th.tensor(690)

    one_hop_subg = sample_neighbors(graph, n_idx, -1)
    print(one_hop_subg.edges())

    one_hop_nodes = one_hop_subg.ndata[dgl.NID]
    two_hop_subg = dgl.node_subgraph(graph, one_hop_nodes)
    print(two_hop_subg)

    num_edges = two_hop_subg.number_of_edges()

    # create an explainer
    explainer = NodeExplainerModule(model=dummy_model,
                                    num_edges=num_edges,
                                    node_feat_dim=feat_dim)

    #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of GNN explainer in DGL')
    parser.add_argument('--model_path', type=str, default='./dummy_model_4_syn1.pth',
                        help='The path of saved model to be explained.')
    # parser.add_argument('--dataset', type=str, default='aifb', help='dataset used for training the model')

    args = parser.parse_args()
    print(args)

    main(args)