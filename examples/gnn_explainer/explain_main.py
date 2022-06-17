# The major idea of the overall GNN model explanation

import argparse
import os
import dgl

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl import load_graphs
from models import dummy_gnn_model
from NodeExplainerModule import NodeExplainerModule
from utils_graph import extract_subgraph, visualize_sub_graph


def main(args):
    # load an exisitng model or ask for training a model
    model_path = os.path.join('./', 'dummy_model_{}.pth'.format(args.dataset))
    if os.path.exists(model_path):
        model_stat_dict = th.load(model_path)
    else:
        raise FileExistsError('No Saved Model file. Please train a GNN model first...')

    # load graph, feat, and label
    g_list, label_dict = load_graphs('./'+args.dataset+'.bin')
    graph = g_list[0]
    labels = graph.ndata['label']
    feats = graph.ndata['feat']
    num_classes = max(labels).item() + 1
    feat_dim = feats.shape[1]
    hid_dim = label_dict['hid_dim'].item()
    
    # create a model and load from state_dict
    dummy_model = dummy_gnn_model(feat_dim, hid_dim, num_classes)
    dummy_model.load_state_dict(model_stat_dict)

    # Choose a node of the target class to be explained and extract its subgraph.
    # Here just pick the first one of the target class.
    target_list = [i for i, e in enumerate(labels) if e==args.target_class]
    n_idx = th.tensor([target_list[0]])

    # Extract the computation graph within k-hop of target node and use it for explainability
    sub_graph, ori_n_idxes, new_n_idx = extract_subgraph(graph, n_idx, hops=args.hop)
    
    #Sub-graph features.
    sub_feats = feats[ori_n_idxes,:]

    # create an explainer
    explainer = NodeExplainerModule(model=dummy_model,
                                    num_edges=sub_graph.number_of_edges(),
                                    node_feat_dim=feat_dim)

    # define optimizer
    optim = th.optim.Adam([explainer.edge_mask, explainer.node_feat_mask], lr=args.lr, weight_decay=args.wd)

    # train the explainer for the given node
    dummy_model.eval()
    model_logits = dummy_model(sub_graph, sub_feats)
    model_predict = F.one_hot(th.argmax(model_logits, dim=-1), num_classes)

    for epoch in range(args.epochs):
        explainer.train()
        exp_logits = explainer(sub_graph, sub_feats)
        loss = explainer._loss(exp_logits[new_n_idx], model_predict[new_n_idx])

        optim.zero_grad()
        loss.backward()
        optim.step()

    # visualize the importance of edges
    edge_weights = explainer.edge_mask.sigmoid().detach()
    visualize_sub_graph(sub_graph, edge_weights.numpy(), ori_n_idxes, n_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of GNN explainer in DGL')
    parser.add_argument('--dataset', type=str, default='syn1',
                        help='The dataset to be explained.')
    parser.add_argument('--target_class', type=int, default='1',
                        help='The class to be explained. In the synthetic 1 dataset, Valid option is from 0 to 4'
                             'Will choose the first node in this class to explain')
    parser.add_argument('--hop', type=int, default='2',
                        help='The hop number of the computation sub-graph. For syn1 and syn2, k=2. For syn3, syn4, and syn5, k=4.')
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate.')
    parser.add_argument('--wd', type=float, default=0.0, help='Weight decay.')
    args = parser.parse_args()
    print(args)

    main(args)
    
