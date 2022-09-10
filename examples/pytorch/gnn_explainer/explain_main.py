# The major idea of the overall GNN model explanation

import argparse
import os
import dgl
from gnnlens import Writer
import torch as th

from dgl import load_graphs
from dgl.nn import GNNExplainer
from models import dummy_gnn_model

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
    
    explainer = GNNExplainer(dummy_model, num_hops=args.hop)
    new_center, sub_graph, feat_mask, edge_mask = explainer.explain_node(n_idx, graph, feats)
    
    #gnnlens2
    # Specify the path to create a new directory for dumping data files.
    writer = Writer('gnn_subgraph')
    writer.add_graph(name=str(args.dataset), graph=graph,
                     nlabels=labels, num_nlabel_types=num_classes)
    writer.add_subgraph(graph_name = str(args.dataset), subgraph_name='IntegratedGradients', node_id=n_idx,
                    subgraph_nids = sub_graph.ndata[dgl.NID],
                    subgraph_eids = sub_graph.edata[dgl.EID],
                    subgraph_eweights = sub_graph.edata['ew'])

    # Finish dumping
    writer.close()
    

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
    
