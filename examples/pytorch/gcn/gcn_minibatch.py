"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with batch processing
"""
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl import DGLSubGraph
from dgl.data import register_data_args, load_data, load_data_dict
import dgl.sampling as sampling
import sys

def gcn_msg(src, edge):
    return src

def gcn_reduce(node, msgs):
    return torch.sum(msgs, 1)

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        #print(node)
        h = self.linear(node)
        if self.activation:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout, 
                 fn_batch,
                 fn_batch_params,
                 fn_seed,
                 fn_seed_params):
        super(GCN, self).__init__()
        self.g = g
        self.dropout = dropout
        self.n_classes = n_classes
        # input layer
        self.layers = nn.ModuleList([NodeApplyModule(in_feats, n_hidden, activation)])

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(NodeApplyModule(n_hidden, n_hidden, activation))

        # output layer
        self.layers.append(NodeApplyModule(n_hidden, n_classes))

        #generate minibatches
        fn_seed_params["V"] = g.nodes
        fn_batch_params["seedset_list"] = fn_seed(**fn_seed_params)
        fn_batch_params["G"] = g
        self.batches = fn_batch(**fn_batch_params)


    def forward(self, features, labels, mask, fn_reduce=F.log_softmax, log=False):
        ret = torch.zeros((features.size()[0], self.n_classes))
        node_count = 0
        for batch in self.batches:
            nodes_prev = None
            for l_id, layer in enumerate(self.layers):               
                nodes = set.union(*[nodeset for depth,nodeset in batch.items() if depth >= l_id])
                if l_id==0:                                      
                    g_sub = DGLSubGraph(parent=self.g, nodes=list(nodes))
                    a = torch.tensor(list(nodes))
                    features_sub = torch.index_select(features, dim=0, index=a)
                    g_sub.set_n_repr(features_sub)
                else:                    
                    idx = np.array([[i, key_i] for i, key_i in enumerate(nodes_prev) if key_i in nodes])
                    g_sub_new = DGLSubGraph(parent=g_sub, nodes=idx[:, 0])
                    g_sub_new.copy_from(parent=g_sub)
                    g_sub = g_sub_new
                nodes_prev = list(nodes)
                g_sub.update_all(gcn_msg, gcn_reduce, layer, batchable=True)
                node_count+= len(g_sub.nodes)
                #print(len(g_sub.nodes))
            idx_sub = torch.tensor(idx[:, 1])
            ret.index_copy_(0, idx_sub, fn_reduce(g_sub.get_n_repr(), 1))
            
        return ret, node_count            
            
def main(params, data = None):
    ret = {'time':[], 'loss':[], 'nodes':[] }
    # load and preprocess dataset
    if data is None:
        data = load_data_dict(params)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    #print(mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels

    if params["gpu"] < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(params["gpu"])
        features = features.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    # create GCN model
    g = DGLGraph(data.graph)
    model = GCN(g,
                in_feats,
                params["hidden"],
                n_classes,
                params["layers"],
                F.relu,
                params["dropout"], 
                fn_batch=params["fn_batch"],
                fn_batch_params=params["fn_batch_params"],
                fn_seed=params["fn_seed"],
                fn_seed_params=params["fn_seed_params"])            
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    for epoch in range(params["epochs"]):
        t=time.time()
        logp, node_count = model(features, labels, mask, fn_reduce=F.log_softmax)            
        loss = params["fn_loss"](logp[mask], labels[mask])  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_delta = time.time()-t
        
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | Nodes {:d}".format(
            epoch, loss.item(), t_delta, node_count))
        ret["time"].append(t_delta)
        ret["loss"].append(loss.item())
        ret["nodes"].append(node_count)
    return ret


def default_params():

    fn_batch_params = {"depth": 2, "fn_neighborhood": sampling.neighborhood_networkx, "max_neighbors":100}
    fn_seed_params = {'seed_size':10}
    
    return {"dataset":"cora",
              "dropout":0,
              "gpu":-1,
              "lr":1e-3,
              "epochs":20, 
              "hidden":16,
              "layers": fn_batch_params["depth"], 
              "fn_batch": sampling.minibatch,
              "fn_seed": sampling.seeds_consume,
              "fn_batch_params":fn_batch_params,
              "fn_seed_params":fn_seed_params,
              "fn_loss": F.nll_loss}

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    args = parser.parse_args()
    #print(args)
    params = default_params()
    params["dataset"] =args.dataset
    params["dropout"]=args.dropout
    params["gpu"]=args.gpu
    params["lr"]=args.lr
    params["epochs"]=args.n_epochs 
    params["hidden"]=args.h_hidden
    params["layers"]=args.n_layers
    r = main(params)

    



