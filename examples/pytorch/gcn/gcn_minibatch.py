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
from dgl.data import register_data_args, load_data
import dgl.sampling as sampling


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


    def forward(self, features, labels, mask, fn_reduce=F.log_softmax):
        ret = torch.zeros((features.size()[0], self.n_classes))
        #print(ret.size())
        for batch in self.batches:
            nodes_prev = None
            for l_id, layer in enumerate(self.layers):
                #print(l_id)                
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
            
            idx_sub = torch.tensor(idx[:, 1])
            ret.index_copy_(0, idx_sub, fn_reduce(g_sub.get_n_repr(), 1))
            
        return ret
            #ret[] = g_sub.get_n_repr(u=batch_map[l_id])  
#            #logits = model(features, labels)
#            logp = fn_reduce(logits, 1)
#            
            
            
def main(args, dict_args={}):
    # load and preprocess dataset
    data = load_data(args)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
    #print(mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        mask = mask.cuda()

    # create GCN model
    g = DGLGraph(data.graph)
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout, 
                fn_batch=dict_args["fn_batch"],
                fn_batch_params=dict_args["fn_batch_params"],
                fn_seed=dict_args["fn_seed"],
                fn_seed_params=dict_args["fn_seed_params"])            
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize graph
    dur = []

    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        
        fn_loss=F.nll_loss
        logp = model(features, labels, mask, fn_reduce=F.log_softmax)
        loss = fn_loss(logp[mask], labels[mask])  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, loss.item(), np.mean(dur), n_edges / np.mean(dur) / 1000))

if __name__ == '__main__':
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
    print(args)

    dict_args = {}
    dict_args["fn_batch"] = sampling.minibatch
    dict_args["fn_seed"] = sampling.seeds
    
    dict_args["fn_batch_params"] ={"depth": 2, "fn_neighborhood": sampling.neighborhood_networkx, "max_neighbors":100}
    dict_args["fn_seed_params"] = {'seed_size':10, "num_batches":270}
    
    main(args, dict_args)


