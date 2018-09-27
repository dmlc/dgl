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
import dgl.data as dgld
import dgl.sampling as sampling
import sys

                 
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        node['h'] = self.linear(node['h'])
        if self.activation:
            node['h'] = self.activation(node['h'])
        return node

class GCN(nn.Module):   
    def gcn_msg(self, src, edge):
        return src  
    def gcn_reduce(self, node, msgs):
        return torch.sum(msgs['h'], 1)   
    def importance_msg(self, src, edge):        
        src = dict(src)
        src['h'] = src['h'] / src['q'].expand_as(src['h'])
        return src   
    def importance_reduce(self, node, msgs):
        return torch.mean(msgs['h'], 1)   
    
    def add_self_edges_networkx(self, G):
        for i in G.nodes:
            G.add_edge(i,i)
        return G
    
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout, 
                 fn_batch,
                 fn_batch_params):
        super(GCN, self).__init__()
        
        #add self loops
        g = self.add_self_edges_networkx(g)
        
        self.fn_batch = fn_batch
        self.fn_batch_params = fn_batch_params
        
        self.g = g    
        self.dropout = dropout
        self.n_classes = n_classes
        
        if fn_batch == sampling.importance_sampling: 
            self.q = sampling.importance_sampling_distribution_networkx(g, 'degree')
        
        # input layer
        self.layers = nn.ModuleList([NodeApplyModule(in_feats, n_hidden, activation)])

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(NodeApplyModule(n_hidden, n_hidden, activation))

        # output layer
        self.layers.append(NodeApplyModule(n_hidden, n_classes))
        
    def forward(self, features, labels, mask, optimizer, fn_reduce=F.log_softmax, fn_loss=F.nll_loss):

        #generate minibatches
        if self.fn_batch == sampling.importance_sampling:  #if Importance Sampling (IS)
            self.fn_batch_params["q"] = self.q 
            self.batch_type={"IS":True, "NS":False}  #convenience flags
            self.batches = self.fn_batch(**self.fn_batch_params)
            q_rep = torch.unsqueeze(torch.tensor([self.q[i] if i in self.q else 0 for i in range(features.shape[0])]),1) #store degree distribution
            gcn_reduce = self.importance_reduce
            gcn_msg = self.importance_msg            
            #self.q = q
        elif self.fn_batch in [sampling.seed_expansion_sampling, sampling.seed_BFS_frontier_sampling, sampling.seed_random_walk_sampling]: #if neighborhood sampling
            self.fn_batch_params["G"] = self.g 
            self.batch_type={"IS":False, "NS":True}
            self.batches = self.fn_batch(**self.fn_batch_params)
            self.g.set_n_repr({'h':features})
            gcn_reduce = self.gcn_reduce
            gcn_msg = self.gcn_msg            
        else:
            raise Exception("sampling not supported") #what are you doing?
        
        #ret = torch.zeros((features.size()[0], self.n_classes)) #output layer aggregate 
        batch_sizes = []
        node_count = 0 #bookkeeping
        for batch in self.batches:
            nodes_prev = None
            for l_id, layer in enumerate(self.layers):            
                #handle subgraph nodes
                if self.batch_type["IS"]: # use fixed node set
                    nodes = batch[0]
                elif self.batch_type["NS"]: #expand node set
                    nodes = set.union(*[nodeset for depth,nodeset in batch.items() if depth >= l_id])

                if l_id==0:  #if level 0 use input features   
                    batch_sizes.append(len(nodes))                                  
                    g_sub = dgl.DGLSubGraph(parent=self.g, nodes=list(nodes))
                    a = torch.tensor(list(nodes))
                    features_sub = torch.index_select(features, dim=0, index=a)
                    labels_sub = torch.index_select(labels, dim=0, index=a)
                    mask_sub = torch.index_select(mask, dim=0, index=a)
                    if self.batch_type["IS"]:  #branch because we might need more fields
                        q_sub = torch.index_select(q_rep, dim=0, index=a)
                        g_sub.set_n_repr({'h':features_sub, 'q':q_sub})
                    elif self.batch_type["NS"]:
                        g_sub.set_n_repr({'h':features_sub})
                else:  #else deeper level                  
                    idx = np.array([[i, key_i] for i, key_i in enumerate(nodes_prev) if key_i in nodes])  #if subgraph shrinks, reindex. e.g. where are old [0, M] indices in [0, N] ? M >= N 
                    g_sub_new = dgl.DGLSubGraph(parent=g_sub, nodes=idx[:, 0])
                    g_sub_new.copy_from(parent=g_sub)
                    labels_sub = torch.index_select(labels_sub, dim=0, index=torch.tensor(idx[:, 0]))
                    mask_sub = torch.index_select(mask_sub, dim=0, index=torch.tensor(idx[:, 0]))
                    g_sub = g_sub_new
                nodes_prev = list(nodes)
                g_sub.update_all(gcn_msg, gcn_reduce, layer, batchable=True)
                node_count+= len(g_sub.nodes)             
            if torch.nonzero(mask_sub).size(0):
                loss = fn_loss(fn_reduce(g_sub.pop_n_repr(key='h'))[mask_sub], labels_sub[mask_sub])  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
        return loss.item(), node_count , batch_sizes           
           
def main(params, data = None):
    ret = {'time':[], 'loss':[], 'nodes':[] }
    # load and preprocess dataset
    if data is None:
        data = dgld.load_data_dict(params)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.ByteTensor(data.train_mask)
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
    g = dgl.DGLGraph(data.graph)
    model = GCN(g,
                in_feats,
                params["hidden"],
                n_classes,
                params["layers"],
                F.relu,
                params["dropout"], 
                fn_batch=params["fn_batch"],
                fn_batch_params=params["fn_batch_params"])            
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    t_total=0
    n_counts = []
    losses = []
    batches = []
    for epoch in range(params["epochs"]):
        t=time.time()
        loss, node_count, batch_size = model(features, labels, mask, optimizer, fn_reduce=F.log_softmax, fn_loss=params["fn_loss"])         
        t_total += time.time()-t
        losses.append(loss)
        batches.extend(batch_size)
        n_counts.append(node_count)    
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | Node Updates {:d} | Mean Subgraph Size {:d}".format(
            epoch, loss, t_total, int(np.round(np.mean(n_counts))), int(np.round(np.mean(batches))) ))
    ret["time"]= t_total
    ret["loss"]= losses[-1]
    ret["mean node updates"] = int(np.round(np.mean(n_counts)))
    ret["mean batch size"] = int(np.round(np.mean(batches)))
    return ret

#build all default parameters for execution
def default_params():
    
    fn_batch_params = {"depth": 3, "fn_neighborhood": sampling.neighborhood_networkx, "max_level_nodes":100, 'seed_size':10, 'percent_nodes':0.9}    
    return {"dataset":"cora",
              "dropout":0,
              "gpu":-1,
              "lr":1e-3,
              "epochs":20, 
              "hidden":16,
              "layers": fn_batch_params["depth"]-1, 
              "fn_batch": sampling.seed_BFS_frontier_sampling,
              "fn_batch_params":fn_batch_params,
              "fn_loss": F.nll_loss}

#handle command line params
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='GCN')
    dgld.register_data_args(parser)
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

    params = default_params()
    params["dataset"] =args.dataset
    params["dropout"]=args.dropout
    params["gpu"]=args.gpu
    params["lr"]=args.lr
    params["epochs"]=args.n_epochs 
    params["hidden"]=args.h_hidden
    params["layers"]=args.n_layers
    r = main(params)