"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn

GCN with batch processing
"""
import argparse
import numpy as np
import numpy.random as npr
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data as dgld
import dgl.sampling as sampling
import sys
import collections as c
                 
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, height=None):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.height = height  #track height so we can apply different functions in forward

    def forward(self, node):
        node = dict(node)
        node['h'] = self.linear(node['h'])             
        if self.activation:
            node['h'] = self.activation(node['h'])
        if 'h_old' in node and self.height == 0: #if we set h_old and we just built h0
            node['h_old'] = node['h'] #copy from activated 
        return node

class GCN(nn.Module): 

    
    ##define different messaging and reductions
    
    ##base
    def gcn_msg(self, src, edge):
        return src  
    def gcn_reduce(self, node, msgs):
        return torch.sum(msgs['h'], 1)   
    
    ##importance sampling
    def importance_msg(self, src, edge):        
        src = dict(src)
        src['h'] = src['h'] / src['q'].expand_as(src['h'])
        return src   
    def importance_reduce(self, node, msgs):
        return torch.mean(msgs['h'], 1)   
    
    ##control variate
    def cv_reduce(self, node, msgs):
        node = dict(node)
        
        if self.layer_curr == 0: #if we're reducing on node features
            h = torch.mean(msgs['h'], 1)
            return {'h':h, 'h_old':h}
        elif msgs['h'].size(1) >= self.cv and node["h"].size(0) == msgs["h"].size(0) and node["h"].size(-1) == msgs["h"].size(-1): #else if we have enough neighbors to subsample and they're correct dimensionality 
                                                                                                                                    #this edge case actually isnt handled  in the paper, we revert to mean 
            #sample 'self.cv' random neighbors 
            idx_select = int(np.floor(msgs['h'].size(1)*self.cv))
            idx = [int(i) for i in npr.choice(msgs['h'].size(1), idx_select)]
            
            #subset and stack
            h = torch.index_select(msgs["h"], dim=1, index=torch.tensor(idx))
            h_old = torch.index_select(msgs["h_old"], dim=1, index=torch.tensor(idx))            
            h_cat = torch.cat((h - h_old, msgs['h_old']), dim=1)  # sum(h_delta) + sum(h_old) in original paper (Eq. 5) 
            
            #mean over stack
            node['h'] = torch.mean(h_cat, 1) 
            
            return node
        else:
            node['h'] = torch.mean(msgs['h'], 1)
            return node            

    def add_self_edges_networkx(self, G):
        for i in G.nodes:
            G.add_edge(i,i)
        return G
        
    def __init__(self,
                 g,
                 in_feats, #in feature shape
                 n_hidden,  #number of hidden nodes
                 n_classes,  #number of output classes
                 n_layers,    #number of layers
                 activation,   #function pointer for node activation
                 dropout,    #dropout fraction
                 fn_batch,   #function pointer for minibatch sampling (see: dgl.sampling)
                 fn_batch_params, #parameter dictionary to be passed to 'fn_batch' using ** unpacking
                 cv=None):   #whether we apply control variate sampling
        super(GCN, self).__init__()
        
        #add self loops
        g = self.add_self_edges_networkx(g)
        
        self.fn_batch = fn_batch
        self.fn_batch_params = fn_batch_params
        
        self.g = g    
        self.dropout = dropout
        self.n_classes = n_classes
        self.layer_curr=0
        if isinstance(cv, (int, float)):
            self.cv = cv
        else:
            self.cv = None
            
        # input layer
        self.layers = nn.ModuleList([NodeApplyModule(in_feats, n_hidden, activation, height=0)])

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(NodeApplyModule(n_hidden, n_hidden, activation, height=i+1))

        # output layer
        self.layers.append(NodeApplyModule(n_hidden, n_classes, height=n_layers))

    def forward(self, 
                features,                    #train features
                labels,                      #train labels
                mask,                        #labels to calculate loss
                optimizer,                   #optimizer class
                cv=False,                    # int the number of neighbors to select for control variate sampling (default: no CV)
                fn_reduce=F.log_softmax,     #output layer reduction function
                fn_loss=F.nll_loss,          #loss function
                validation=False):           #are we validating a trained model? (no backprop)
                #TODO: the pytorch superclass forces this as .forward(). I'd prefer .train()
                #since we're iterating over minibatches (and therefore need to backprop) 
                
        self.fn_batch_params["G"] = self.g 

        if self.dropout:
            features = F.dropout(features, p=self.dropout)
        #generate minibatches
        if self.fn_batch == sampling.importance_sampling_wrapper_networkx:  #if Importance Sampling (IS)
            batches, self.q = sampling.importance_sampling_wrapper_networkx(**self.fn_batch_params)
            q_rep = torch.unsqueeze(torch.tensor([self.q[i] if i in self.q else 0 for i in range(features.shape[0])]),1) #store degree distribution
            gcn_reduce = self.importance_reduce
            gcn_msg = self.importance_msg     
            
            batch_type={"IS":True, "NS":False, "Null":False}  #convenience flags
            #self.q = q
        elif self.fn_batch in [sampling.seed_expansion_sampling, sampling.seed_BFS_frontier_sampling, sampling.seed_random_walk_sampling, sampling.full_batch_networkx]: #if neighborhood sampling
            gcn_reduce = self.gcn_reduce
            gcn_msg = self.gcn_msg
            batches = self.fn_batch(**self.fn_batch_params)
            batch_type={"IS":False, "NS":True, "Null":False}     
        elif self.fn_batch is None:
            batch_type={"IS":False, "NS":False, "Null":True}
            batches = [[]]
        else:
            raise Exception("sampling not supported") #what are you doing?
        if self.cv: #if control variate
            gcn_reduce = self.cv_reduce   
        if validation: #if we're not training, run full-batch with respect to G
            batches = [{i:set(self.g.nodes) for i in range(len(self.layers))}]       
        
        #bookkeeping
        books = c.defaultdict(list)
        book_keys = ['batch sizes', 'total node updates','time batch sample', 'time data copy', 'time node update', 'time backward', 'total loss', 'eval node counts']
        
        t_batch_start = 0
        for batch in batches:
            for k_iter in book_keys:
                books[k_iter].append(0)
                
            if t_batch_start != 0:
                books['time batch sample'][-1] += time.time()-t_batch_start
                
            nodes_prev = None
            for l_id, layer in enumerate(self.layers):  
                self.layer_curr = l_id
                
                #handle subgraph nodes
                if batch_type["IS"]: # use fixed node set
                    nodes = batch[0]
                elif batch_type["NS"]: #expand node set
                    nodes = set.union(*[nodeset for depth,nodeset in batch.items() if depth >= l_id])
                elif batch_type["Null"]:
                    break

                if l_id==0:  #if level 0 use input features   
                    books['batch sizes'][-1] += len(nodes)                     
                    t = time.time()
                    g_sub = dgl.DGLSubGraph(parent=self.g, nodes=list(nodes))
                    a = torch.tensor(list(nodes))
                    features_sub = torch.index_select(features, dim=0, index=a)
                    labels_sub = torch.index_select(labels, dim=0, index=a)
                    mask_sub = torch.index_select(mask, dim=0, index=a)
                    if batch_type["IS"]:  #branch because we might need more fields
                        q_sub = torch.index_select(q_rep, dim=0, index=a)
                        g_sub.set_n_repr({'h':features_sub, 'q':q_sub})
                    elif batch_type["NS"]:
                        g_sub.set_n_repr({'h':features_sub})
                    books['time data copy'][-1] += time.time()-t
                else:  #else deeper level  
                    #subset our graph
                    
                    t = time.time()
                    idx = np.array([[i, key_i] for i, key_i in enumerate(nodes_prev) if key_i in nodes])  #if subgraph shrinks, reindex. e.g. where are old [0, M] indices in [0, N] ? M >= N 
                    g_sub_new = dgl.DGLSubGraph(parent=g_sub, nodes=idx[:, 0])
                    g_sub_new.copy_from(parent=g_sub)
                    labels_sub = torch.index_select(labels_sub, dim=0, index=torch.tensor(idx[:, 0]))
                    mask_sub = torch.index_select(mask_sub, dim=0, index=torch.tensor(idx[:, 0]))
                    books['time data copy'][-1] += time.time()-t
                    g_sub = g_sub_new
                nodes_prev = list(nodes)
                t = time.time()
                g_sub.update_all(gcn_msg, gcn_reduce, layer, batchable=True)
                books['time node update'][-1] += time.time()-t
                books['total node updates'][-1] += len(g_sub.nodes)
                
            if batch_type["Null"]:
                mask_sub = mask
                rep = torch.tensor(npr.random(size=(labels.size(0), self.n_classes)))
                validation = True
                labels_sub = labels
            else:
                t = time.time()
                rep = g_sub.pop_n_repr(key='h')
                books['time data copy'][-1] += time.time()-t
            if torch.nonzero(mask_sub).size(0): #if we have evaluation labels in this batch
                t = time.time()
                l_pred = fn_reduce(rep, dim=0)[mask_sub]                  
                loss = fn_loss(l_pred, labels_sub[mask_sub], reduction='sum')  #sum reduction because we'll aggregate over minibatches
                books["total loss"][-1] += float(loss)
                books["eval node counts"][-1] += float(torch.nonzero(mask_sub).size(0))  #number of labeled instances
                if not validation: #if we're training, backprop 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  
                books['time backward'][-1] += time.time()-t
            t_batch_start = time.time() 
        return sum(books["total loss"])/sum(books["eval node counts"]), books         
           
def main(params, data = None):
    # load and preprocess dataset
    if data is None:
        data = dgld.load_data_dict(params)
    if "train fraction" not in params:
        params["train fraction"] = 0.8

    features = torch.FloatTensor(data.features)
        
    split_ind = int(np.floor(len(data.labels)*params["train fraction"]))
    idx = npr.permutation(range(len(data.labels)))
    mask = np.zeros(len(data.labels))
    mask[idx[0:split_ind]] = 1
    mask = torch.ByteTensor(mask)    
    mask_test = np.zeros(len(data.labels))
    mask_test[idx[split_ind::]] = 1
    mask_test= torch.ByteTensor(mask_test)
    
    labels = torch.LongTensor(data.labels)
    
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
                fn_batch_params=params["fn_batch_params"], 
                cv=params["cv"])            
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    books = c.defaultdict(int)
    book_sum_keys = ['time batch sample', 'time data copy', 'time node update', 'time backward']
    
    
    t_train_total=0
    t_test_total=0

    n_counts = []
    loss_train = []
    loss_test = []
    batches = []
    for epoch in range(params["epochs"]):
        t=time.time()
        l_train, books_train = model(features, labels, mask, optimizer, fn_reduce=F.log_softmax, fn_loss=params["fn_loss"])   
        t_train_total += time.time()-t
        
        t = time.time()
        l_test, _ = model(features, labels, mask_test, optimizer=None, fn_reduce=F.log_softmax, fn_loss=params["fn_loss"], validation=True) 
        t_test_total += time.time()-t
        
        loss_train.append(l_train)
        loss_test.append(l_test)
        batches.extend(books_train['batch sizes'])
        #batches.extend(batch_size)
        n_counts.append(sum(books_train['total node updates']))
        for k_iter in book_sum_keys:
            books[k_iter] += sum(books_train[k_iter])

        if "verbose" in params and params["verbose"]:
            total_time = books['time batch sample']+ books['time data copy']+ books['time node update']+ books['time backward']
            print("[TRAINING]: Epoch {:05d} | Loss (Train, Test) ({:.3f}, {:.3f}) | Time (Train, Test)  ({:.3f}, {:.3f}) | Time % (Batch, Data, Update, Back) ({:.3f}, {:.3f}, {:.3f}, {:.3f})  | Mean Subgraph Size {:d}".format(
                epoch, l_train, l_test, t_train_total, t_test_total, books['time batch sample']/total_time, books['time data copy']/total_time, books['time node update']/total_time, books['time backward']/total_time, int(np.round(np.mean(batches))) ))   
    books["loss test"] =loss_test 
    books["time train"]= t_train_total
    books["time test"] = t_test_total
    books["loss train"]= loss_train
    books["loss test"]= loss_test
    books["mean node updates"] = int(np.round(np.mean(n_counts)))
    books["mean batch size"] = int(np.round(np.mean(batches)))
    print("[TEST]: Best Loss (Train, Test) ({:.4f}, {:.4f}) | Time (Train, Test) ({:.4f}, {:.4f})".format(np.min(books["loss train"]),np.min(books["loss test"]), books["time train"], books["time test"]))
    return books

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
              "fn_loss": F.nll_loss,
              "cv":None}

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