from numpy.core.numeric import full
import torch
import dgl
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os
from functools import partial
from tqdm import tqdm

# TODO: Preprocesser
def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True):
  #Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)



# TODO: Define dataset

def TemporalWikipediaDataset():
    # Download the dataset
    if not os.path.exists('./data/wikipedia.bin'):
        if not os.path.exists('./data/wikipedia.csv'):
            if not os.path.exists('./data'):
                os.mkdir('./data')

            url = 'https://snap.stanford.edu/jodie/wikipedia.csv'
            print("Start Downloading File....")
            r = requests.get(url,stream=True)
            with open("./data/wikipedia.csv","wb") as handle:
                for data in tqdm(r.iter_content()):
                    handle.write(data)

        print("Start Process Data ...")
        run('wikipedia')
        raw_connection = pd.read_csv('./data/ml_wikipedia.csv')
        raw_feature    = np.load('./data/ml_wikipedia.npy')
        # -1 for re-index the node
        src = raw_connection['u'].to_numpy()-1
        dst = raw_connection['i'].to_numpy()-1
        # Create directed graph
        g = dgl.graph((src,dst))
        g.edata['timestamp'] =torch.from_numpy(raw_connection['ts'].to_numpy())
        g.edata['label']     =torch.from_numpy(raw_connection['label'].to_numpy())
        g.edata['feats']     =torch.from_numpy(raw_feature[1:,:]).float()
        dgl.save_graphs('./data/wikipedia.bin',[g])
    else:
        print("Data is exist directly loaded.")
        gs,_ = dgl.load_graphs('./data/wikipedia.bin')
        g = gs[0]
    return g

def TemporalRedditDataset():
    # Download the dataset
    if not os.path.exists('./data/reddit.bin'):
        if not os.path.exists('./data/reddit.csv'):
            if not os.path.exists('./data'):
                os.mkdir('./data')

            url = 'https://snap.stanford.edu/jodie/reddit.csv'
            print("Start Downloading File....")
            r = requests.get(url,stream=True)
            with open("./data/reddit.csv","wb") as handle:
                for data in tqdm(r.iter_content()):
                    handle.write(data)

        print("Start Process Data ...")
        run('reddit')
        raw_connection = pd.read_csv('./data/ml_reddit.csv')
        raw_feature    = np.load('./data/ml_reddit.npy')
        src = raw_connection['u'].to_numpy()-1
        dst = raw_connection['i'].to_numpy()-1
        # Create directed graph
        g = dgl.graph((src,dst))
        g.edata['timestamp'] =torch.from_numpy(raw_connection['ts'].to_numpy())
        g.edata['label']     =torch.from_numpy(raw_connection['label'].to_numpy())
        g.edata['feats']     =torch.from_numpy(raw_feature[1:,:]).float()
        dgl.save_graphs('./data/reddit.bin',[g])
    else:
        print("Data is exist directly loaded.")
        gs,_ = dgl.load_graphs('./data/reddit.bin')
        g = gs[0]
    return g

def negative_sampler(g,size):
    # Supposingly size 1
    n_edge = g.num_edges()
    idx = np.random.randint(0,n_edge,size)
    dst = g.edges()[1] # A torch tensor
    ret = dst[idx]
    return ret

def temporal_sort(g,key):
    edge_keys = list(g.edata.keys())
    node_keys = list(g.ndata.keys())

    sorted_idx = g.edata[key].sort()[1]
    buf_graph = dgl.graph((g.edges()[0][sorted_idx],g.edges()[1][sorted_idx]))
    # copy back edge and node data
    for ek in edge_keys:
        buf_graph.edata[ek] = g.edata[ek][sorted_idx]

    # Since node index unchanged direct copy
    for nk in node_keys:
        buf_graph.ndata[nk] = g.ndata[nk]
    return buf_graph

# Probe ahead to get the valid spliting point
def probe_division(g,div):
    for i in range(div,-1,-1):
        ret = (g.edata[dgl.EID]==i).int().argmax()
        if ret:
            return ret

# TODO: Define the dataloader
class DictNode:
    def __init__(self,parent,NIDdict=None):
        self.parent = parent
        if NIDdict != None:
            self.NIDdict = NIDdict.numpy()
        else:
            self.NIDdict = None 
        if (parent!=None and NIDdict==None) or (parent == None and NIDdict!=None):
            raise ValueError("Parent and Dict Unmatched")
            
        #self.child = None

    def GetRootID(self,index):
        # To enable batch process use numpy array
        if self.parent==None:
            return index
        map_index = self.NIDdict[index]
        return self.parent.GetRootID(map_index)

class TemporalDataLoader:
    def __init__(self,g,batch_size,n_neighbors,sampling_method='topk'):
        self.g  = g
        self.bg = dgl.add_reverse_edges(self.g,copy_edata=True)
        #print("Init",self.bg.in_degrees(0))
        self.d_g= DictNode(parent=None)
        # TODO: Implement Training, Test Validation division as well as unseen nodes
        # Assume graph edge id follow chronological order.
        num_edges = g.num_edges()
        #num_nodes = g.num_nodes()
        train_div = int(0.7*num_edges)
        valid_div = int(0.85*num_edges)

        self.nn_test_g  = dgl.edge_subgraph(self.g,range(valid_div,num_edges))
        self.nn_test_g = temporal_sort(self.nn_test_g,'timestamp')

        self.d_nn_test_g= DictNode(parent=self.d_g,NIDdict=self.nn_test_g.ndata[dgl.NID])
        # New Node random non-repeated choice
        np.random.seed(2020)
        # the new node is chosen from test graph and applied to validation and training set
        test_nodes = self.nn_test_g.ndata[dgl.NID].numpy()
        #print(test_nodes.shape)
        n_test_nodes = len(test_nodes)
        # We Need to get node subgraph from previous training graph
        #print(int(0.1*n_test_nodes))
        new_node_idxs = np.random.choice(test_nodes,replace=False,size=int(0.1*n_test_nodes))
        non_new_node_idxs = np.delete(self.g.nodes().numpy(),new_node_idxs)
        #print(new_node_idxs)
        # Need to get the training and validation subgraph
        self.nn_mask_g   = dgl.node_subgraph(self.g,non_new_node_idxs)
        # Sort graph for order keeping
        self.nn_mask_g = temporal_sort(self.nn_mask_g,'timestamp')
    
        self.nn_mask_bg  = dgl.add_reverse_edges(self.nn_mask_g,copy_edata=True)
        
        # 
        self.d_nn_mask_g = DictNode(self.d_g,self.nn_mask_g.ndata[dgl.NID])
        #self.parentn_id = self.nn_mask_g.ndata[dgl.NID].clone()
        #self.parente_id = self.nn_mask_g.edata[dgl.EID].clone()
        # Divide the training validation and test subgraph
        # Remapping
        train_div = probe_division(self.nn_mask_g,train_div)
        # edge might be removed by masking Then the idea will be condition by time.
        valid_div = probe_division(self.nn_mask_g,valid_div)
        print("Validation div: ",valid_div)
        self.train_g = dgl.edge_subgraph(self.nn_mask_g,range(0,train_div))
        self.train_g = temporal_sort(self.train_g,'timestamp')
        self.d_train_g = DictNode(self.d_nn_mask_g,self.train_g.ndata[dgl.NID])
        self.valid_g = dgl.edge_subgraph(self.nn_mask_g,range(train_div,valid_div))
        self.valid_g = temporal_sort(self.valid_g,'timestamp')
        self.d_valid_g = DictNode(self.d_nn_mask_g,self.valid_g.ndata[dgl.NID])
        self.test_g  = dgl.edge_subgraph(self.nn_mask_g,range(valid_div,self.nn_mask_g.num_edges()))
        self.test_g  = temporal_sort(self.test_g,'timestamp')
        self.d_test_g  = DictNode(self.d_nn_mask_g,self.test_g.ndata[dgl.NID])
        self.graph_dict = {'train':self.train_g,'valid':self.valid_g,'nn_test':self.nn_test_g,'test':self.test_g}
        self.dict_dict  = {'train':self.d_train_g,'valid':self.d_valid_g,'nn_test':self.d_nn_test_g,'test':self.d_test_g}
        self.batch_size = batch_size
        self.batch_cnt = 0
        #self.max_cnt = self.g.num_edges()//self.batch_size
        if sampling_method == 'topk':
            self.sampler = partial(dgl.sampling.select_topk,k=n_neighbors,weight='timestamp')
        else:
            self.sampler = partial(dgl.sampling.sample_neighbors,fanout=n_neighbors)
    
    def get_next_batch(self,mode='train'):
        '''
        Return the edge index list
        Here batch is the minimum unit for memory update
        The gradient update should use hyperbatch
        Since graph here is already subgraph. No need offset
        Maybe need negative sampling mechanism? Only on edge index domain
        '''
        graph = self.graph_dict[mode]
        p_dict= self.dict_dict[mode]
        done = False
        #print(self.g.edges()[1].shape)
        src_list = graph.edges()[0][self.batch_cnt*self.batch_size:min((self.batch_cnt+1)*self.batch_size,graph.num_edges())]
        #print("Direct Indexing",graph.edges()[0])
        dst_list = graph.edges()[1][self.batch_cnt*self.batch_size:min((self.batch_cnt+1)*self.batch_size,graph.num_edges())]
        #print(dst_list)
        t_stamps = graph.edata['timestamp'][self.batch_cnt*self.batch_size:min((self.batch_cnt+1)*self.batch_size,graph.num_edges())]
        # The index is problematic.
        edge_ids = range(self.batch_cnt*self.batch_size,min((self.batch_cnt+1)*self.batch_size,graph.num_edges()))
        subgraph = dgl.edge_subgraph(graph,edge_ids)
        working_d = DictNode(parent=p_dict,NIDdict=subgraph.ndata[dgl.NID])
        subgraph.ndata[dgl.NID] = torch.from_numpy(working_d.GetRootID(range(subgraph.num_nodes())))
        self.batch_cnt += 1
        if subgraph.num_edges() < self.batch_size:
            self.batch_cnt = 0
            done = True
        return done, src_list,dst_list,t_stamps,subgraph

    # Temporal Sampling Module
    def get_nodes_affiliation(self,nodes,timestamp,mode='train'):
        if type(nodes) != list:
            nodes = [int(nodes)]
        else:
            nodes = [int(nodes[0]),int(nodes[1])]
        origins  = self.dict_dict[mode].GetRootID(nodes).tolist()
        if mode == 'nn_test':
            frontier = origins
            graph = self.bg
            dict_ = self.d_g 
        else:
            frontier = self.graph_dict[mode].ndata[dgl.NID][nodes].tolist()
            graph = self.nn_mask_bg
            dict_ = self.d_nn_mask_g
        
        # All Neighbor regardless of time
        full_neighbor_subgraph = dgl.in_subgraph(graph,frontier)
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
                                               torch.tensor(frontier),
                                               torch.tensor(frontier))
        
        # Temporal sampling
        temporal_edge_mask = full_neighbor_subgraph.edata['timestamp'] < timestamp
        temporal_subgraph = dgl.edge_subgraph(full_neighbor_subgraph,temporal_edge_mask)

        # Build a working dict for parent node backtrace
        working_dict = DictNode(dict_, temporal_subgraph.ndata[dgl.NID])
        root_nid = working_dict.GetRootID(range(temporal_subgraph.num_nodes()))
        temporal_subgraph.ndata[dgl.NID] = torch.from_numpy(root_nid)

        # Update the frontier to current graph
        node1_mask = temporal_subgraph.ndata[dgl.NID]==origins[0]
        frontier = [int(torch.arange(0,temporal_subgraph.num_nodes())[node1_mask])]
        if len(nodes)!=1:
            node2_mask = temporal_subgraph.ndata[dgl.NID]==origins[1]
            frontier.append(int(torch.arange(0,temporal_subgraph.num_nodes())[node2_mask]))
        
        # Top k or random sample from subgraph to reduce complexity
        final_subgraph = self.sampler(g=temporal_subgraph,nodes=frontier)
        final_subgraph = dgl.remove_self_loop(final_subgraph)
        final_subgraph = dgl.add_self_loop(final_subgraph)

        return frontier, final_subgraph

    def reset(self):
        self.batch_cnt = 0
        