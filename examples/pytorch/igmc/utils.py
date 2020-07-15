import os
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import torch as th
import dgl 

class MetricLogger(object):
    def __init__(self, save_dir, log_interval):
        self.save_dir = save_dir
        self.log_interval = log_interval

    def log(self, info, model, optimizer):
        epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
                epoch, train_loss, test_rmse))
        if type(epoch) == int and epoch % self.log_interval == 0:
            print('Saving model states...')
            model_name = os.path.join(self.save_dir, 'model_checkpoint{}.pth'.format(epoch))
            optimizer_name = os.path.join(
                self.save_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
            )
            if model is not None:
                th.save(model.state_dict(), model_name)
            if optimizer is not None:
                th.save(optimizer.state_dict(), optimizer_name)

def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])

def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str

# igmc: sample nodes based on target edge 
def get_neighbor_nodes_labels(ind, graph, bipartite=True,
        hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
    
    if bipartite:
        # 1. neighbor nodes sampling
        dist = 0
        u_nodes, v_nodes = ind[0].unsqueeze(dim=0), ind[1].unsqueeze(dim=0)
        u_dist, v_dist = th.tensor([0]), th.tensor([0])
        u_visited, v_visited = th.unique(u_nodes), th.unique(v_nodes)
        u_fringe, v_fringe = th.unique(u_nodes), th.unique(v_nodes)

        for dist in range(1, hop+1):
            # sample neigh alternately
            u_fringe, v_fringe = graph.in_edges(v_fringe)[0], graph.in_edges(u_fringe)[0]
            u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
            v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
            u_visited = th.unique(th.cat([u_visited, u_fringe]))
            v_visited = th.unique(th.cat([v_visited, v_fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(u_fringe):
                    shuffled_idx = th.randperm(len(u_fringe))
                    u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
                if max_nodes_per_hop < len(v_fringe):
                    shuffled_idx = th.randperm(len(u_fringe))
                    v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = th.cat([u_nodes, u_fringe])
            v_nodes = th.cat([v_nodes, v_fringe])
            u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
            v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])
 
        nodes = th.cat([u_nodes, v_nodes])

        # 2. node labeling
        u_node_labels = th.stack([x*2 for x in u_dist])
        v_node_labels = th.stack([x*2+1 for x in v_dist])
        node_labels = th.cat([u_node_labels, v_node_labels])
    else: 
        # 1. neighbor nodes sampling
        dist = 0
        nodes = th.stack(ind)
        dists = nodes.zeros_like(nodes) 
        visited = th.unique(nodes)
        fringe = th.unique(nodes)

        for dist in range(1, hop+1):
            fringe = graph.in_edges(fringe)[0]    
            fringe = th.from_numpy(np.setdiffed(fringe.numpy(), visited.numpy()))
            visited = th.unique(th.cat([visited, fringe]))

            if sample_ratio < 1.0:
                shuffled_idx = th.randperm(len(fringe))
                fringe = fringe[shuffled_idx[:int(sample_ratio*len(fringe))]]
            if max_nodes_per_hop is not None and max_nodes_per_hop < len(fringe):
                shuffled_idx = th.randperm(len(fringe))
                fringe = fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(fringe) == 0:
                break
            nodes = th.cat([nodes, fringe])
            dists = th.cat([dists, th.full((len(fringe), ), dist, dtype=th.int64)])
        
        node_labels = dists
    return nodes, node_labels

def subgraph_extraction_labeling(ind, graph, bipartite=True, 
                                 hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
    # extract the h-hop enclosing subgraph around link 'ind'
    nodes, node_labels = get_neighbor_nodes_labels(ind, graph, bipartite, 
                                                   hop, sample_ratio, max_nodes_per_hop)

    subgraph = graph.subgraph(nodes)
    label_size = (hop+1) * 2 if bipartite else hop+1
    subgraph.ndata['x'] = one_hot(node_labels, label_size)
    
    # set edge weight to zero as to remove links between target nodes in training process
    subgraph.edata['weight'] = th.ones(subgraph.number_of_edges())
    su = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[0]]
    sv = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[1]]
    _, _, target_edges = subgraph.edge_ids([su, sv], [sv, su], return_uv=True)
    subgraph.edata['weight'][target_edges] = 0

    return subgraph

def one_hot(idx, length):
    x = th.zeros([len(idx), length])
    x[th.arange(len(idx)), idx] = 1.0
    return x
