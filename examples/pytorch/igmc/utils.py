import os
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import torch as th
import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

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

def links2subgraphs(
        adj, # label_values, pool,
        train_indices, val_indices, test_indices, 
        train_labels, val_labels, test_labels, 
        hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=3, 
        testing=False, parallel=True):

    def helper(adj, links, g_labels):
        g_list = []
        if not parallel: # or max_node_label is None:
            with tqdm(total=len(links[0])) as pbar:
                for u, v, g_label in zip(links[0], links[1], g_labels):
                    g = subgraph_extraction_labeling(
                        g_label, (u, v), adj, 
                        hop, sample_ratio, max_node_label, max_nodes_per_hop)
                    g_list.append(g) 
                    pbar.update(1)
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(subgraph_extraction_labeling, 
                                        [(g_label, (u, v), adj, 
                                          hop, sample_ratio, max_node_label, max_nodes_per_hop) 
                                          for u, v, g_label in zip(links[0], links[1], g_labels)])
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            g_list += results.get()
            pool.close()
            pool.join()
            pbar.close()
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(adj, train_indices, train_labels)
    val_graphs = helper(adj, val_indices, val_labels) if not testing else []
    test_graphs = helper(adj, test_indices, test_labels)
    return train_graphs, val_graphs, test_graphs

# import networkx as nx
# from refex import RecursiveExtractor
def subgraph_extraction_labeling(g_label, ind, adj, 
                                 hop=1, sample_ratio=1.0, max_node_label=3, max_nodes_per_hop=200):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    # g
    # u, v = edge
    # nodes = th.array([u, v])
    # hop1 = g.in_subgraph(nodes)  # nodes: shape (N,)
    # src, _ = hop1.edges()
    # hop2 = g.in_subgraph(th.unique(src))
    # src2, _ = hop2.edges()
    # all_nodes = th.unique(th.cat([src, src2]))
    # sg = g.subgraph(all_nodes)

    for dist in range(1, hop+1):
        v_fringe, u_fringe = neighbors(u_fringe, adj, True), neighbors(v_fringe, adj, False)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = adj[u_nodes, :][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0

    # reindex u and v, v nodes start after u
    u, v, r = sp.find(subgraph)
    v += len(u_nodes)
    r = r.astype(int)

    # Add bidirection link
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    # RelGraphConv count relation from 0
    bi_r = np.concatenate([r, r]) - 1 
    subgraph_info = {'g_label': g_label, 'src': src, 'dst': dst, 'etype': bi_r}

    # get structural node labels
    # only use subgraph here
    u_node_labels = [x*2 for x in u_dist]
    v_node_labels = [x*2+1 for x in v_dist]
    u_x = one_hot(u_node_labels, max_node_label+1)
    v_x = one_hot(v_node_labels, max_node_label+1)

    subgraph_info['x'] = np.concatenate([u_x, v_x], axis=0)
    
    # nx_graph = nx.from_edgelist(list(zip(*(subgraph_info['src'], subgraph_info['dst']))))
    # recurser = RecursiveExtractor(nx_graph, n_recursion=1, pruning_cutoff=0.5, bins=4)
    # subgraph_info['refex_feature'] = recurser.create_features()

    return subgraph_info

def neighbors(fringe, adj, row=True):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        if row:
            _, nei, _ = sp.find(adj[node, :])
        else:
            nei, _, _ = sp.find(adj[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x

def dgl_subgraph_extraction_labeling(ind, graph, 
                                 hop=1, sample_ratio=1.0, max_node_label=3, max_nodes_per_hop=200):
    # extract the h-hop enclosing subgraph around link 'ind'
    u, v = ind 

    # 1. neighbor sampling
    dist = 0
    u_nodes, v_nodes = ind[0].unsqueeze(dim=0), ind[1].unsqueeze(dim=0)
    u_dist, v_dist = th.tensor([0]), th.tensor([0])
    u_visited, v_visited = th.unique(u_nodes), th.unique(v_nodes)
    u_fringe, v_fringe = th.unique(u_nodes), th.unique(v_nodes)

    for dist in range(1, hop+1):
        # sample neigh separately as the dist(label) of u nodes is different from v nodes
        u_fringe, v_fringe = graph.in_edges(v_fringe)[0], graph.out_edges(u_fringe)[1]
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
                u_fringe = u_fringe[shuffled_idx[: max_nodes_per_hop]]
            if max_nodes_per_hop < len(v_fringe):
                shuffled_idx = th.randperm(len(u_fringe))
                v_fringe = v_fringe[shuffled_idx[: max_nodes_per_hop]]
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = th.cat([u_nodes, u_fringe])
        v_nodes = th.cat([v_nodes, v_fringe])
        u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
        v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])
    
    # 2. subgrpah creation
    subgraph = train_graph.subgraph(th.cat([u_nodes, v_nodes]))

    su = subgraph.nodes()[subgraph.ndata[dgl.NID]==u]
    sv = subgraph.nodes()[subgraph.ndata[dgl.NID]==v]
    target_edges = subgraph.edge_ids([su, sv], [sv, su])
    # set edge weight 0 as to remove link between target nodes
    subgraph.edata['weight'] = th.ones(subgraph.number_of_edges())
    subgraph.edata['weight'][target_edges] = 0.
    
    # 3. node labeling
    u_node_labels = th.stack([x*2 for x in u_dist])
    v_node_labels = th.stack([x*2+1 for x in v_dist])
    
    u_x = dgl_one_hot(u_node_labels, max_node_label+1)
    v_x = dgl_one_hot(v_node_labels, max_node_label+1)
    subgraph.ndata['x'] = th.cat([u_x, v_x])

    return subgraph

def dgl_one_hot(idx, length):
    x = th.zeros([len(idx), length])
    x[th.arange(len(idx)), idx] = 1.0
    return x