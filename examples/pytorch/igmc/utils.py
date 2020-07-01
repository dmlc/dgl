import csv
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()

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

def subgraph_extraction_labeling(g_label, ind, adj, 
                                 hop=1, sample_ratio=1.0, max_node_label=3, max_nodes_per_hop=200):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

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
    # NOTE: RelGraphConv count relation from 0??
    bi_r = np.concatenate([r, r]) - 1 
    subgraph_info = {'g_label': g_label, 'src': src, 'dst': dst, 'etype': bi_r}

    # get structural node labels
    # NOTE: only use subgraph here
    u_node_labels = [x*2 for x in u_dist]
    v_node_labels = [x*2+1 for x in v_dist]
    u_x = one_hot(u_node_labels, max_node_label+1)
    v_x = one_hot(v_node_labels, max_node_label+1)

    subgraph_info['x'] = np.concatenate([u_x, v_x], axis=0)
    
    return subgraph_info

def neighbors(fringe, adj, row=True):
    # TODO [zhoujf] use sample_neighbors fn
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
