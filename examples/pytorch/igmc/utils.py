import os
import random

import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

import numpy as np
import torch as th
import torch.nn.functional as F
import dgl 
import dgl.function as fn

class MetricLogger(object):
    def __init__(self, save_dir, log_interval):
        self.save_dir = save_dir
        self.log_interval = log_interval

    def log(self, info, model, optimizer):
        epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
                epoch, train_loss, test_rmse))
        # if type(epoch) == int and epoch % self.log_interval == 0:
        #     print('Saving model states...')
        #     model_name = os.path.join(self.save_dir, 'model_checkpoint{}.pth'.format(epoch))
        #     optimizer_name = os.path.join(
        #         self.save_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
        #     )
        #     if model is not None:
        #         th.save(model.state_dict(), model_name)
        #     if optimizer is not None:
        #         th.save(optimizer.state_dict(), optimizer_name)

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

def one_hot(idx, length):
    x = th.zeros([len(idx), length])
    x[th.arange(len(idx)), idx] = 1.0
    return x

def cal_dist(csr_graph, node_to_remove):
    # cal dist to node 0, with target edge nodes 0/1 removed
    nodes = list(set(range(csr_graph.shape[1])) - set([node_to_remove]))
    csr_graph = csr_graph[nodes, :][:, nodes]
    dists = np.clip(sp.csgraph.dijkstra(
                        csr_graph, indices=0, directed=False, unweighted=True, limit=1e6
                    )[1:], 0, 1e7)
    return dists.astype(np.int64)

def get_neighbor_nodes_labels(ind, graph, mode="bipartite",
        hop=1, sample_ratio=1.0, max_nodes_per_hop=200):
    
    if mode=="bipartite":
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
                    shuffled_idx = th.randperm(len(v_fringe))
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
    
    elif mode=="homo": 
        # 1. neighbor nodes sampling
        dist = 0
        nodes = th.stack(ind)
        dists = th.zeros_like(nodes) 
        visited = th.unique(nodes)
        fringe = th.unique(nodes)

        for dist in range(1, hop+1):
            fringe = graph.in_edges(fringe)[0]    
            fringe = th.from_numpy(np.setdiff1d(fringe.numpy(), visited.numpy()))
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
        
        # 2. node labeling
        node_labels = dists
    
    elif mode=="grail":
        # 1. neighbor nodes sampling
        # make sure ind not in uv nodes.
        u_nodes, v_nodes = th.tensor([], dtype=th.long), th.tensor([], dtype=th.long)
        # u_dist, v_dist = th.tensor([0]), th.tensor([0])
        u_visited, v_visited = th.tensor([ind[0]]), th.tensor([ind[1]])
        u_fringe, v_fringe = th.tensor([ind[0]]), th.tensor([ind[1]])

        for dist in range(1, hop+1):
            # sample neigh separately
            u_fringe = graph.in_edges(u_fringe)[0]
            v_fringe = graph.in_edges(v_fringe)[0]

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
                    shuffled_idx = th.randperm(len(v_fringe))
                    v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
            if len(u_fringe) == 0 and len(v_fringe) == 0:
                break
            u_nodes = th.cat([u_nodes, u_fringe])
            v_nodes = th.cat([v_nodes, v_fringe])
            # u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
            # v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])
    
        nodes = th.from_numpy(np.intersect1d(u_nodes.numpy(), v_nodes.numpy()))
        # concatenate ind to front, and node labels of ind can be added easily.
        nodes = th.cat([ind, nodes])
       
        # 2. node labeling
        csr_subgraph = graph.subgraph(nodes).adjacency_matrix_scipy(return_edge_ids=False)
        dists = th.stack([th.tensor(cal_dist(csr_subgraph, 1)), 
                          th.tensor(cal_dist(csr_subgraph, 0))], axis=1)
        ind_labels = th.tensor([[0, 1], [1, 0]])
        node_labels = th.cat([ind_labels, dists]) if dists.size() else ind_labels

        # 3. prune nodes that are at a distance greater than hop from neigh of the target nodes
        pruned_mask = th.max(node_labels, axis=1)[0] <= hop
        nodes, node_labels = nodes[pruned_mask], node_labels[pruned_mask]
    else:
        raise NotImplementedError
    return nodes, node_labels

# @profile
def subgraph_extraction_labeling(ind, graph, mode="bipartite", 
                                 hop=1, sample_ratio=1.0, max_nodes_per_hop=200):

    # extract the h-hop enclosing subgraph nodes around link 'ind'
    nodes, node_labels = get_neighbor_nodes_labels(ind, graph, mode, 
                                                   hop, sample_ratio, max_nodes_per_hop)

    subgraph = graph.subgraph(nodes)

    if mode == "bipartite":
        subgraph.ndata['x'] = one_hot(node_labels, (hop+1)*2)
    elif mode == "homo":
        subgraph.ndata['x'] = one_hot(node_labels, hop+1)
    elif mode == "grail":
        subgraph.ndata['x'] = th.cat([one_hot(node_labels[:, 0], hop+1), 
                                    one_hot(node_labels[:, 1], hop+1)], dim=1)
    else:
        raise NotImplementedError
    # subgraph.ndata['x'] = th.cat([subgraph.ndata['x'], subgraph.ndata['refex']], dim=1)

    # refex_feature = extract_refex_feature(subgraph).to(th.float)
    # subgraph.ndata['x'] = th.cat([subgraph.ndata['x'], refex_feature], dim=1)

    # set edge weight to zero as to remove links between target nodes in training process
    subgraph.edata['weight'] = th.ones(subgraph.number_of_edges())
    su = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[0]]
    sv = subgraph.nodes()[subgraph.ndata[dgl.NID]==ind[1]]
    _, _, target_edges = subgraph.edge_ids([su, sv], [sv, su], return_uv=True)
    subgraph.edata['weight'][target_edges] = 0

    return subgraph

def MinMaxScaling(x, dim=0):
    dist = x.max(dim=dim, keepdim=True)[0] - x.min(dim=dim, keepdim=True)[0]
    x = (x - x.min(dim=dim, keepdim=True)[0]) / (dist + 1e-7)
    return x

def get_recursive_feature(graph, basic_feature, n_iter=1):
    with graph.local_scope():

        init_feature = basic_feature
        recursive_feature = []
        for iter_idx in range(n_iter):
            graph.srcdata['h'] = init_feature
            graph.update_all(fn.copy_u('h', 'msg'), fn.mean('msg', 'neigh_mean'))
            graph.update_all(fn.copy_u('h', 'msg'), fn.sum('msg', 'neigh_sum'))

            init_feature = th.cat([graph.dstdata['neigh_mean'], 
                                   graph.dstdata['neigh_sum']], dim=1)
            # there is no feature pruning at the moment
            recursive_feature.append(init_feature)
        
        return th.cat(recursive_feature, dim=1)

def get_ego_feature(graph):
    ego_feature = []
    for node in graph.nodes():
        neighs = graph.in_edges(node)[0]
        nodes = th.cat([node.view(1), neighs])

        internal_degree = th.tensor(graph.subgraph(nodes).number_of_edges(), dtype=th.float32)
        overall_degree = th.sum(graph.in_degrees(nodes), dtype=th.float32)
        # there 32 isolated nodes in train graph
        external_degree = overall_degree - internal_degree
        if overall_degree == 0:
            overall_degree = th.tensor(float('inf'))
            external_degree = th.tensor(0.)
        ego_feature.append(th.cat([internal_degree.view(1), 
                                   external_degree.view(1), 
                                   (internal_degree/overall_degree).view(1),
                                   (external_degree/overall_degree).view(1)]))
    return th.stack(ego_feature)

def get_local_feature(graph):
    degree = graph.in_degrees().to(th.float32)
    return degree.unsqueeze(1)

def get_basic_feature(graph, normalize=True):
    local_feature = get_local_feature(graph)
    ego_feature = get_ego_feature(graph)
    basic_feature = th.cat([local_feature, ego_feature], dim=1)
    if normalize:
        basic_feature = MinMaxScaling(basic_feature, dim=0)
    return basic_feature

def extract_refex_feature(graph):
    basic_feature = get_basic_feature(graph, normalize=True)
    recursive_feature = get_recursive_feature(graph, basic_feature, n_iter=1)
    return th.cat([basic_feature, recursive_feature], dim=1)

if __name__ == "__main__":
    import time
    from data import MovieLens
    movielens = MovieLens("ml-100k", testing=True)

    train_edges = movielens.train_rating_pairs
    train_graph = movielens.train_graph

    idx = 0
    u, v = train_edges[0][idx], train_edges[1][idx]
    subgraph = subgraph_extraction_labeling(
                    (u, v), train_graph, 
                    hop=1, sample_ratio=1.0, max_nodes_per_hop=200)
    # t_start = time.time()
    # refex_feature = extract_refex_feature(train_graph).to(th.float)
    # print("Epoch time={:.2f}".format(time.time()-t_start))
    pass
