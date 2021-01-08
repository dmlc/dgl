import numpy as np
import torch
import networkx as nx

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


# TODO: Implement All pair Shortest path generation mechanism using networkx
# TODO: For each node rank the of path of same length by cost and select top k
# TODO: Set the node two features to be of two lengths all node use it as predecessor 

# Assume in the graph the attention weight has been assigned
# Only return shortest path of a given length
def shortest_path(dgl_graph,path_len,r=1):
    '''
    Args:
    dgl_graph: <DGLGraph> input graph 
    path_len : <int> desired path hop length of all shortest paths
    r        : <int> top in_degree*r paths which has lowest cost

    Return:
    apsp     : <dict<dict>> A concatenated dictionary:
               apsp[center] : all selected paths of center node
               apsp[center][endnode]: Path from 
    '''
    g_nx = dgl_graph.to_networkx(edge_attrs=['e'])
    N = dgl_graph.num_nodes()
    # Index by center node
    apsp = {}
    for i in range(N):
        paths_single_node = nx.shortest_path(g_nx,source=i)
        costs_single_node = nx.shortes_path_length(g_nx,source=i)
        paths_selected = {}
        # From paper, we need to implement doubled self loop
        paths_selected[i] = [i]*path_len
        k = int(r*g_nx.in_degree(i))
        # Screen the path acoording to length
        for key in paths_single_node.keys():
            path = paths_single_node[key]
            cost = costs_single_node[key]
            if len(path) == path_len and key!=i:
                paths_selected[key] = (path,cost)
        # Ordered top paths, sorted by costs
        topk_paths = dict(list(sorted(paths_selected.items(),key=lambda x: paths_selected[x[0]][1])[:k]))
        apsp[i] = topk_paths
    return apsp

# This is a hard task
def graph_decomposition(dgl_graph,path_lens,apsps):
    '''
    Args:
    dgl_graph: <DGLGraph> input graph
    path_lens: <list<int>> A list of all path lengths in apsps
    apsps    : <dict<dict<dict>>> A dictionary of all pair shortest 
                                  path with all hop length
    
    Return:
    graphs   : <list<DGLGraph>> A list of all decomposed graph/multigraphs 
                                which can perform parallel attention computation 
    '''

    for len in path_lens:
        
    pass
        


        


# TODO: Provide a decomposition method which can decompose the entire graph into several multigraphs by intermediate as well as several multihop graph
