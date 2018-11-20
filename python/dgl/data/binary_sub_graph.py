import dgl
from ..batched_graph import batch
from ..graph import DGLGraph
from ..utils import Index

import numpy as np
import networkx as nx
import itertools

class CORABinary:
    """
    Simple binary subgraph constrcutor for CORA.
    Should be able to be used to extract binary 
    subgraph from other graph dataset.
    
    Parameters
    ----------
    
    g : DGLGraph object
        the graph
    features : numpy ndarray
        features of nodes
    labels : numpy ndarray
        labels of nodes
    num_classes : int
        number of classes at interest
    """
    
    def __init__(self, g, features, labels, num_classes):
        self._g = g
        self._features = features
        self._labels = labels
        self._n_classes = num_classes
        self._community_to_node = [[] for i in range(max(labels)+1)]
        for node in range(len(labels)):
            self._community_to_node[labels[node]].append(node)
        self._g_nx = g.to_networkx()
        self._subgraphs = []
        self._subfeatures = []
        self._sublabels = []
        
        for i,j in itertools.combinations([i for i in range(num_classes)],2):
            subg = g.subgraph([0,1,2])
            print(type(subg))
            subgraph, new2oldindex = self.binary_subgraph_cora(self._g_nx, 
                                                               self._community_to_node, 
                                                               i, 
                                                               j)
            subfeature = self._features[list(subgraph.nodes()),:]
            sublabel = self._labels[list(new2oldindex[i] 
                                   for i in subgraph.nodes())]
            #relabel the subgraph
            sublabel[sublabel<=i] = 0
            sublabel[sublabel>i] = 1
            self._subgraphs.append(DGLGraph(subgraph))
            #/TODO: directly build DGL subgraph
            self._subfeatures.append(subfeature)
            self._sublabels.append(sublabel)
        
        
        #Contruct line graph
        
        self._line_graphs = [g.line_graph(backtracking=False) 
                             for g in self._subgraphs]
        in_degrees = lambda g: g.in_degrees(
                             Index(np.arange(0, g.number_of_nodes()))).unsqueeze(1).float()
        #TODO : replace th.arange to F.arange when migrating back to /data
        
        self._g_degs = [in_degrees(g) for g in self._subgraphs]
        self._lg_degs = [in_degrees(lg) for lg in self._line_graphs]
        self._pm_pds = list(zip(*[g.edges() for g in self._subgraphs]))[0]
        
        self._equi_labels = []
        for label in self._sublabels:
            mirror_label = np.ones(label.shape) - label
            self._equi_labels.append(mirror_label)
    
    def binary_subgraph_cora(self,
                             original_graph, 
                             community_to_node, 
                             classA=0, 
                             classB=1):
        sub = nx.DiGraph(original_graph.subgraph(community_to_node[classA]+
                                                 community_to_node[classB]))
        #cast to undirected graph to find connected component
        sub_und = nx.Graph(sub)
        candidate_edge_list = []
        for edge in sub.edges():
            # Only look at src node of edges that cross communities
            if (self._labels[edge[0]] != self._labels[edge[1]]):
                candidate_edge_list.append(edge)
        component_list = []
        for edge in candidate_edge_list:
            component_list.append(
                nx.node_connected_component(sub_und,edge[0]))
        component_list.sort(key=len, reverse=True)
        
        # find the largest connected component to be the candidate subgraph
        largest_subgraph = component_list[0]
        old2newindex = {}
        new2oldindex = {}
        reindex_nodes = [i for i in range(len(largest_subgraph))]
        reindex_edges = []
        for i, node in enumerate(largest_subgraph):
            old2newindex[node] = i
            new2oldindex[i] = node
        for src,dst in sub.edges(largest_subgraph):
            reindex_edges.append((old2newindex[src], old2newindex[dst]))

        #use old2newindex to manually reindex the subgraph
        #/TODO(hq): send nodelist directly to DGLgraph
        reindex_graph = nx.DiGraph()
        reindex_graph.add_nodes_from(reindex_nodes)
        reindex_graph.add_edges_from(reindex_edges)


        return reindex_graph, new2oldindex
    
    def __len__(self):
        return len(self._subgraphs)
    
    def __getitem__(self, idx):
        return self._subgraphs[idx], self._line_graphs[idx], self._g_degs[idx], self._lg_degs[idx], self._pm_pds[idx], self._subfeatures[idx], self._sublabels[idx], self._equi_labels[idx]
    
    def collate_fn(self, x):
        subgraph, line_graph, deg_g, deg_lg, pm_pd, subfeature, sublabel, equi_label = zip(*x)
        subgraph_batch = batch(subgraph)
        line_graph_batch = batch(line_graph)
        # TODO : change to F.cat when migrating to /data
        deg_g_batch = np.concatenate(deg_g, axis=0)
        deg_lg_batch = np.concatenate(deg_lg, axis=0)
        
        self.total = 0
        def offset(pm_pd):
            prev_total = self.total
            self.total += pm_pd.size(0)
            return prev_total
        pm_pd_batch = np.concatenate([x + offset(x) for i, x in enumerate(pm_pd)], 
                             axis=0)
        
        subfeature_batch = np.concatenate(subfeature, axis=0)
        sublabel_batch = np.concatenate(sublabel, axis=0)
        equilabel_batch = np.concatenate(equi_label, axis=0)
        
        return subgraph_batch, line_graph_batch, deg_g_batch, deg_lg_batch, pm_pd_batch,subfeature_batch, sublabel_batch, equilabel_batch    

