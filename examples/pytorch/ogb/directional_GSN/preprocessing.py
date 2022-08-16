from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
import torch
import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.topology as gt_topology
from torch.utils.data import Dataset
import types
import time
from tqdm import tqdm
import dgl
import os

def to_undirected(edge_index):
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index

def automorphism_orbits(edge_list):
    
    ##### vertex automorphism orbits ##### 
    graph = gt.Graph(directed=False)
    graph.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph)
    gt.stats.remove_parallel_edges(graph)

    # compute the vertex automorphism group
    aut_group = gt_topology.subgraph_isomorphism(graph, graph, induced=False, subgraph=True, generator=False)

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v
    
    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, vertex in enumerate(aut):
            role = min(original, orbit_membership[vertex])
            orbit_membership[vertex] = role
        
    orbit_membership_list = [[],[]]
    for vertex, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(vertex)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(orbit_membership_list[1], return_inverse = True)

    orbit_membership = {vertex: contiguous_orbit_membership[i] for i,vertex in enumerate(orbit_membership_list[0])}

    orbit_partition = {}
    for vertex, orbit in orbit_membership.items():
        orbit_partition[orbit] = [vertex] if orbit not in orbit_partition else orbit_partition[orbit]+[vertex]
    
    aut_count = len(aut_group)

    return graph, orbit_partition, orbit_membership, aut_count

def induced_edge_automorphism_orbits(edge_list):
    
    ##### induced edge automorphism orbits (according to the vertex automorphism group) #####
    graph, orbit_partition, orbit_membership, aut_count = automorphism_orbits(edge_list=edge_list)
    edge_orbit_partition = dict()
    edge_orbit_membership = dict()
    edge_orbits2inds = dict()
    ind = 0
    
    edge_list = to_undirected(torch.tensor(graph.get_edges()).transpose(1,0)).transpose(1,0).tolist()

    # infer edge automorphisms from the vertex automorphisms
    for i,edge in enumerate(edge_list):
        edge_orbit = frozenset([orbit_membership[edge[0]], orbit_membership[edge[1]]])
        if edge_orbit not in edge_orbits2inds:
            edge_orbits2inds[edge_orbit] = ind
            ind_edge_orbit = ind
            ind += 1
        else:
            ind_edge_orbit = edge_orbits2inds[edge_orbit]

        if ind_edge_orbit not in edge_orbit_partition:
            edge_orbit_partition[ind_edge_orbit] = [tuple(edge)]
        else:
            edge_orbit_partition[ind_edge_orbit] += [tuple(edge)] 

        edge_orbit_membership[i] = ind_edge_orbit

    print('Edge orbit partition of given substructure: {}'.format(edge_orbit_partition)) 
    print('Number of edge orbits: {}'.format(len(edge_orbit_partition)))
    print('Graph (vertex) automorphism count: {}'.format(aut_count))
    
    return graph, edge_orbit_partition, edge_orbit_membership, aut_count

def subgraph_isomorphism_edge_counts(edge_index, subgraph_dict):
    
    ##### edge structural identifiers #####
    
    edge_index = edge_index.transpose(1,0).cpu().numpy()
    edge_dict = {}
    for i, edge in enumerate(edge_index):         
        edge_dict[tuple(edge)] = i
        
    subgraph_edges = to_undirected(torch.tensor(subgraph_dict['subgraph'].get_edges().tolist()).transpose(1,0)).transpose(1,0).tolist()

    G_gt = gt.Graph(directed=False)
    G_gt.add_edge_list(list(edge_index))
    gt.stats.remove_self_loops(G_gt)
    gt.stats.remove_parallel_edges(G_gt)  
       
    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(subgraph_dict['subgraph'], G_gt, induced=True, subgraph=True, generator=True)
    
    counts = np.zeros((edge_index.shape[0], len(subgraph_dict['orbit_partition'])))
    
    for sub_iso_curr in sub_iso:
        mapping = sub_iso_curr.get_array()
        for i,edge in enumerate(subgraph_edges): 
            
            # for every edge in the graph H, find the edge in the subgraph G_S to which it is mapped
            # (by finding where its endpoints are matched). 
            # Then, increase the count of the matched edge w.r.t. the corresponding orbit
            # Repeat for the reverse edge (the one with the opposite direction)
            
            edge_orbit = subgraph_dict['orbit_membership'][i]
            mapped_edge = tuple([mapping[edge[0]], mapping[edge[1]]])
            counts[edge_dict[mapped_edge], edge_orbit] += 1
            
    counts = counts/subgraph_dict['aut_count']
    
    counts = torch.tensor(counts)
    
    return counts

def prepare_dataset(path, name, **subgraph_params):
    
    k = 8

    data_folder = os.path.join(path, 'processed')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    data_file = os.path.join(data_folder, 'cycle_graph_induced_{}.pt'.format(k))
        
    graphs_dgl, orbit_partition_sizes, split_idx = generate_dataset(path, name, **subgraph_params)
    if data_file is not None:
        print("Saving dataset to {}".format(data_file))
        torch.save((graphs_dgl, orbit_partition_sizes, split_idx), data_file)

    return graphs_dgl, split_idx

def generate_dataset(path, name):

    ### compute the orbits of earch substructure in the list, as well as the vertex automorphism count
    subgraph_dicts = []
    orbit_partition_sizes = []

    edge_lists = []
    for k in range(3, 8 + 1):
        graphs_nx = getattr(nx, 'cycle_graph')(k)
        if isinstance(graphs_nx, list) or isinstance(graphs_nx, types.GeneratorType):
            edge_lists += [list(graph_nx.edges) for graph_nx in graphs_nx]
        else:
            edge_lists.append(list(graphs_nx.edges))

    for edge_list in edge_lists:
        subgraph, orbit_partition, orbit_membership, aut_count = induced_edge_automorphism_orbits(edge_list=edge_list)
        subgraph_dicts.append({'subgraph':subgraph, 'orbit_partition': orbit_partition, 
                               'orbit_membership': orbit_membership, 'aut_count': aut_count})
        orbit_partition_sizes.append(len(orbit_partition))
        
    ### load and preprocess dataset
    dataset = DglGraphPropPredDataset(name=name, root=path)
    split_idx = dataset.get_idx_split()
        
    # computation of subgraph isomoprhisms & creation of data structure
    graphs_dgl = list()
    for i, datapoint in tqdm(enumerate(dataset)):
        g, label = datapoint
        g = _prepare(g, subgraph_dicts)
        graphs_dgl.append((g, label))

    return graphs_dgl, orbit_partition_sizes, split_idx
        
def _prepare(g, subgraph_dicts):

    edge_index = torch.stack(g.edges())
    
    identifiers = None
    for subgraph_dict in subgraph_dicts:
        counts = subgraph_isomorphism_edge_counts(edge_index, subgraph_dict)
        identifiers = counts if identifiers is None else torch.cat((identifiers, counts),1) 

    g.edata['subgraph_counts'] = identifiers.long()
    
    return g

class pcbaDGL(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.split = split
        
        self.data = [data[split_ind] for split_ind in self.split]
        
        self.graph_lists = []
        self.graph_labels = []
        for g in self.data:
            if g[0].number_of_nodes() > 5:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)
        
        new_graph_lists = []
        for g in self.graph_lists:
            g = self.get_subgraphs(g)
            g.edata['eig'] = g.edata['eig'].float()
            new_graph_lists.append(g)
        self.graph_lists = new_graph_lists

    def get_subgraphs(self, g):
        vector_field = g.edata['subgraph_counts']
        g.edata['eig'] = vector_field
        return g

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]

class pcbaDataset(Dataset):
    def __init__(self,
                 name,
                 path='dataset/ogbg-molpcba',
                 verbose=True,
                 **subgraph_params):
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        
        self.dataset, self.split_idx = prepare_dataset(path, name, **subgraph_params)
        print("One hot encoding substructure counts... ", end='')
        self.d_id = [1]*self.dataset[0][0].edata['subgraph_counts'].shape[1]
            
        self.train = pcbaDGL(self.dataset, self.split_idx['train'])
        self.val = pcbaDGL(self.dataset, self.split_idx['valid'])
        self.test = pcbaDGL(self.dataset, self.split_idx['test'])
        self.evaluator = Evaluator(name='ogbg-molpcba')

        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)

        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e

if __name__ == '__main__':
    path = os.path.join('./', 'dataset', "ogbg-molpcba")
    dataset = pcbaDataset("ogbg-molpcba", path=path, verbose=True)