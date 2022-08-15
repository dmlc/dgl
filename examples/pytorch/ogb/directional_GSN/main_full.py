from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
import torch
import numpy as np
import networkx as nx
import graph_tool as gt
import graph_tool.topology as gt_topology
from torch.utils.data import Dataset, DataLoader
import types
import time
from tqdm import tqdm
import glob
import re
import dgl
import os
import random
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn.functional as F
from functools import partial
import torch.optim as optim
import argparse

def aggregate_mean(h, vector_field, h_in):
    return torch.mean(h, dim=1)

def aggregate_max(h, vector_field, h_in):
    return torch.max(h, dim=1)[0]

def aggregate_sum(h, vector_field, h_in):
    return torch.sum(h, dim=1)

def aggregate_dir_dx(h, vector_field, h_in, eig_idx):
    eig_w = ((vector_field[:, :, eig_idx]) /
             (torch.sum(torch.abs(vector_field[:, :, eig_idx]), keepdim=True, dim=1) + 1e-8)).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)

AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max,
               'dir1-dx': partial(aggregate_dir_dx, eig_idx=1)}

SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}

def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(torch.nn.modules.activation)[activation]()

class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, in_size, hidden_size, out_size, last_activation='none',
                 dropout=0., last_b_norm=False, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        self.fully_connected.append(FCLayer(in_size, out_size, activation=last_activation, b_norm=last_b_norm,
                                            device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class FCLayer(nn.Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer
    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None,
                 device='cpu'):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout, device=device)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'

class DGNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, avg_d, residual):
        super().__init__()

        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        self.aggregators = aggregators

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim, hidden_size=in_dim,
                            out_size=in_dim, last_activation='none')
        self.posttrans = MLP(in_size=(len(aggregators) * 1 + 1) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, last_activation='none')
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        vector_field = edges.data['eig'] if vector_field is None else torch.cat((vector_field, edges.data['eig']), dim=1)
        return {'e': self.pretrans(z2), 'vector_field': vector_field}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'vector_field': edges.data['vector_field'].to('cuda' if torch.cuda.is_available() else 'cpu')}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']

        vector_field = nodes.mailbox['vector_field']
        D = h.shape[-2]

        h = torch.cat([aggregate(h, vector_field, h_in) for aggregate in self.aggregators], dim=1)

        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):

        h_in = h
        g.ndata['h'] = h

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization and residual
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)

        return h

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2, decreasing_dim=True):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class DGNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        device = net_params['device']
        self.device = device

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)

        # retrieve the aggregators and scalers functions
        self.aggregators = [AGGREGATORS[aggr] for aggr in self.aggregators.split()]

        self.layers = nn.ModuleList([DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      avg_d=self.avg_d) for _ in range(n_layers - 1)])
        self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, 
                                    avg_d=self.avg_d))

        # 128 out dim since ogbg-molpcba has 128 tasks
        self.MLP_layer = MLPReadout(out_dim, 128)


    def forward(self, g, h, e, snorm_n):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            h = h_t

        g.ndata['h'] = h

        hg = dgl.sum_nodes(g, 'h')

        return self.MLP_layer(hg)

    def loss(self, scores, labels):
        
        loss = 0
        loss_fn = torch.nn.BCEWithLogitsLoss()

        for i in range(labels.shape[1]):
            # ignore nan values
            is_labeled = labels[:,i] == labels[:,i]

            # otherwise loss will be 'nan'
            if (len(labels[is_labeled, i]) == 0):
                continue

            loss += loss_fn(scores[is_labeled,i], labels[is_labeled,i].to('cuda'))
        return loss

def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    epoch_train_AP = 0
    list_scores = []
    list_labels = []
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        batch_labels = batch_labels.to(device)
        batch_graphs = batch_graphs.to(device)
        optimizer.zero_grad()

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n)
        
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        list_scores.append(batch_scores.detach())
        list_labels.append(batch_labels.detach())

    epoch_loss /= (iter + 1)

    evaluator = Evaluator(name='ogbg-molpcba')
    epoch_train_AP = evaluator.eval({'y_pred': torch.cat(list_scores),
                                        'y_true': torch.cat(list_labels)})['ap']

    return epoch_loss, epoch_train_AP, optimizer


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_AP = 0
    with torch.no_grad():
        list_scores = []
        list_labels = []
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_labels = batch_labels.to(device)
            batch_graphs = batch_graphs.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n)

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            list_scores.append(batch_scores.detach())
            list_labels.append(batch_labels.detach())

        epoch_test_loss /= (iter + 1)

        evaluator = Evaluator(name='ogbg-molpcba')
        epoch_test_AP = evaluator.eval({'y_pred': torch.cat(list_scores),
                                           'y_true': torch.cat(list_labels)})['ap']

    return epoch_test_loss, epoch_test_AP

def view_model_param(net_params):
    model = DGNNet(net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print('DGN Total parameters:', total_param)
    return total_param

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
        
    loaded = False

    # try to load, possibly downgrading
    if os.path.exists(data_file):  # load
        print("Loading dataset from {}".format(data_file))
        graphs_dgl, orbit_partition_sizes, split_idx = torch.load(data_file)
        loaded = True

    else:  # try downgrading. Currently works only when for each k there is only one substructure in the family
        k_min = 3
        succeded, graphs_dgl, orbit_partition_sizes, split_idx = try_downgrading(data_folder, k, k_min)
        if succeded:  # save the dataset
            print("Saving dataset to {}".format(data_file))
            torch.save((graphs_dgl, orbit_partition_sizes, split_idx), data_file)
            loaded = True

    if not loaded:
        graphs_dgl, orbit_partition_sizes, split_idx = generate_dataset(path, name, **subgraph_params)
        if data_file is not None:
            print("Saving dataset to {}".format(data_file))
            torch.save((graphs_dgl, orbit_partition_sizes, split_idx), data_file)

    return graphs_dgl, split_idx

def try_downgrading(data_folder, k, k_min):
    '''
        Extracts the substructures of size up to the `k`, if a collection of substructures
        with size larger than k has already been computed.
    '''
    found_data_filename, k_found = find_id_filename(data_folder, k)
    if found_data_filename is not None:
        print("Loading dataset from {}".format(found_data_filename))
        graphs_dgl, orbit_partition_sizes, split_idx = torch.load(found_data_filename)
        print("Downgrading k from dataset {}...".format(found_data_filename))
        graphs_dgl, orbit_partition_sizes = downgrade_k(graphs_dgl, k, orbit_partition_sizes, k_min)
        return True, graphs_dgl, orbit_partition_sizes, split_idx
    else:
        return False, None, None, None

def find_id_filename(data_folder, k):
    '''
        Looks for existing precomputed datasets in `data_folder` with counts for substructure 
        `id_type` larger `k`.
    '''
    pattern = os.path.join(data_folder, 'cycle_graph_induced_[0-9]*.pt')
    filenames = glob.glob(pattern)
    for name in filenames:
        k_found = int(re.findall(r'\d+', name)[-1])
        if k_found >= k:
            return name, k_found
    return None, None

def downgrade_k(dataset, k, orbit_partition_sizes, k_min):
    '''
        Donwgrades `dataset` by keeping only the orbits of the requested substructures.
    '''
    feature_vector_size = sum(orbit_partition_sizes[0:k-k_min+1])
    graphs_dgl = list()
    for datapoint in dataset:
        g, label = datapoint
        g.edata['subgraph_counts'] = g.edata['subgraph_counts'][:, 0:feature_vector_size]
        graphs_dgl.append((g, label))
        
    return graphs_dgl, orbit_partition_sizes[0:k-k_min+1]

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

def train(dataset, params, net_params):
    t0 = time.time()
    per_epoch_time = []

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    device = net_params['device']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = DGNNet(net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_APs, epoch_val_APs, epoch_test_APs = [], [], []

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate, pin_memory=True)

    with tqdm(range(params['epochs']), unit='epoch') as t:
        for epoch in t:
            t.set_description('Epoch %d' % epoch)

            start = time.time()

            epoch_train_loss, epoch_train_ap, optimizer = train_epoch(model, optimizer, device, train_loader)
            epoch_val_loss, epoch_val_ap = evaluate_network(model, device, val_loader)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_APs.append(epoch_train_ap.item())
            epoch_val_APs.append(epoch_val_ap.item())

            _, epoch_test_ap = evaluate_network(model, device, test_loader)

            epoch_test_APs.append(epoch_test_ap.item())

            t.set_postfix(
                            train_loss=epoch_train_loss, 
                            train_AP=epoch_train_ap.item(), val_AP=epoch_val_ap.item(),
                            refresh=False)

            per_epoch_time.append(time.time() - start)

            scheduler.step(-epoch_val_ap.item())

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            print('')

    best_val_epoch = np.argmax(np.array(epoch_val_APs))
    best_train_epoch = np.argmax(np.array(epoch_train_APs))
    best_val_ap = epoch_val_APs[best_val_epoch]
    best_val_test_ap = epoch_test_APs[best_val_epoch]
    best_val_train_ap = epoch_train_APs[best_val_epoch]
    best_train_ap = epoch_train_APs[best_train_epoch]

    print("Best Train AP: {:.4f}".format(best_train_ap))
    print("Best Val AP: {:.4f}".format(best_val_ap))
    print("Test AP of Best Val: {:.4f}".format(best_val_test_ap))
    print("Train AP of Best Val: {:.4f}".format(best_val_train_ap))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, help="Please give a value for gpu id")
    parser.add_argument('--dataset', default="ogbg-molpcba", help="Please give a value for dataset name")
    parser.add_argument('--seed', default=41, help="Please give a value for seed")
    parser.add_argument('--epochs', default=450, help="Please give a value for epochs")
    parser.add_argument('--batch_size', default=2048, help="Please give a value for batch_size")
    parser.add_argument('--init_lr', default=0.0008, help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', default=0.8, help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', default=8, help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', default=0.00001, help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', default=1e-5, help="Please give a value for weight_decay")
    parser.add_argument('--L', default=4, help="Please give a value for L")
    parser.add_argument('--hidden_dim', default=420, help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', default=420, help="Please give a value for out_dim")
    parser.add_argument('--in_feat_dropout', default=0.0, help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', default=0.2, help="Please give a value for dropout")
    parser.add_argument('--graph_norm', default=True, help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', default=True, help="Please give a value for batch_norm")

    # dgn params
    parser.add_argument('--aggregators', default="mean sum max dir1-dx", type=str, help='Aggregators to use.')
    parser.add_argument('--divide_input_first', default=False, type=bool, help='Whether to divide the input in first layer.')
    parser.add_argument('--divide_input_last', default=True, type=bool, help='Whether to divide the input in last layers.')
    parser.add_argument('--root_folder', type=str, default='./')

    args = parser.parse_args()

    # device
    if args.gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

        if torch.cuda.is_available():
            print('cuda available with GPU:', torch.cuda.get_device_name(0))
            device = torch.device("cuda")
        else:
            print('cuda not available')
            device = torch.device("cpu")

    path = os.path.join('./', 'dataset', "ogbg-molpcba")
    dataset = pcbaDataset("ogbg-molpcba", path=path, verbose=True)

    # parameters
    params = {}
    params['seed'] = int(args.seed)
    params['epochs'] = int(args.epochs)
    params['batch_size'] = int(args.batch_size)
    params['init_lr'] = float(args.init_lr)
    params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    params['min_lr'] = float(args.min_lr)
    params['weight_decay'] = float(args.weight_decay)

    # network parameters
    net_params = {}
    net_params['device'] = device
    net_params['gpu_id'] = int(args.gpu_id)
    net_params['batch_size'] = params['batch_size']
    net_params['L'] = int(args.L)
    net_params['hidden_dim'] = int(args.hidden_dim)
    net_params['out_dim'] = int(args.out_dim)
    net_params['residual'] = True
    net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    net_params['dropout'] = float(args.dropout)
    net_params['graph_norm'] = True if args.graph_norm == 'True' else False
    net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    net_params['aggregators'] = args.aggregators
        
    D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                       dataset.train.graph_lists])
    net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))

    net_params['total_param'] = view_model_param(net_params)
    train(dataset, params, net_params)