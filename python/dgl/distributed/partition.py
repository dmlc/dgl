"""Functions for partitions. """

import json
import os
import time
import numpy as np
import torch

from .. import backend as F
from ..base import NID, EID
from ..random import choice as random_choice
from ..data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from ..transform import metis_partition_assignment, partition_graph_with_halo
from .graph_partition_book import BasicPartitionBook, RangePartitionBook

def load_partition(part_config, part_id):
    ''' Load data of a partition from the data path.

    A partition data includes a graph structure of the partition, a dict of node tensors,
    a dict of edge tensors and some metadata. The partition may contain the HALO nodes,
    which are the nodes replicated from other partitions. However, the dict of node tensors
    only contains the node data that belongs to the local partition. Similarly, edge tensors
    only contains the edge data that belongs to the local partition. The metadata include
    the information of the global graph (not the local partition), which includes the number
    of nodes, the number of edges as well as the node assignment of the global graph.

    The function currently loads data through the local filesystem interface.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition Id.

    Returns
    -------
    DGLGraph
        The graph partition structure.
    dict of tensors
        Node features.
    dict of tensors
        Edge features.
    GraphPartitionBook
        The graph partition information.
    str
        The graph name
    '''
    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert 'part-{}'.format(part_id) in part_metadata, "part-{} does not exist".format(part_id)
    part_files = part_metadata['part-{}'.format(part_id)]
    assert 'node_feats' in part_files, "the partition does not contain node features."
    assert 'edge_feats' in part_files, "the partition does not contain edge feature."
    assert 'part_graph' in part_files, "the partition does not contain graph structure."
    node_feats = load_tensors(part_files['node_feats'])
    edge_feats = load_tensors(part_files['edge_feats'])
    graph = load_graphs(part_files['part_graph'])[0][0]

    assert NID in graph.ndata, "the partition graph should contain node mapping to global node Id"
    assert EID in graph.edata, "the partition graph should contain edge mapping to global edge Id"

    gpb, graph_name = load_partition_book(part_config, part_id, graph)
    nids = F.boolean_mask(graph.ndata[NID], graph.ndata['inner_node'])
    partids = gpb.nid2partid(nids)
    assert np.all(F.asnumpy(partids == part_id)), 'load a wrong partition'
    return graph, node_feats, edge_feats, gpb, graph_name

def load_partition_book(part_config, part_id, graph=None):
    ''' Load a graph partition book from the partition config file.

    Parameters
    ----------
    part_config : str
        The path of the partition config file.
    part_id : int
        The partition Id.
    graph : DGLGraph
        The graph structure

    Returns
    -------
    GraphPartitionBook
        The global partition information.
    str
        The graph name
    '''
    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
    assert 'num_parts' in part_metadata, 'num_parts does not exist.'
    assert part_metadata['num_parts'] > part_id, \
            'part {} is out of range (#parts: {})'.format(part_id, part_metadata['num_parts'])
    num_parts = part_metadata['num_parts']
    assert 'num_nodes' in part_metadata, "cannot get the number of nodes of the global graph."
    assert 'num_edges' in part_metadata, "cannot get the number of edges of the global graph."
    assert 'node_map' in part_metadata, "cannot get the node map."
    assert 'edge_map' in part_metadata, "cannot get the edge map."
    assert 'graph_name' in part_metadata, "cannot get the graph name"

    # If this is a range partitioning, node_map actually stores a list, whose elements
    # indicate the boundary of range partitioning. Otherwise, node_map stores a filename
    # that contains node map in a NumPy array.
    is_range_part = isinstance(part_metadata['node_map'], list)
    node_map = part_metadata['node_map'] if is_range_part else np.load(part_metadata['node_map'])
    edge_map = part_metadata['edge_map'] if is_range_part else np.load(part_metadata['edge_map'])
    assert isinstance(node_map, list) == isinstance(edge_map, list), \
            "The node map and edge map need to have the same format"

    if is_range_part:
        return RangePartitionBook(part_id, num_parts, np.array(node_map),
                                  np.array(edge_map)), part_metadata['graph_name']
    else:
        return BasicPartitionBook(part_id, num_parts, node_map, edge_map,
                                  graph), part_metadata['graph_name']

def partition_graph(g, num_parts, part_method="metis", **kwargs):
    if num_parts == 1:
        return torch.zeros(g.number_of_nodes()).long()
    elif part_method == 'metis':
        group = metis_partition_assignment(g, num_parts, **kwargs)
    elif part_method == 'random':
        group = random_choice(num_parts, g.number_of_nodes())
    else:
        raise Exception('Unknown partitioning method: ' + part_method)
    return group
