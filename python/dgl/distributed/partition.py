"""Functions for partitions.

For distributed training, a graph is partitioned and partitions are stored in files
organized as follows:

```
data_root_dir/
  |-- part_conf.json      # partition configuration file in JSON
  |-- node_map            # partition id of each node stored in a numpy array
  |-- edge_map            # partition id of each edge stored in a numpy array
  |-- part0/              # data for partition 0
      |-- node_feats      # node features stored in binary format
      |-- edge_feats      # edge features stored in binary format
      |-- graph           # graph structure of this partition stored in binary format
  |-- part1/              # data for partition 1
      |-- node_feats
      |-- edge_feats
      |-- graph
```

The partition configuration file stores the file locations. For the above example,
the configuration file will look like the following:

```
{
  "graph_name" : "test",
  "part_method" : "metis",
  "num_parts" : 2,
  "halo_hops" : 1,
  "node_map" : "data_root_dir/node_map.npy",
  "edge_map" : "data_root_dir/edge_map.npy"
  "num_nodes" : 1000000,
  "num_edges" : 52000000,
  "part-0" : {
    "node_feats" : "data_root_dir/part0/node_feats.dgl",
    "edge_feats" : "data_root_dir/part0/edge_feats.dgl",
    "part_graph" : "data_root_dir/part0/graph.dgl",
  },
  "part-1" : {
    "node_feats" : "data_root_dir/part1/node_feats.dgl",
    "edge_feats" : "data_root_dir/part1/edge_feats.dgl",
    "part_graph" : "data_root_dir/part1/graph.dgl",
  },
}
```

Here are the definition of the fields in the partition configuration file:
    * `graph_name` is the name of the graph given by a user.
    * `part_method` is the method used to assign nodes to partitions.
      Currently, it supports "random" and "metis".
    * `num_parts` is the number of partitions.
    * `halo_hops` is the number of HALO nodes we want to include in a partition.
    * `node_map` is the node assignment map, which tells the partition Id a node is assigned to.
    * `edge_map` is the edge assignment map, which tells the partition Id an edge is assigned to.
    * `num_nodes` is the number of nodes in the global graph.
    * `num_edges` is the number of edges in the global graph.
    * `part-*` stores the data of a partition.

Nodes in each partition is *relabeled* to always start with zero. We call the node
ID in the original graph, *global ID*, while the relabeled ID in each partition,
*local ID*. Each partition graph has an integer node data tensor stored under name
`dgl.NID` and each value is the node's global ID. Similarly, edges are relabeled too
and the mapping from local ID to global ID is stored as an integer edge data tensor
under name `dgl.EID`.

Note that each partition can contain *HALO* nodes and edges, those belonging to
other partitions but are included in this partition for integrity or efficiency concerns.
We call nodes and edges that truly belong to one partition *local nodes/edges*, while
the rest "HALO nodes/edges".

Node and edge features are splitted and stored together with each graph partition.
We do not store features of HALO nodes and edges.

Two useful functions in this module:
    * :func:`~dgl.distributed.load_partition` loads one partition and the meta data into memory.
    * :func:`~dgl.distributed.partition` partitions a graph into files organized as above.

"""

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
from .graph_partition_book import GraphPartitionBook, RangePartitionBook

def load_partition(conf_file, part_id):
    ''' Load data of a partition from the data path in the DistGraph server.

    A partition data includes a graph structure of the partition, a dict of node tensors,
    a dict of edge tensors and some metadata. The partition may contain the HALO nodes,
    which are the nodes replicated from other partitions. However, the dict of node tensors
    only contains the node data that belongs to the local partition. Similarly, edge tensors
    only contains the edge data that belongs to the local partition. The metadata include
    the information of the global graph (not the local partition), which includes the number
    of nodes, the number of edges as well as the node assignment of the global graph.

    The function currently loads data through the normal filesystem interface. In the future,
    we need to support loading data from other storage such as S3 and HDFS.

    Parameters
    ----------
    conf_file : str
        The path of the partition config file.
    part_id : int
        The partition Id.

    Returns
    -------
    DGLGraph
        The graph partition structure.
    dict of tensors
        All node features.
    dict of tensors
        All edge features.
    GraphPartitionBook
        The global partition information.
    str
        The graph name
    '''
    with open(conf_file) as conf_f:
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

    gpb, graph_name = load_partition_book(conf_file, part_id, graph)
    nids = F.boolean_mask(graph.ndata[NID], graph.ndata['inner_node'])
    partids = gpb.nid2partid(nids)
    assert np.all(F.asnumpy(partids == part_id)), 'load a wrong partition'
    return graph, node_feats, edge_feats, gpb, graph_name

def load_partition_book(conf_file, part_id, graph=None):
    ''' Load a graph partition book from the partition config file.

    Parameters
    ----------
    conf_file : str
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
    with open(conf_file) as conf_f:
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
        return GraphPartitionBook(part_id, num_parts, node_map, edge_map,
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
