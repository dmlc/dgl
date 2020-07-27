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

def partition_graph(g, graph_name, num_parts, out_path, num_hops=1, part_method="metis",
                    reshuffle=True, balance_ntypes=None, balance_edges=False):
    ''' Partition a graph for distributed training and store the partitions on files.

    The partitioning occurs in three steps: 1) run a partition algorithm (e.g., Metis) to
    assign nodes to partitions; 2) construct partition graph structure based on
    the node assignment; 3) split the node features and edge features based on
    the partition result.

    The partitioned data is stored into multiple files.

    First, the metadata of the original graph and the partitioning is stored in a JSON file
    named after `graph_name`. This JSON file contains the information of the originla graph
    as well as the file names that store each partition.

    The node assignment is stored in a separate file if we don't reshuffle node Ids to ensure
    that all nodes in a partition fall into a contiguous Id range. The node assignment is stored
    in a numpy file.

    All node features in a partition are stored in a file with DGL format. The node features are
    stored in a dictionary, in which the key is the node data name and the value is a tensor.

    All edge features in a partition are stored in a file with DGL format. The edge features are
    stored in a dictionary, in which the key is the edge data name and the value is a tensor.

    The graph structure of a partition is stored in a file with the DGLGraph format. The DGLGraph
    contains the mapping of node/edge Ids to the Ids in the global graph. The mappings can be
    accessed with `part.ndata[dgl.NID]` and `part.edata[dgl.NID]`, where `part` is the partition
    graph structure. In addition to the mapping, the partition graph contains node data
    ("inner_node" and "orig_id") and edge data ("inner_edge").

    * "inner_node" indicates whether a node belongs to a partition.
    * "inner_edge" indicates whether an edge belongs to a partition.
    * "orig_id" exists when reshuffle=True. It indicates the original node Ids in the original
    graph before reshuffling.

    When performing Metis partitioning, we can put some constraint on the partitioning.
    Current, it supports two constrants to balance the partitioning. By default, Metis
    always tries to balance the number of nodes in each partition.

    * `balance_ntypes` balances the number of nodes of different types in each partition.
    * `balance_edges` balances the number of edges in each partition.

    To balance the node types, a user needs to pass a vector of N elements to indicate
    the type of each node. N is the number of nodes in the input graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph to partition
    graph_name : str
        The name of the graph.
    num_parts : int
        The number of partitions
    num_hops : int
        The number of hops of HALO nodes we construct on a partition graph structure.
    part_method : str
        The partition method. It supports "random" and "metis".
    out_path : str
        The path to store the files for all partitioned data.
    reshuffle : bool
        Reshuffle nodes and edges so that nodes and edges in a partition are in
        contiguous Id range.
    balance_ntypes : tensor
        Node type of each node
    balance_edges : bool
        Indicate whether to balance the edges.
    '''
    if num_parts == 1:
        parts = {0: g}
        node_parts = F.zeros((g.number_of_nodes(),), F.int64, F.cpu())
        g.ndata[NID] = F.arange(0, g.number_of_nodes())
        g.edata[EID] = F.arange(0, g.number_of_edges())
        g.ndata['inner_node'] = F.ones((g.number_of_nodes(),), F.int8, F.cpu())
        g.edata['inner_edge'] = F.ones((g.number_of_edges(),), F.int8, F.cpu())
        if reshuffle:
            g.ndata['orig_id'] = F.arange(0, g.number_of_nodes())
            g.edata['orig_id'] = F.arange(0, g.number_of_edges())
    elif part_method == 'metis':
        node_parts = metis_partition_assignment(g, num_parts, balance_ntypes=balance_ntypes,
                                                balance_edges=balance_edges)
        parts = partition_graph_with_halo(g, node_parts, num_hops, reshuffle=reshuffle)
    elif part_method == 'random':
        node_parts = random_choice(num_parts, g.number_of_nodes())
        parts = partition_graph_with_halo(g, node_parts, num_hops, reshuffle=reshuffle)
    else:
        raise Exception('Unknown partitioning method: ' + part_method)

    # Let's calculate edge assignment.
    if not reshuffle:
        start = time.time()
        # We only optimize for reshuffled case. So it's fine to use int64 here.
        edge_parts = np.zeros((g.number_of_edges(),), dtype=np.int64) - 1
        for part_id in parts:
            part = parts[part_id]
            # To get the edges in the input graph, we should use original node Ids.
            local_edges = F.boolean_mask(part.edata[EID], part.edata['inner_edge'])
            edge_parts[F.asnumpy(local_edges)] = part_id
        print('Calculate edge assignment: {:.3f} seconds'.format(time.time() - start))

    os.makedirs(out_path, mode=0o775, exist_ok=True)
    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)

    # Without reshuffling, we have to store the entire node/edge mapping in a file.
    if not reshuffle:
        node_part_file = os.path.join(out_path, "node_map")
        edge_part_file = os.path.join(out_path, "edge_map")
        np.save(node_part_file, F.asnumpy(node_parts), allow_pickle=False)
        np.save(edge_part_file, edge_parts, allow_pickle=False)
        node_map_val = node_part_file + ".npy"
        edge_map_val = edge_part_file + ".npy"
    else:
        # With reshuffling, we can ensure that all nodes and edges are reshuffled
        # and are in contiguous Id space.
        if num_parts > 1:
            node_map_val = [F.as_scalar(F.sum(F.astype(parts[i].ndata['inner_node'], F.int64),
                                              0)) for i in parts]
            node_map_val = np.cumsum(node_map_val).tolist()
            assert node_map_val[-1] == g.number_of_nodes()
            edge_map_val = [F.as_scalar(F.sum(F.astype(parts[i].edata['inner_edge'], F.int64),
                                              0)) for i in parts]
            edge_map_val = np.cumsum(edge_map_val).tolist()
            assert edge_map_val[-1] == g.number_of_edges()
        else:
            node_map_val = [g.number_of_nodes()]
            edge_map_val = [g.number_of_edges()]

    start = time.time()
    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g.number_of_nodes(),
                     'num_edges': g.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val}
    for part_id in range(num_parts):
        part = parts[part_id]

        # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            # To get the edges in the input graph, we should use original node Ids.
            ndata_name = 'orig_id' if reshuffle else NID
            edata_name = 'orig_id' if reshuffle else EID
            local_nodes = F.boolean_mask(part.ndata[ndata_name], part.ndata['inner_node'])
            local_edges = F.boolean_mask(part.edata[edata_name], part.edata['inner_edge'])
            print('part {} has {} nodes and {} edges.'.format(
                part_id, part.number_of_nodes(), part.number_of_edges()))
            print('{} nodes and {} edges are inside the partition'.format(
                len(local_nodes), len(local_edges)))
            tot_num_inner_edges += len(local_edges)
            for name in g.ndata:
                if name in [NID, 'inner_node']:
                    continue
                node_feats[name] = F.gather_row(g.ndata[name], local_nodes)
            for name in g.edata:
                if name in [EID, 'inner_edge']:
                    continue
                edge_feats[name] = F.gather_row(g.edata[name], local_edges)
        else:
            for name in g.ndata:
                if name in [NID, 'inner_node']:
                    continue
                node_feats[name] = g.ndata[name]
            for name in g.edata:
                if name in [EID, 'inner_edge']:
                    continue
                edge_feats[name] = g.edata[name]

        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                    'edge_feats': edge_feat_file,
                                                    'part_graph': part_graph_file}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)
        save_graphs(part_graph_file, [part])

    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)
    print('Save partitions: {:.3f} seconds'.format(time.time() - start))

    num_cuts = g.number_of_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print('There are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), num_cuts, num_parts))
