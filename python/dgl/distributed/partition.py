from .. import backend as F
from ..base import ALL, NID, EID
from ..data.utils import load_graphs, save_graphs
from ..transform import metis_partition_assignment, partition_graph_with_halo

import json
import numpy as np
import pickle
import os

def load_partition(conf_file, part_id):
    ''' Load data of a partition from the data path in the DistGraph server.

    Here we load data through the normal filesystem interface. In the future, we need to support
    loading data from other storage such as S3 and HDFS.
    '''
    with open(conf_file) as f:
        part_metadata = json.load(f)
    graph_name = part_metadata['graph_name']
    part_files = part_metadata['part-{}'.format(part_id)]
    node_feats = pickle.load(open(part_files['node_feats'], 'rb'))
    edge_feats = pickle.load(open(part_files['edge_feats'], 'rb'))
    client_g = load_graphs(part_files['part_graph'])[0][0]
    node_map = pickle.load(open(part_metadata['node_map'], 'rb'))
    meta = (part_metadata['num_nodes'], part_metadata['num_edges'], node_map)

    part_ids = node_map[client_g.ndata[NID]]
    # TODO we need to fix this. DGL backend doesn't support boolean or byte.
    # int64 is unnecessary.
    client_g.ndata['local_node'] = F.astype(part_ids == part_id, F.int64)

    return client_g, node_feats, edge_feats, meta

def partition_graph(g, graph_name, num_parts, num_hops, part_method, out_path):
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
    in a pickle file.

    All node features in a partition are stored in a pickle file. The node features are stored
    in a dictionary, in which the key is the node data name and the value is a tensor.

    All edge features in a partition are stored in a pickle file. The edge features are stored
    in a dictionary, in which the key is the edge data name and the value is a tensor.

    The graph structure of a partition is stored in a file with the DGLGraph format. The DGLGraph
    contains the mapping of node/edge Ids to the Ids in the original graph.

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
    '''
    if num_parts == 1:
        client_parts = {0: g}
        node_parts = F.zeros((g.number_of_nodes()), F.int64, F.cpu())
        g.ndata[NID] = F.arange(0, g.number_of_nodes())
        g.edata[EID] = F.arange(0, g.number_of_edges())
    elif part_method == 'metis':
        node_parts = metis_partition_assignment(g, num_parts)
        client_parts = partition_graph_with_halo(g, node_parts, num_hops)
    elif part_method == 'random':
        node_parts = np.random.choice(num_parts, g.number_of_nodes())
        client_parts = partition_graph_with_halo(g, node_parts, num_hops)
    else:
        raise Exception('unknown partitioning method: ' + part_method)

    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)
    node_part_file = '{}/{}-node_part.pkl'.format(out_path, graph_name)
    pickle.dump(node_parts, open(node_part_file, 'wb'))
    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g.number_of_nodes(),
                     'num_edges': g.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_split': 'original',
                     'node_map': node_part_file}
    for part_id in range(num_parts):
        part = client_parts[part_id]

        # Get the node Ids that belong to this partition.
        part_ids = node_parts[part.ndata[NID]]
        local_nids = F.asnumpy(part.ndata[NID])[F.asnumpy(part_ids) == part_id]

        # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            local_nodes = F.asnumpy(part.ndata[NID])[F.asnumpy(part.ndata['inner_node']) == 1]
            local_edges = F.asnumpy(part.edata[EID])[F.asnumpy(part.edata['inner_edge']) == 1]
            print('part {} has {} nodes and {} edges. {} nodes and {} edges are inside the partition'.format(
                part_id, part.number_of_nodes(), part.number_of_edges(),
                len(local_nodes), len(local_edges)))
            tot_num_inner_edges += len(local_edges)
            for name in g.ndata:
                node_feats[name] = g.ndata[name][local_nodes]
            for name in g.edata:
                edge_feats[name] = g.edata[name][local_edges]
        else:
            for name in g.ndata:
                node_feats[name] = g.ndata[name]
            for name in g.edata:
                edge_feats[name] = g.edata[name]

        node_feat_file = '{}/{}-node_feat-{}.pkl'.format(out_path, graph_name, part_id)
        edge_feat_file = '{}/{}-edge_feat-{}.pkl'.format(out_path, graph_name, part_id)
        part_graph_file = '{}/{}-part_graph-{}.dgl'.format(out_path, graph_name, part_id)
        part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                    'edge_feats': edge_feat_file,
                                                    'part_graph': part_graph_file}
        pickle.dump(node_feats, open(node_feat_file, 'wb'))
        pickle.dump(edge_feats, open(edge_feat_file, 'wb'))
        save_graphs(part_graph_file, [part])

    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

    num_cuts = g.number_of_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print('there are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), num_cuts, num_parts))
