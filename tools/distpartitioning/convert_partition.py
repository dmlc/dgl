import os
import json
import time
import argparse
import numpy as np
import dgl
import torch as th
import pyarrow
import pandas as pd
import constants
from pyarrow import csv

def create_dgl_object(graph_name, num_parts, \
                        schema, part_id, node_data, \
                        edge_data, nodeid_offset, edgeid_offset):
    """
    This function creates dgl objects for a given graph partition, as in function
    arguments. 

    Parameters:
    -----------
    graph_name : string
        name of the graph
    num_parts : int
        total no. of partitions (of the original graph)
    schame : json object
        json object created by reading the graph metadata json file
    part_id : int
        partition id of the graph partition for which dgl object is to be created
    node_data : numpy ndarray
        node_data, where each row is of the following format:
        <global_nid> <ntype_id> <global_type_nid>
    edge_data : numpy ndarray
        edge_data, where each row is of the following format: 
        <global_src_id> <global_dst_id> <etype_id> <global_type_eid>
    nodeid_offset : int
        offset to be used when assigning node global ids in the current partition
    edgeid_offset : int
        offset to be used when assigning edge global ids in the current partition

    return compact_g2, node_map_val, edge_map_val, ntypes_map, etypes_map

    Returns: 
    --------
    dgl object
        dgl object created for the current graph partition
    dictionary
        map between node types and the range of global node ids used
    dictionary
        map between edge types and the range of global edge ids used
    dictionary
        map between node type(string)  and node_type_id(int)
    dictionary
        map between edge type(string)  and edge_type_id(int)
    """

    #create auxiliary data structures from the schema object
    global_nid_ranges = schema['nid']
    global_eid_ranges = schema['eid']
    global_nid_ranges = {key: np.array(global_nid_ranges[key]).reshape(
        1, 2) for key in global_nid_ranges}
    global_eid_ranges = {key: np.array(global_eid_ranges[key]).reshape(
        1, 2) for key in global_eid_ranges}
    id_map = dgl.distributed.id_map.IdMap(global_nid_ranges)

    ntypes = [(key, global_nid_ranges[key][0, 0]) for key in global_nid_ranges]
    ntypes.sort(key=lambda e: e[1])
    ntype_offset_np = np.array([e[1] for e in ntypes])
    ntypes = [e[0] for e in ntypes]
    ntypes_map = {e: i for i, e in enumerate(ntypes)}
    etypes = [(key, global_eid_ranges[key][0, 0]) for key in global_eid_ranges]
    etypes.sort(key=lambda e: e[1])
    etype_offset_np = np.array([e[1] for e in etypes])
    etypes = [e[0] for e in etypes]
    etypes_map = {e: i for i, e in enumerate(etypes)}

    node_map_val = {ntype: [] for ntype in ntypes}
    edge_map_val = {etype: [] for etype in etypes}

    shuffle_global_nids, ntype_ids, global_type_nid = node_data[constants.SHUFFLE_GLOBAL_NID], \
            node_data[constants.NTYPE_ID], node_data[constants.GLOBAL_TYPE_NID]

    global_homo_nid = ntype_offset_np[ntype_ids] + global_type_nid
    assert np.all(shuffle_global_nids[1:] - shuffle_global_nids[:-1] == 1)
    shuffle_global_nid_range = (shuffle_global_nids[0], shuffle_global_nids[-1])


    # Determine the node ID ranges of different node types.
    for ntype_name in global_nid_ranges:
        ntype_id = ntypes_map[ntype_name]
        type_nids = shuffle_global_nids[ntype_ids == ntype_id]
        node_map_val[ntype_name].append(
            [int(type_nids[0]), int(type_nids[-1]) + 1])

    #process edges
    shuffle_global_src_id, shuffle_global_dst_id, global_src_id, global_dst_id, global_edge_id, etype_ids = \
                edge_data[constants.SHUFFLE_GLOBAL_SRC_ID], edge_data[constants.SHUFFLE_GLOBAL_DST_ID], \
                edge_data[constants.GLOBAL_SRC_ID], edge_data[constants.GLOBAL_DST_ID], \
                edge_data[constants.GLOBAL_TYPE_EID], edge_data[constants.ETYPE_ID]
    print('There are {} edges in partition {}'.format(len(shuffle_global_src_id), part_id))

    # It's not guaranteed that the edges are sorted based on edge type.
    # Let's sort edges and all attributes on the edges.
    sort_idx = np.argsort(etype_ids)
    shuffle_global_src_id, shuffle_global_dst_id, global_src_id, global_dst_id, global_edge_id, etype_ids = \
            shuffle_global_src_id[sort_idx], shuffle_global_dst_id[sort_idx], global_src_id[sort_idx], \
            global_dst_id[sort_idx], global_edge_id[sort_idx], etype_ids[sort_idx]
    assert np.all(np.diff(etype_ids) >= 0)

    # Determine the edge ID range of different edge types.
    edge_id_start = edgeid_offset 
    for etype_name in global_eid_ranges:
        etype_id = etypes_map[etype_name]
        edge_map_val[etype_name].append([edge_id_start,
                                         edge_id_start + np.sum(etype_ids == etype_id)])
        edge_id_start += np.sum(etype_ids == etype_id)

    # Here we want to compute the unique IDs in the edge list.
    # It is possible that a node that belongs to the partition but it doesn't appear
    # in the edge list. That is, the node is assigned to this partition, but its neighbor
    # belongs to another partition so that the edge is assigned to another partition.
    # This happens in a directed graph.
    # To avoid this kind of nodes being removed from the graph, we add node IDs that
    # belong to this partition.
    ids = np.concatenate(
        [shuffle_global_src_id, shuffle_global_dst_id, np.arange(shuffle_global_nid_range[0], shuffle_global_nid_range[1] + 1)])
    uniq_ids, idx, inverse_idx = np.unique(
        ids, return_index=True, return_inverse=True)
    assert len(uniq_ids) == len(idx)
    # We get the edge list with their node IDs mapped to a contiguous ID range.
    part_local_src_id, part_local_dst_id = np.split(inverse_idx[:len(shuffle_global_src_id) * 2], 2)
    compact_g = dgl.graph((part_local_src_id, part_local_dst_id))
    compact_g.edata['orig_id'] = th.as_tensor(global_edge_id)
    compact_g.edata[dgl.ETYPE] = th.as_tensor(etype_ids)
    compact_g.edata['inner_edge'] = th.ones(
        compact_g.number_of_edges(), dtype=th.bool)

    # The original IDs are homogeneous IDs.
    # Similarly, we need to add the original homogeneous node IDs
    global_nids = np.concatenate([global_src_id, global_dst_id, global_homo_nid])
    global_homo_ids = global_nids[idx]
    ntype, per_type_ids = id_map(global_homo_ids)
    compact_g.ndata['orig_id'] = th.as_tensor(per_type_ids)
    compact_g.ndata[dgl.NTYPE] = th.as_tensor(ntype)
    compact_g.ndata[dgl.NID] = th.as_tensor(uniq_ids)
    compact_g.ndata['inner_node'] = th.as_tensor(np.logical_and(
        uniq_ids >= shuffle_global_nid_range[0], uniq_ids <= shuffle_global_nid_range[1]))
    part_local_nids = compact_g.ndata[dgl.NID][compact_g.ndata['inner_node'].bool()]
    assert np.all((part_local_nids == th.arange(
        part_local_nids[0], part_local_nids[-1] + 1)).numpy())
    print('|V|={}'.format(compact_g.number_of_nodes()))
    print('|E|={}'.format(compact_g.number_of_edges()))

    # We need to reshuffle nodes in a partition so that all local nodes are labelled starting from 0.
    reshuffle_nodes = th.arange(compact_g.number_of_nodes())
    reshuffle_nodes = th.cat([reshuffle_nodes[compact_g.ndata['inner_node'].bool()],
                              reshuffle_nodes[compact_g.ndata['inner_node'] == 0]])
    compact_g1 = dgl.node_subgraph(compact_g, reshuffle_nodes)
    compact_g1.ndata['orig_id'] = compact_g.ndata['orig_id'][reshuffle_nodes]
    compact_g1.ndata[dgl.NTYPE] = compact_g.ndata[dgl.NTYPE][reshuffle_nodes]
    compact_g1.ndata[dgl.NID] = compact_g.ndata[dgl.NID][reshuffle_nodes]
    compact_g1.ndata['inner_node'] = compact_g.ndata['inner_node'][reshuffle_nodes]
    compact_g1.edata['orig_id'] = compact_g.edata['orig_id'][compact_g1.edata[dgl.EID]]
    compact_g1.edata[dgl.ETYPE] = compact_g.edata[dgl.ETYPE][compact_g1.edata[dgl.EID]]
    compact_g1.edata['inner_edge'] = compact_g.edata['inner_edge'][compact_g1.edata[dgl.EID]]

    # reshuffle edges on ETYPE as node_subgraph relabels edges
    idx = th.argsort(compact_g1.edata[dgl.ETYPE])
    u, v = compact_g1.edges()
    u = u[idx]
    v = v[idx]
    compact_g2 = dgl.graph((u, v))
    compact_g2.ndata['orig_id'] = compact_g1.ndata['orig_id']
    compact_g2.ndata[dgl.NTYPE] = compact_g1.ndata[dgl.NTYPE]
    compact_g2.ndata[dgl.NID] = compact_g1.ndata[dgl.NID]
    compact_g2.ndata['inner_node'] = compact_g1.ndata['inner_node']
    compact_g2.edata['orig_id'] = compact_g1.edata['orig_id'][idx]
    compact_g2.edata[dgl.ETYPE] = compact_g1.edata[dgl.ETYPE][idx]
    compact_g2.edata['inner_edge'] = compact_g1.edata['inner_edge'][idx]
    compact_g2.edata[dgl.EID] = th.arange(
        edgeid_offset, edgeid_offset + compact_g2.number_of_edges(), dtype=th.int64)
    edgeid_offset += compact_g2.number_of_edges()

    return compact_g2, node_map_val, edge_map_val, ntypes_map, etypes_map

def create_metadata_json(graph_name, num_nodes, num_edges, num_parts, node_map_val, \
                            edge_map_val, ntypes_map, etypes_map, output_dir ):
    """
    Auxiliary function to create json file for the graph partition metadata

    Parameters:
    -----------
    graph_name : string
        name of the graph
    num_nodes : int
        no. of nodes in the graph partition
    num_edges : int
        no. of edges in the graph partition
    num_parts : int
        total no. of partitions of the original graph
    node_map_val : dictionary
        map between node types and the range of global node ids used
    edge_map_val : dictionary
        map between edge types and the range of global edge ids used
    ntypes_map : dictionary
        map between node type(string)  and node_type_id(int)
    etypes_map : dictionary
        map between edge type(string)  and edge_type_id(int)
    output_dir : string
        directory where the output files are to be stored 

    Returns:
    --------
    dictionary
        map describing the graph information

    """
    part_metadata = {'graph_name': graph_name,
                     'num_nodes': num_nodes,
                     'num_edges': num_edges,
                     'part_method': 'metis',
                     'num_parts': num_parts,
                     'halo_hops': 1,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val,
                     'ntypes': ntypes_map,
                     'etypes': etypes_map}

    for part_id in range(num_parts):
        part_dir = 'part' + str(part_id)
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                'edge_feats': edge_feat_file,
                                                'part_graph': part_graph_file}
    return part_metadata
