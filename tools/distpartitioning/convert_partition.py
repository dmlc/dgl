import argparse
import gc
import json
import logging
import os
import time

import dgl
import numpy as np
import pandas as pd
import pyarrow
import torch as th
from pyarrow import csv

import constants
from utils import get_idranges, memory_snapshot, read_json
from dgl.distributed.partition import (
    RESERVED_FIELD_DTYPE,
    _etype_str_to_tuple,
    _etype_tuple_to_str,
)


def create_dgl_object(schema, part_id, node_data, edge_data, edgeid_offset,
                        return_orig_nids=False, return_orig_eids=False):
    """
    This function creates dgl objects for a given graph partition, as in function
    arguments. 

    The "schema" argument is a dictionary, which contains the metadata related to node ids
    and edge ids. It contains two keys: "nid" and "eid", whose value is also a dictionary
    with the following structure. 

    1. The key-value pairs in the "nid" dictionary has the following format.
       "ntype-name" is the user assigned name to this node type. "format" describes the 
       format of the contents of the files. and "data" is a list of lists, each list has
       3 elements: file-name, start_id and end_id. File-name can be either absolute or
       relative path to this file and starting and ending ids are type ids of the nodes 
       which are contained in this file. These type ids are later used to compute global ids
       of these nodes which are used throughout the processing of this pipeline. 
        "ntype-name" : {
            "format" : "csv", 
            "data" : [
                    [ <path-to-file>/ntype0-name-0.csv, start_id0, end_id0], 
                    [ <path-to-file>/ntype0-name-1.csv, start_id1, end_id1],
                    ...
                    [ <path-to-file>/ntype0-name-<p-1>.csv, start_id<p-1>, end_id<p-1>],
            ]
        }

    2. The key-value pairs in the "eid" dictionary has the following format.
       As described for the "nid" dictionary the "eid" dictionary is similarly structured
       except that these entries are for edges. 
        "etype-name" : {
            "format" : "csv", 
            "data" : [
                    [ <path-to-file>/etype0-name-0, start_id0, end_id0], 
                    [ <path-to-file>/etype0-name-1 start_id1, end_id1],
                    ...
                    [ <path-to-file>/etype0-name-1 start_id<p-1>, end_id<p-1>]
            ]
        }

    In "nid" dictionary, the type_nids are specified that
    should be assigned to nodes which are read from the corresponding nodes file. 
    Along the same lines dictionary for the key "eid" is used for edges in the 
    input graph.

    These type ids, for nodes and edges, are used to compute global ids for nodes
    and edges which are stored in the graph object.

    Parameters:
    -----------
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
    edgeid_offset : int
        offset to be used when assigning edge global ids in the current partition
    return_orig_ids : bool, optional
        Indicates whether to return original node/edge IDs.

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
    dict of tensors
        If `return_orig_nids=True`, return a dict of 1D tensors whose key is the node type
        and value is a 1D tensor mapping between shuffled node IDs and the original node
        IDs for each node type. Otherwise, ``None`` is returned.
    dict of tensors
        If `return_orig_eids=True`, return a dict of 1D tensors whose key is the edge type
        and value is a 1D tensor mapping between shuffled edge IDs and the original edge
        IDs for each edge type. Otherwise, ``None`` is returned.
    """
    #create auxiliary data structures from the schema object
    memory_snapshot("CreateDGLObj_Begin", part_id)
    _, global_nid_ranges = get_idranges(schema[constants.STR_NODE_TYPE],
                                    schema[constants.STR_NUM_NODES_PER_CHUNK])
    _, global_eid_ranges = get_idranges(schema[constants.STR_EDGE_TYPE],
                                    schema[constants.STR_NUM_EDGES_PER_CHUNK])

    id_map = dgl.distributed.id_map.IdMap(global_nid_ranges)

    ntypes = [(key, global_nid_ranges[key][0, 0]) for key in global_nid_ranges]
    ntypes.sort(key=lambda e: e[1])
    ntype_offset_np = np.array([e[1] for e in ntypes])
    ntypes = [e[0] for e in ntypes]
    ntypes_map = {e: i for i, e in enumerate(ntypes)}
    etypes = [(key, global_eid_ranges[key][0, 0]) for key in global_eid_ranges]
    etypes.sort(key=lambda e: e[1])
    etypes = [e[0] for e in etypes]
    etypes_map = {_etype_str_to_tuple(e): i for i, e in enumerate(etypes)}

    node_map_val = {ntype: [] for ntype in ntypes}
    edge_map_val = {_etype_str_to_tuple(etype): [] for etype in etypes}

    memory_snapshot("CreateDGLObj_AssignNodeData", part_id)
    shuffle_global_nids = node_data[constants.SHUFFLE_GLOBAL_NID]
    node_data.pop(constants.SHUFFLE_GLOBAL_NID)
    gc.collect()

    ntype_ids = node_data[constants.NTYPE_ID]
    node_data.pop(constants.NTYPE_ID)
    gc.collect()

    global_type_nid = node_data[constants.GLOBAL_TYPE_NID]
    node_data.pop(constants.GLOBAL_TYPE_NID)
    node_data = None
    gc.collect()

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
    memory_snapshot("CreateDGLObj_AssignEdgeData: ", part_id)
    shuffle_global_src_id = edge_data[constants.SHUFFLE_GLOBAL_SRC_ID]
    edge_data.pop(constants.SHUFFLE_GLOBAL_SRC_ID)
    gc.collect()

    shuffle_global_dst_id = edge_data[constants.SHUFFLE_GLOBAL_DST_ID]
    edge_data.pop(constants.SHUFFLE_GLOBAL_DST_ID)
    gc.collect()

    global_src_id = edge_data[constants.GLOBAL_SRC_ID]
    edge_data.pop(constants.GLOBAL_SRC_ID)
    gc.collect()

    global_dst_id = edge_data[constants.GLOBAL_DST_ID]
    edge_data.pop(constants.GLOBAL_DST_ID)
    gc.collect()

    global_edge_id = edge_data[constants.GLOBAL_TYPE_EID]
    edge_data.pop(constants.GLOBAL_TYPE_EID)
    gc.collect()

    etype_ids = edge_data[constants.ETYPE_ID]
    edge_data.pop(constants.ETYPE_ID)
    edge_data = None
    gc.collect()    
    logging.info(f'There are {len(shuffle_global_src_id)} edges in partition {part_id}')

    # It's not guaranteed that the edges are sorted based on edge type.
    # Let's sort edges and all attributes on the edges.
    if not np.all(np.diff(etype_ids) >= 0):
        sort_idx = np.argsort(etype_ids)
        shuffle_global_src_id, shuffle_global_dst_id, global_src_id, global_dst_id, global_edge_id, etype_ids = \
                shuffle_global_src_id[sort_idx], shuffle_global_dst_id[sort_idx], global_src_id[sort_idx], \
                global_dst_id[sort_idx], global_edge_id[sort_idx], etype_ids[sort_idx]
        assert np.all(np.diff(etype_ids) >= 0)
    else:
        print(f'[Rank: {part_id} Edge data is already sorted !!!')

    # Determine the edge ID range of different edge types.
    edge_id_start = edgeid_offset 
    for etype_name in global_eid_ranges:
        etype = _etype_str_to_tuple(etype_name)
        assert len(etype) == 3
        etype_id = etypes_map[etype]
        edge_map_val[etype].append([edge_id_start,
            edge_id_start + np.sum(etype_ids == etype_id)])
        edge_id_start += np.sum(etype_ids == etype_id)
    memory_snapshot("CreateDGLObj_UniqueNodeIds: ", part_id)

    # get the edge list in some order and then reshuffle.
    # Here the order of nodes is defined by the `np.unique` function
    # node order is as listed in the uniq_ids array
    ids = np.concatenate(
        [shuffle_global_src_id, shuffle_global_dst_id,
            np.arange(shuffle_global_nid_range[0], shuffle_global_nid_range[1] + 1)])
    uniq_ids, idx, inverse_idx = np.unique(
        ids, return_index=True, return_inverse=True)
    assert len(uniq_ids) == len(idx)

    # We get the edge list with their node IDs mapped to a contiguous ID range.
    part_local_src_id, part_local_dst_id = np.split(inverse_idx[:len(shuffle_global_src_id) * 2], 2)
    inner_nodes = th.as_tensor(np.logical_and(
            uniq_ids >= shuffle_global_nid_range[0],
            uniq_ids <= shuffle_global_nid_range[1]))

    #get the list of indices, from inner_nodes, which will sort inner_nodes as [True, True, ...., False, False, ...]
    #essentially local nodes will be placed before non-local nodes.
    reshuffle_nodes = th.arange(len(uniq_ids))
    reshuffle_nodes = th.cat([reshuffle_nodes[inner_nodes.bool()],
                              reshuffle_nodes[inner_nodes == 0]])

    '''
    Following procedure is used to map the part_local_src_id, part_local_dst_id to account for
    reshuffling of nodes (to order localy owned nodes prior to non-local nodes in a partition)
    1. Form a node_map, in this case a numpy array, which will be used to map old node-ids (pre-reshuffling)
    to post-reshuffling ids.
    2. Once the map is created, use this map to map all the node-ids in the part_local_src_id 
    and part_local_dst_id list to their appropriate `new` node-ids (post-reshuffle order).
    3. Since only the node's order is changed, we will have to re-order nodes related information when
    creating dgl object: this includes dgl.NTYPE, dgl.NID and inner_node.
    4. Edge's order is not changed. At this point in the execution path edges are still ordered by their etype-ids.
    5. Create the dgl object appropriately and return the dgl object.
    
    Here is a  simple example to understand the above flow better.

    part_local_nids = [0, 1, 2, 3, 4, 5]
    part_local_src_ids = [0, 0, 0, 0, 2, 3, 4]
    part_local_dst_ids = [1, 2, 3, 4, 4, 4, 5]

    Assume that nodes {1, 5} are halo-nodes, which are not owned by this partition.

    reshuffle_nodes = [0, 2, 3, 4, 1, 5]

    A node_map, which maps node-ids from old to reshuffled order is as follows:
    node_map = np.zeros((len(reshuffle_nodes,)))
    node_map[reshuffle_nodes] = np.arange(len(reshuffle_nodes))

    Using the above map, we have mapped part_local_src_ids and part_local_dst_ids as follows:
    part_local_src_ids = [0, 0, 0, 0, 1, 2, 3]
    part_local_dst_ids = [4, 1, 2, 3, 3, 3, 5]

    In this graph above, note that nodes {0, 1, 2, 3} are inner_nodes and {4, 5} are NON-inner-nodes

    Since the edge are re-ordered in any way, there is no reordering required for edge related data
    during the DGL object creation.
    '''
    #create the mappings to generate mapped part_local_src_id and part_local_dst_id
    #This map will map from unshuffled node-ids to reshuffled-node-ids (which are ordered to prioritize 
    #locally owned nodes).
    nid_map = np.zeros((len(reshuffle_nodes,)))
    nid_map[reshuffle_nodes] = np.arange(len(reshuffle_nodes))

    #Now map the edge end points to reshuffled_values.
    part_local_src_id, part_local_dst_id = nid_map[part_local_src_id], nid_map[part_local_dst_id]

    #create the graph here now.
    part_graph = dgl.graph(data=(part_local_src_id, part_local_dst_id), num_nodes=len(uniq_ids))
    part_graph.edata[dgl.EID] = th.arange(
        edgeid_offset, edgeid_offset + part_graph.number_of_edges(), dtype=th.int64)
    part_graph.edata[dgl.ETYPE] = th.as_tensor(etype_ids, dtype=RESERVED_FIELD_DTYPE[dgl.ETYPE])
    part_graph.edata['inner_edge'] = th.ones(part_graph.number_of_edges(),
        dtype=RESERVED_FIELD_DTYPE['inner_edge'])


    #compute per_type_ids and ntype for all the nodes in the graph.
    global_ids = np.concatenate(
            [global_src_id, global_dst_id, global_homo_nid])
    part_global_ids = global_ids[idx]
    part_global_ids = part_global_ids[reshuffle_nodes]
    ntype, per_type_ids = id_map(part_global_ids)

    #continue with the graph creation
    part_graph.ndata[dgl.NTYPE] = th.as_tensor(ntype, dtype=RESERVED_FIELD_DTYPE[dgl.NTYPE])
    part_graph.ndata[dgl.NID] = th.as_tensor(uniq_ids[reshuffle_nodes])
    part_graph.ndata['inner_node'] = th.as_tensor(inner_nodes[reshuffle_nodes],
        dtype=RESERVED_FIELD_DTYPE['inner_node'])

    orig_nids = None
    orig_eids = None
    if return_orig_nids:
        orig_nids = {}
        for ntype, ntype_id in ntypes_map.items():
            mask = th.logical_and(part_graph.ndata[dgl.NTYPE] == ntype_id,
                                    part_graph.ndata['inner_node'])
            orig_nids[ntype] = th.as_tensor(per_type_ids[mask])
    if return_orig_eids:
        orig_eids = {}
        for etype, etype_id in etypes_map.items():
            mask = th.logical_and(part_graph.edata[dgl.ETYPE] == etype_id,
                                    part_graph.edata['inner_edge'])
            orig_eids[_etype_tuple_to_str(etype)] = th.as_tensor(global_edge_id[mask])


    return part_graph, node_map_val, edge_map_val, ntypes_map, etypes_map, \
        orig_nids, orig_eids

def create_metadata_json(graph_name, num_nodes, num_edges, part_id, num_parts, node_map_val, \
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
    part_id : int
       integer indicating the partition id
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

    part_dir = 'part' + str(part_id)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                'edge_feats': edge_feat_file,
                                                'part_graph': part_graph_file}
    return part_metadata
