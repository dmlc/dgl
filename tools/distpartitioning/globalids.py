import itertools
import operator

import numpy as np
import torch

import constants
from dist_lookup import DistLookupService
from gloo_wrapper import allgather_sizes, alltoallv_cpu
from utils import memory_snapshot


def get_shuffle_global_nids(rank, world_size, global_nids_ranks, node_data):
    """ 
    For nodes which are not owned by the current rank, whose global_nid <-> shuffle_global-nid mapping
    is not present at the current rank, this function retrieves their shuffle_global_ids from the owner rank

    Parameters: 
    -----------
    rank : integer
        rank of the process
    world_size : integer
        total no. of ranks configured
    global_nids_ranks : list
        list of numpy arrays (of global_nids), index of the list is the rank of the process
                    where global_nid <-> shuffle_global_nid mapping is located. 
    node_data : dictionary
        node_data is a dictionary with keys as column names and values as numpy arrays

    Returns:
    --------
    numpy ndarray
        where the column-0 are global_nids and column-1 are shuffle_global_nids which are retrieved
        from other processes. 
    """
    #build a list of sizes (lengths of lists)
    global_nids_ranks = [torch.from_numpy(x) for x in global_nids_ranks]
    recv_nodes = alltoallv_cpu(rank, world_size, global_nids_ranks)

    # Use node_data to lookup global id to send over.
    send_nodes = []
    for proc_i_nodes in recv_nodes:
        #list of node-ids to lookup
        if proc_i_nodes is not None: 
            global_nids = proc_i_nodes.numpy()
            if(len(global_nids) != 0):
                common, ind1, ind2 = np.intersect1d(node_data[constants.GLOBAL_NID], global_nids, return_indices=True)
                shuffle_global_nids = node_data[constants.SHUFFLE_GLOBAL_NID][ind1]
                send_nodes.append(torch.from_numpy(shuffle_global_nids).type(dtype=torch.int64))
            else:
                send_nodes.append(torch.empty((0), dtype=torch.int64))
        else:
            send_nodes.append(torch.empty((0), dtype=torch.int64))

    #send receive global-ids
    recv_shuffle_global_nids = alltoallv_cpu(rank, world_size, send_nodes)
    shuffle_global_nids = np.concatenate([x.numpy() if x is not None else [] for x in recv_shuffle_global_nids])
    global_nids = np.concatenate([x for x in global_nids_ranks])
    ret_val = np.column_stack([global_nids, shuffle_global_nids])
    return ret_val

def lookup_shuffle_global_nids_edges(rank, world_size, edge_data, id_lookup, node_data):
    '''
    This function is a helper function used to lookup shuffle-global-nids for a given set of
    global-nids using a distributed lookup service.

    Parameters:
    -----------
    rank : integer
        rank of the process
    world_size : integer
        total number of processes used in the process group
    edge_data : dictionary
        edge_data is a dicitonary with keys as column names and values as numpy arrays representing
        all the edges present in the current graph partition
    id_lookup : instance of DistLookupService class
        instance of a distributed lookup service class which is used to retrieve partition-ids and
        shuffle-global-nids for any given set of global-nids
    node_data : dictionary
        node_data is a dictionary with keys as column names and values as numpy arrays representing
        all the nodes owned by the current process

    Returns:
    --------
    dictionary :
        dictionary where keys are column names and values are numpy arrays representing all the
        edges present in the current graph partition
    '''
    memory_snapshot("GlobalToShuffleIDMapBegin: ", rank)
    node_list = np.concatenate([edge_data[constants.GLOBAL_SRC_ID], edge_data[constants.GLOBAL_DST_ID]])
    shuffle_ids = id_lookup.get_shuffle_nids(node_list,
                                            node_data[constants.GLOBAL_NID],
                                            node_data[constants.SHUFFLE_GLOBAL_NID])

    edge_data[constants.SHUFFLE_GLOBAL_SRC_ID], edge_data[constants.SHUFFLE_GLOBAL_DST_ID] = np.split(shuffle_ids, 2)
    memory_snapshot("GlobalToShuffleIDMap_AfterLookupServiceCalls: ", rank)
    return edge_data

def assign_shuffle_global_nids_nodes(rank, world_size, node_data):
    """
    Utility function to assign shuffle global ids to nodes at a given rank
    node_data gets converted from [ntype, global_type_nid, global_nid]
    to [shuffle_global_nid, ntype, global_type_nid, global_nid, part_local_type_nid]
    where shuffle_global_nid : global id of the node after data shuffle
            ntype : node-type as read from xxx_nodes.txt
            global_type_nid : node-type-id as read from xxx_nodes.txt
            global_nid : node-id as read from xxx_nodes.txt, implicitly 
                            this is the line no. in the file
            part_local_type_nid : type_nid assigned by the current rank within its scope
            
    Parameters:
    -----------
    rank : integer
        rank of the process
    world_size : integer
        total number of processes used in the process group
    ntype_counts: list of tuples
        list of tuples (x,y), where x=ntype and y=no. of nodes whose shuffle_global_nids are needed
    node_data : dictionary
        node_data is a dictionary with keys as column names and values as numpy arrays
    """
    # Compute prefix sum to determine node-id offsets
    prefix_sum_nodes = allgather_sizes([node_data[constants.GLOBAL_NID].shape[0]], world_size)

    # assigning node-ids from localNodeStartId to (localNodeEndId - 1)
    # Assuming here that the nodeDataArr is sorted based on the nodeType.
    shuffle_global_nid_start = prefix_sum_nodes[rank]
    shuffle_global_nid_end = prefix_sum_nodes[rank + 1]

    # add a column with global-ids (after data shuffle)
    shuffle_global_nids = np.arange(shuffle_global_nid_start, shuffle_global_nid_end, dtype=np.int64)
    node_data[constants.SHUFFLE_GLOBAL_NID] = shuffle_global_nids


def assign_shuffle_global_nids_edges(rank, world_size, edge_data):
    """
    Utility function to assign shuffle_global_eids to edges
    edge_data gets converted from [global_src_nid, global_dst_nid, global_type_eid, etype]
    to [shuffle_global_src_nid, shuffle_global_dst_nid, global_src_nid, global_dst_nid, global_type_eid, etype]

    Parameters:
    -----------
    rank : integer
        rank of the current process
    world_size : integer
        total count of processes in execution
    etype_counts : list of tuples
        list of tuples (x,y), x = rank, y = no. of edges
    edge_data : numpy ndarray
        edge data as read from xxx_edges.txt file

    Returns:
    --------
    integer
        shuffle_global_eid_start, which indicates the starting value from which shuffle_global-ids are assigned to edges
        on this rank
    """
    #get prefix sum of edge counts per rank to locate the starting point
    #from which global-ids to edges are assigned in the current rank
    prefix_sum_edges = allgather_sizes([edge_data[constants.GLOBAL_SRC_ID].shape[0]], world_size)
    shuffle_global_eid_start = prefix_sum_edges[rank]
    shuffle_global_eid_end = prefix_sum_edges[rank + 1]

    # assigning edge-ids from localEdgeStart to (localEdgeEndId - 1)
    # Assuming here that the edge_data is sorted by edge_type
    shuffle_global_eids = np.arange(shuffle_global_eid_start, shuffle_global_eid_end, dtype=np.int64)
    edge_data[constants.SHUFFLE_GLOBAL_EID] = shuffle_global_eids
    return shuffle_global_eid_start
