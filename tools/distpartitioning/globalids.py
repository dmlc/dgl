import itertools
import operator

import constants

import numpy as np
import torch
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
    # build a list of sizes (lengths of lists)
    global_nids_ranks = [torch.from_numpy(x) for x in global_nids_ranks]
    recv_nodes = alltoallv_cpu(rank, world_size, global_nids_ranks)

    # Use node_data to lookup global id to send over.
    send_nodes = []
    for proc_i_nodes in recv_nodes:
        # list of node-ids to lookup
        if proc_i_nodes is not None:
            global_nids = proc_i_nodes.numpy()
            if len(global_nids) != 0:
                common, ind1, ind2 = np.intersect1d(
                    node_data[constants.GLOBAL_NID],
                    global_nids,
                    return_indices=True,
                )
                shuffle_global_nids = node_data[constants.SHUFFLE_GLOBAL_NID][
                    ind1
                ]
                send_nodes.append(
                    torch.from_numpy(shuffle_global_nids).type(
                        dtype=torch.int64
                    )
                )
            else:
                send_nodes.append(torch.empty((0), dtype=torch.int64))
        else:
            send_nodes.append(torch.empty((0), dtype=torch.int64))

    # send receive global-ids
    recv_shuffle_global_nids = alltoallv_cpu(rank, world_size, send_nodes)
    shuffle_global_nids = np.concatenate(
        [x.numpy() if x is not None else [] for x in recv_shuffle_global_nids]
    )
    global_nids = np.concatenate([x for x in global_nids_ranks])
    ret_val = np.column_stack([global_nids, shuffle_global_nids])
    return ret_val


def lookup_shuffle_global_nids_edges(
    rank, world_size, num_parts, edge_data, id_lookup, node_data
):
    """
    This function is a helper function used to lookup shuffle-global-nids for a given set of
    global-nids using a distributed lookup service.

    Parameters:
    -----------
    rank : integer
        rank of the process
    world_size : integer
        total number of processes used in the process group
    num_parts : integer
        total number of output graph partitions
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
    """
    # Make sure that the outgoing message size does not exceed 2GB in size.
    # Even though gloo can handle upto 10GB size of data in the outgoing messages,
    # it needs additional memory to store temporary information into the buffers which will increase
    # the memory needs of the process.
    MILLION = 1000 * 1000
    BATCH_SIZE = 250 * MILLION
    memory_snapshot("GlobalToShuffleIDMapBegin: ", rank)

    local_nids = []
    local_shuffle_nids = []
    for local_part_id in range(num_parts // world_size):
        local_nids.append(
            node_data[constants.GLOBAL_NID + "/" + str(local_part_id)]
        )
        local_shuffle_nids.append(
            node_data[constants.SHUFFLE_GLOBAL_NID + "/" + str(local_part_id)]
        )

    local_nids = np.concatenate(local_nids)
    local_shuffle_nids = np.concatenate(local_shuffle_nids)

    for local_part_id in range(num_parts // world_size):
        node_list = edge_data[
            constants.GLOBAL_SRC_ID + "/" + str(local_part_id)
        ]

        # Determine the no. of times each process has to send alltoall messages.
        all_sizes = allgather_sizes(
            [node_list.shape[0]], world_size, num_parts, return_sizes=True
        )
        max_count = np.amax(all_sizes)
        num_splits = max_count // BATCH_SIZE + 1

        # Split the message into batches and send.
        splits = np.array_split(node_list, num_splits)
        shuffle_mappings = []
        for item in splits:
            shuffle_ids = id_lookup.get_shuffle_nids(
                item, local_nids, local_shuffle_nids, world_size
            )
            shuffle_mappings.append(shuffle_ids)

        shuffle_ids = np.concatenate(shuffle_mappings)
        assert shuffle_ids.shape[0] == node_list.shape[0]
        edge_data[
            constants.SHUFFLE_GLOBAL_SRC_ID + "/" + str(local_part_id)
        ] = shuffle_ids

        # Destination end points of edges are owned by the current node and therefore
        # should have corresponding SHUFFLE_GLOBAL_NODE_IDs.
        # Here retrieve SHUFFLE_GLOBAL_NODE_IDs for the destination end points of local edges.
        uniq_ids, inverse_idx = np.unique(
            edge_data[constants.GLOBAL_DST_ID + "/" + str(local_part_id)],
            return_inverse=True,
        )
        common, idx1, idx2 = np.intersect1d(
            uniq_ids,
            node_data[constants.GLOBAL_NID + "/" + str(local_part_id)],
            assume_unique=True,
            return_indices=True,
        )
        assert len(common) == len(uniq_ids)

        edge_data[
            constants.SHUFFLE_GLOBAL_DST_ID + "/" + str(local_part_id)
        ] = node_data[constants.SHUFFLE_GLOBAL_NID + "/" + str(local_part_id)][
            idx2
        ][
            inverse_idx
        ]
        assert len(
            edge_data[
                constants.SHUFFLE_GLOBAL_DST_ID + "/" + str(local_part_id)
            ]
        ) == len(edge_data[constants.GLOBAL_DST_ID + "/" + str(local_part_id)])

    memory_snapshot("GlobalToShuffleIDMap_AfterLookupServiceCalls: ", rank)
    return edge_data


def assign_shuffle_global_nids_nodes(rank, world_size, num_parts, node_data):
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
    num_parts : integer
        total number of output graph partitions
    node_data : dictionary
        node_data is a dictionary with keys as column names and values as numpy arrays
    """
    # Compute prefix sum to determine node-id offsets
    local_row_counts = []
    for local_part_id in range(num_parts // world_size):
        local_row_counts.append(
            node_data[constants.GLOBAL_NID + "/" + str(local_part_id)].shape[0]
        )

    # Perform allgather to compute the local offsets.
    prefix_sum_nodes = allgather_sizes(local_row_counts, world_size, num_parts)

    for local_part_id in range(num_parts // world_size):
        shuffle_global_nid_start = prefix_sum_nodes[
            rank + (local_part_id * world_size)
        ]
        shuffle_global_nid_end = prefix_sum_nodes[
            rank + 1 + (local_part_id * world_size)
        ]
        shuffle_global_nids = np.arange(
            shuffle_global_nid_start, shuffle_global_nid_end, dtype=np.int64
        )
        node_data[
            constants.SHUFFLE_GLOBAL_NID + "/" + str(local_part_id)
        ] = shuffle_global_nids


def assign_shuffle_global_nids_edges(rank, world_size, num_parts, edge_data):
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
    num_parts : integer
        total number of output graph partitions
    edge_data : numpy ndarray
        edge data as read from xxx_edges.txt file

    Returns:
    --------
    integer
        shuffle_global_eid_start, which indicates the starting value from which shuffle_global-ids are assigned to edges
        on this rank
    """
    # get prefix sum of edge counts per rank to locate the starting point
    # from which global-ids to edges are assigned in the current rank
    local_row_counts = []
    for local_part_id in range(num_parts // world_size):
        local_row_counts.append(
            edge_data[constants.GLOBAL_SRC_ID + "/" + str(local_part_id)].shape[
                0
            ]
        )

    shuffle_global_eid_offset = []
    prefix_sum_edges = allgather_sizes(local_row_counts, world_size, num_parts)
    for local_part_id in range(num_parts // world_size):
        shuffle_global_eid_start = prefix_sum_edges[
            rank + (local_part_id * world_size)
        ]
        shuffle_global_eid_end = prefix_sum_edges[
            rank + 1 + (local_part_id * world_size)
        ]
        shuffle_global_eids = np.arange(
            shuffle_global_eid_start, shuffle_global_eid_end, dtype=np.int64
        )
        edge_data[
            constants.SHUFFLE_GLOBAL_EID + "/" + str(local_part_id)
        ] = shuffle_global_eids
        shuffle_global_eid_offset.append(shuffle_global_eid_start)

    return shuffle_global_eid_offset
