import os
import sys
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import operator
import itertools

from timeit import default_timer as timer
from datetime import timedelta
from msg_alltoall import get_global_node_ids 

def getGlobalIdsForEdges( rank, size, edge_data, metis_partitions, node_data ): 

    #form a list of all the received node ids from rank-0
    #<src_id> <dst_id> <type_edge_id> <edge_type> <attributes>
    #convert partition.py requires edge data to be in this format. 
    #<src_id> <dst_id> <orig_src_id> <orig_dst_id> <orig_edge_id> <edge_type> <attributes>
    #
    #node_data --> [ global_id, node_type, node_type_id, line_id ]
    #lookup map to use to extract global_ids for received nodes is 
    #between global_id and line_id

    node_ids = np.unique(np.concatenate([edge_data[:, 0], edge_data[:, 1], node_data[:, 3]]))
    print( 'Rank: ', rank, ' No. nodes owned: ', len(node_ids))

    partitions = np.array(list(map(list, metis_partitions.items())))
    commons, ind1, ind2 = np.intersect1d(partitions[:,0], node_ids, return_indices=True)
    partitions = partitions[ ind1,:]

    nodeids_ranks = [] 
    for i in range(size): 
        if (i == rank): 
            nodeids_ranks.append([])
            continue
        not_owned_nodes = partitions[:,0][partitions[:,1] == i]
        nodeids_ranks.append(not_owned_nodes)

    #Retrieve Global-ids for respective node owners
    global_node_ids, nodeids_ranks = get_global_node_ids(rank, size, nodeids_ranks, partitions, node_data)

    for i in range(size): 
        if (i == rank): 
            own_nodeids = partitions[:,0][partitions[:,1] == i]
            common, ind1, ind2 = np.intersect1d(node_data[:,3], own_nodeids, return_indices=True)
            local_mappings = node_data[ind1,0]
            global_node_ids.extend( local_mappings )
            nodeids_ranks.extend( own_nodeids )

    resolved_mappings = dict(zip(nodeids_ranks, global_node_ids))
    src_id = [ resolved_mappings[ x ] for x in edge_data[:, 0] ]
    dst_id = [ resolved_mappings[ x ] for x in edge_data[:, 1] ]

    print( 'Rank: ', rank, ' ', len(src_id), ' ', len(dst_id), ' ', edge_data.shape)
    return np.c_[ src_id, np.c_[ dst_id, edge_data] ]

