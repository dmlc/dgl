import os
import sys
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from timeit import default_timer as timer
from datetime import timedelta

from msg_allgather import executeAllGather

import operator 
import itertools

'''
Perform all-to-all and compute the prefix sum of the resulting array
to compute the offsets at each of the ranks
'''
def exchangeLocalCounts( rank, data, worldSize ): 

    rankwiseCounts = executeAllGather( data, worldSize )

    # Compute Prefix sum to get the offsets. 
    rankwiseOffsets = [0] * (len(rankwiseCounts) + 1)
    totalCount = 0
    for idx, count in enumerate( rankwiseCounts ): 
        totalCount += count
        rankwiseOffsets[ idx + 1 ] = totalCount

    return rankwiseOffsets

'''
lstNodeTypeCounts = list of tuples, (nodeType, nodeTypeCount)
nodeDataArr = [ global-id, node-type, orig-type-node-id ]
    global-id --> row-idx from the xxx_nodes.txt
    node-type --> node-type from all the node types
    orig-type-node-id --> node-type-ids as defined in the xxx_nodes.txt
    line-id --> line no. from the xxx_nodes.txt for this node

Also append a new column, which is analogous to orig_type_nid, 
Call this local_type_nid and this always starts with '0' and ends with len(features) - 1.
'''
def assignGlobalNodeIds ( rank, worldSize, lstNodeTypeCounts, nodeDataArr ): 
    # sort the list of tuples by nodeType
    lstNodeTypeCounts.sort( key=lambda x: x[0] )

    # across all node types. 
    idx0 = operator.itemgetter(1)
    localCount = sum( list( map(idx0, lstNodeTypeCounts) ) )

    # Compute prefix sum to determine node-id offsets
    rankwiseOffsets = exchangeLocalCounts(rank,  [ localCount ], worldSize )

    # assigning node-ids from localNodeStartId to (localNodeEndId - 1)
    # Assuming here that the nodeDataArr is sorted based on the nodeType. 
    localNodeStartId = rankwiseOffsets[ rank ]
    localNodeEndId = rankwiseOffsets[ rank + 1 ]

    # add a column with global-ids (after data shuffle)
    localIdx = np.arange(localNodeStartId, localNodeEndId)
    node_data_aug = np.c_[ localIdx, nodeDataArr ]

    #Add a new column, which will mimic the orgi_type_nid, but locally
    #This column will be used to index into the node-features. 
    #nodeDataArr is already sorted based on ntype
    local_type_nid = []
    for x in lstNodeTypeCounts:
        ntype_id = x[0] 
        ntype_id_count = x[1]
        local_type_nid.extend( [ x for i in range(ntype_id_count) ] )
       
    #Add this column to the node_data
    node_data_aug = np.c_[ node_data_aug, local_type_nid ]

    return node_data_aug, localNodeStartId

'''
EdgeDataArr [ orig_src_id, orig_dst_id, orig-type-id, orig-type ]
to 
EdgeDataArr[ src_id, dst_id, orig_src-id, orig-dst-id, orig-type-id, orig-type ]
'''
def assignGlobalEdgeIds ( rank, worldSize, lstEdgeTypeCounts, edgeDataArr ): 
    #sort the list of tuples by edgeType
    lstEdgeTypeCounts.sort( key=lambda x: x[0] )

    # across all edge types
    idx0 = operator.itemgetter( 1 )
    localCounts = sum( list( map( idx0, lstEdgeTypeCounts ) ) )

    rankwiseOffsets = exchangeLocalCounts( rank, [ localCounts ], worldSize )

    #assigning edge-ids from localEdgeStart to (localEdgeEndId - 1)
    localEdgeStartId = rankwiseOffsets[ rank ]
    localEdgeEndId = rankwiseOffsets[ rank + 1 ]

    # assigning edge-ids from localEdgeStart to (localEdgeEndId - 1)
    # Assuming here that the edgeDataArr is sorted by edgeType
    localIdx = np.arange( localEdgeStartId, localEdgeEndId )
    edge_data_aug  = np.c_[ edgeDataArr, localIdx ]

    return edge_data_aug


