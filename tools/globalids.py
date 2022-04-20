import numpy as np
import torch
import operator
import itertools

from gloo_wrapper import allgather_sizes, alltoall_cpu, alltoallv_cpu

def get_global_node_ids(rank, world_size, nodeids_ranks, partitions, node_data):
    """
    For non-local node, whose orig-node-id <-> global-id is not present at the current rank
    retrieve their global_ids. 

    Parameters: 
    -----------
    rank : rank of the process
    world_size : total no. of ranks configured
    nodeids_ranks : list of lists of orig-node-ids, index of the list is the rank of the process
                    where orig-node-id <-> global_id mapping is located. 
    partitions : metis partitions, which are orig-node-id <-> rank mappings. 
    node_data : node data (augmented) 
    """

    #build a list of sizes (lengths of lists)
    sizes = [len(x) for x in nodeids_ranks]

    #compute total_nodes whose mappings should be resolved, between orig-node-id <-> global-id
    total_nodes = np.sum(sizes)
    if (total_nodes == 0):
        print('Rank: ', rank, ' -- All mappings are present locally... No need for to send any info.')
        return [], []

    #determine the no. of orig-node-ids to send and receive and perform alltoall
    send_counts = list(torch.Tensor(sizes).type(dtype=torch.int32).chunk(world_size))
    recv_counts = list(torch.zeros([world_size], dtype=torch.int32).chunk(world_size))
    alltoall_cpu(rank, world_size, recv_counts, send_counts)

    #allocate buffers to receive node-ids
    recv_nodes = []
    for i in recv_counts:
        recv_nodes.append(torch.zeros([i.item()], dtype=torch.int32))

    #form the outgoing message
    send_nodes = []
    for i in range(world_size):
        send_nodes.append(torch.Tensor(nodeids_ranks[i]).type(dtype=torch.int32))

    #send-recieve messages
    alltoallv_cpu(rank, world_size, recv_nodes, send_nodes)

    # for each of the received orig-node-id requests lookup and send out the global node id
    send_sizes = [len(x.tolist()) for x in recv_nodes]
    send_counts = list(torch.Tensor(send_sizes).type(dtype=torch.int32).chunk(world_size))
    recv_counts = list(torch.zeros([world_size], dtype=torch.int32).chunk(world_size))
    alltoall_cpu( rank, world_size, recv_counts, send_counts)

    # allocate buffers to receive global-ids
    recv_global_ids = []
    for i in recv_counts:
        recv_global_ids.append(torch.zeros([i.item()], dtype=torch.int32))

    # Use node_data to lookup global id to send over.
    send_nodes = []
    for i in recv_nodes:
        #list of node-ids to lookup
        node_ids = i.tolist()
        if (len(node_ids) != 0):
            common, ind1, ind2 = np.intersect1d(node_data[:,3], node_ids, return_indices=True)
            values = node_data[ind1,0]
            send_nodes.append(torch.Tensor(values).type(dtype=torch.int32))
        else:
            send_nodes.append(torch.Tensor([]).type(dtype=torch.int32))

    #send receive global-ids
    alltoallv_cpu(rank, world_size, recv_global_ids, send_nodes)

    #form the lists with global-ids and orig-node-ids
    recv_global_ids = [ x.tolist() for x in recv_global_ids ]
    global_ids = list(itertools.chain(*recv_global_ids))
    send_nodes = list(itertools.chain(*nodeids_ranks))

    return global_ids, send_nodes


def get_globalids_edges( rank, world_size, edge_data, metis_partitions, node_data ):
    """
    Edges which are owned by this rank, may have orig-node-ids whose global-ids are present locally.
    This function retrieves global-ids for such orig-node-ids.

    Parameters: 
    -----------
    rank : rank of the process
    world_size : total no. of processes used
    edge_data : edge_data (augmented) as read from the xxx_edges.txt file
    metis_partitions : orig-node-id to rank/partition-id mappins as determined by METIS
    node_data : node_data (augmented) as read from xxx_nodes.txt file
    
    """

    #determine unique node-ids present locally
    node_ids = np.unique(np.concatenate([edge_data[:, 0], edge_data[:, 1], node_data[:, 3]]))

    #determine the rank which owns orig-node-id <-> partition/rank mappings
    partitions = np.array(list(map(list, metis_partitions.items())))
    commons, ind1, ind2 = np.intersect1d(partitions[:,0], node_ids, return_indices=True)
    partitions = partitions[ ind1,:]

    #form list of lists, each list includes orig-node-ids whose mappings needs to be resovlved.
    #and rank will be the process which owns mappings of these orig-node-ids
    nodeids_ranks = []
    for i in range(world_size):
        if (i == rank):
            nodeids_ranks.append([])
            continue
        not_owned_nodes = partitions[:,0][partitions[:,1] == i]
        nodeids_ranks.append(not_owned_nodes)

    #Retrieve Global-ids for respective node owners
    global_node_ids, nodeids_ranks = get_global_node_ids(rank, world_size, nodeids_ranks, partitions, node_data)

    #Add orig-node-id <-> global-id mappings to the received data
    for i in range(world_size):
        if (i == rank):
            own_nodeids = partitions[:,0][partitions[:,1] == i]
            common, ind1, ind2 = np.intersect1d(node_data[:,3], own_nodeids, return_indices=True)
            local_mappings = node_data[ind1,0]
            global_node_ids.extend( local_mappings )
            nodeids_ranks.extend( own_nodeids )

    #form a dictionary of mappings between orig-node-ids and global-ids
    resolved_mappings = dict(zip(nodeids_ranks, global_node_ids))

    #determine global-ids for the orig-src-id and orig-dst-id
    global_src_id = [ resolved_mappings[ x ] for x in edge_data[:, 0] ]
    global_dst_id = [ resolved_mappings[ x ] for x in edge_data[:, 1] ]

    return np.c_[ global_src_id, np.c_[ global_dst_id, edge_data] ]


def assign_globalids_nodes(rank, world_size, typewise_nodecount, node_data):
    """
    Utility function to assign global ids to nodes at a given rank
    node_data gets converted from [orig-node-type, orig-type-node-id, orig-node-id]
    to [global-id, orig-node-type, orig-type-node-id, orig-node-id, partition-ntype-id]
    where global-id : global id of the node after data shuffle
            orig-node-type : node-type as read from xxx_nodes.txt
            orig-type-node-id : node-type-id as read from xxx_nodes.txt
            orig-node-id : node-id as read from xxx_nodes.txt, implicitly 
                            this is the line no. in the file
            partition-ntype-id : ntype_id assigned by the current rank within its scope
    """
    # sort the list of tuples by nodeType
    typewise_nodecount.sort( key=lambda x: x[0] )

    #add all types of nodes at the current rank
    idx0 = operator.itemgetter(1)
    local_node_count = sum(list(map(idx0, typewise_nodecount)))

    # Compute prefix sum to determine node-id offsets
    prefix_sum_nodes = allgather_sizes([local_node_count], world_size)

    # assigning node-ids from localNodeStartId to (localNodeEndId - 1)
    # Assuming here that the nodeDataArr is sorted based on the nodeType.
    global_id_start = prefix_sum_nodes[rank]
    global_id_end = prefix_sum_nodes[rank + 1]

    # add a column with global-ids (after data shuffle)
    global_ids = np.arange(global_id_start, global_id_end)
    augmted_node_data = np.c_[global_ids, node_data]

    #Add a new column, which will mimic the orgi_type_nid, but locally
    #This column will be used to index into the node-features.
    #nodeDataArr is already sorted based on ntype
    partition_ntype_id = []
    for x in typewise_nodecount:
        ntype_id = x[0]
        ntype_id_count = x[1]
        partition_ntype_id.extend([i for i in range(ntype_id_count)])

    #Add this column to the node_data
    return np.c_[ augmted_node_data, partition_ntype_id], global_id_start


def assign_globalids_edges(rank, world_size, typewise_edgecount, edge_data):
    """
    Utility function to assign global-ids to edges
    edge_data gets converted from [ orig-src-id, orig-dst-id, orig-etype-id, orig-etype ]
    to [ global-src-id, global-dst-id, orig-src-id, orig-dst-id, orig-etype-id, orig-etype ]

    Parameters:
    -----------
    rank : rank of the current process
    world_size : total count of processes in execution
    typewise_edgecount : list of tuples (x,y), x = rank, y = no. of edges
    edge_data : edge data as read from xxx_edges.txt file
    """
    #sort the list of tuples by edgeType
    typewise_edgecount.sort( key=lambda x: x[0] )

    #compute total no. of edges  
    idx0 = operator.itemgetter( 1 )
    local_edge_count = sum(list(map(idx0, typewise_edgecount)))
    
    #get prefix sum of edge counts per rank to locate the starting point
    #from which global-ids to edges are assigned in the current rank
    prefix_sum_edges = allgather_sizes([local_edge_count], world_size)
    global_id_start = prefix_sum_edges[rank]
    global_id_end = prefix_sum_edges[rank + 1]

    # assigning edge-ids from localEdgeStart to (localEdgeEndId - 1)
    # Assuming here that the edge_data is sorted by edge_type
    global_ids = np.arange(global_id_start, global_id_end)
    return np.c_[ edge_data, global_ids], global_id_start
