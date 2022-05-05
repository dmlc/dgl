import numpy as np
import torch
import operator
import itertools
from gloo_wrapper import allgather_sizes, alltoall_cpu, alltoallv_cpu

def get_shuffle_global_nids(rank, world_size, global_nids_ranks, node_data):
    """ 
    For nodes which are not owned by the current rank, whose global_nid <-> shuffle_global-nid 
    is not present at the current rank, this function retrieves their shuffle_global_ids from the owning
    rank

    Parameters: 
    -----------
    rank : integer
        rank of the process
    world_size : integer
        total no. of ranks configured
    global_nids_ranks : list
        list of numpy arrays (of global_nids), index of the list is the rank of the process
                    where global_nid <-> shuffle_global_nid mapping is located. 
    node_data : numpy ndarray, integers
        node data with additional columns inserted

    Returns:
    --------
    numpy ndarray
        where the column-0 are global_nids and column-1 are shuffle_global_nids which are retrieved
        from other processes. 
    """
    #build a list of sizes (lengths of lists)
    sizes = [len(x) for x in global_nids_ranks]

    #compute total_nodes whose mappings should be resolved, between orig-node-id <-> global-id
    total_nodes = np.sum(sizes)
    if (total_nodes == 0):
        print('Rank: ', rank, ' -- All mappings are present locally... No need for to send any info.')
        return None

    #determine the no. of global_node_ids to send and receive and perform alltoall
    send_counts = list(torch.Tensor(sizes).type(dtype=torch.int64).chunk(world_size))
    recv_counts = list(torch.zeros([world_size], dtype=torch.int64).chunk(world_size))
    alltoall_cpu(rank, world_size, recv_counts, send_counts)

    #allocate buffers to receive node-ids
    recv_nodes = []
    for i in recv_counts:
        recv_nodes.append(torch.zeros([i.item()], dtype=torch.int64))

    #form the outgoing message
    send_nodes = []
    for i in range(world_size):
        send_nodes.append(torch.Tensor(global_nids_ranks[i]).type(dtype=torch.int64))

    #send-recieve messages
    alltoallv_cpu(rank, world_size, recv_nodes, send_nodes)

    #TODO: This code is not needed, and will be same as the sizes of the sent global node ids
    # in the very first exchange. Please remove this piece of code. 
    # for each of the received orig-node-id requests lookup and send out the global node id
    send_sizes = [len(x.tolist()) for x in recv_nodes]
    send_counts = list(torch.Tensor(send_sizes).type(dtype=torch.int64).chunk(world_size))
    recv_counts = list(torch.zeros([world_size], dtype=torch.int64).chunk(world_size))
    alltoall_cpu( rank, world_size, recv_counts, send_counts)

    # allocate buffers to receive global-ids
    recv_shuffle_global_nids = []
    for i in recv_counts:
        recv_shuffle_global_nids.append(torch.zeros([i.item()], dtype=torch.int64))

    # Use node_data to lookup global id to send over.
    send_nodes = []
    for proc_i_nodes in recv_nodes:
        #list of node-ids to lookup
        global_nids = proc_i_nodes.numpy()
        if (len(global_nids) != 0):
            common, ind1, ind2 = np.intersect1d(node_data[:,3], global_nids, return_indices=True)
            values = node_data[ind1,0]
            send_nodes.append(torch.Tensor(values).type(dtype=torch.int64))
        else:
            send_nodes.append(torch.Tensor(np.empty(shape=(0,))).type(dtype=torch.int64))

    #send receive global-ids
    alltoallv_cpu(rank, world_size, recv_shuffle_global_nids, send_nodes)

    #form the lists with global-ids and orig-node-ids
    #recv_shuffle_global_nids = [x.tolist() for x in recv_shuffle_global_nids]
    #shuffle_global_nids = list(itertools.chain(*recv_shuffle_global_nids))
    #global_nids = list(itertools.chain(*global_nids_ranks))
    #return shuffle_global_nids, global_nids 

    shuffle_global_nids = [x.numpy() for x in recv_shuffle_global_nids]
    global_nids = [x for x in global_nids_ranks]
    return np.column_stack((np.concatenate(global_nids), np.concatenate(shuffle_global_nids)))


def get_shuffle_global_nids_edges(rank, world_size, edge_data, node_part_ids, node_data):
    """
    Edges which are owned by this rank, may have global_nids whose shuffle_global_nids are NOT present locally.
    This function retrieves shuffle_global_nids for such global_nids.

    Parameters: 
    -----------
    rank : integer
        rank of the process
    world_size : integer
        total no. of processes used
    edge_data : numpy ndarray
        edge_data (augmented) as read from the xxx_edges.txt file
    node_part_ids : numpy array 
        list of partition ids indexed by global node ids.
    node_data : numpy ndarray
        node_data (augmented) as read from xxx_nodes.txt file

    Returns:
    --------
    numpy ndarray
        edge_data, with two new columns (shuffle_global_src_id, shuffle_global_dst_id) which are global ids after data shuffling
    """

    #determine unique node-ids present locally
    global_nids = np.sort(np.unique(np.concatenate([edge_data[:, 0], edge_data[:, 1], node_data[:, 3]])))

    #determine the rank which owns orig-node-id <-> partition/rank mappings
    #part_ids = np.array(list(map(list, parts_map.items())))
    #commons, ind1, ind2 = np.intersect1d(part_ids[:,0], global_nids, return_indices=True)
    #part_ids = part_ids[ ind1,:]
    part_ids = node_part_ids[global_nids]

    #form list of lists, each list includes global_nids whose mappings (shuffle_global_nids) needs to be retrieved.
    #and rank will be the process which owns mappings of these global_nids
    global_nids_ranks = []
    for i in range(world_size):
        if (i == rank):
            global_nids_ranks.append(np.empty(shape=(0)))
            continue

        #not_owned_nodes = part_ids[:,0][part_ids[:,1] == i]
        not_owned_node_ids = np.where(part_ids == i)[0] 
        if not_owned_node_ids.shape[0] == 0: 
            not_owned_nodes = np.empty(shape=(0))
        else: 
            not_owned_nodes = global_nids[not_owned_node_ids]
        global_nids_ranks.append(not_owned_nodes)

    #Retrieve Global-ids for respective node owners
    #shuffle_global_nids, global_nids_ranks = get_shuffle_global_nids(rank, world_size, global_nids_ranks, node_data)
    resolved_global_nids = get_shuffle_global_nids(rank, world_size, global_nids_ranks, node_data)

    #Add global_nid <-> shuffle_global_nid mappings to the received data
    for i in range(world_size):
        if (i == rank):
            #own_nodeids = part_ids[:,0][part_ids[:,1] == i]
            own_node_ids = np.where(part_ids == i)[0]
            own_global_nids = global_nids[own_node_ids]
            common, ind1, ind2 = np.intersect1d(node_data[:,3], own_global_nids, return_indices=True)
            my_shuffle_global_nids = node_data[ind1,0]
            #shuffle_global_nids.extend(my_shuffle_global_nids.tolist())
            #global_nids_ranks.extend(own_global_nids[ind2].tolist())
            local_mappings = np.column_stack((own_global_nids, my_shuffle_global_nids))
            resolved_global_nids = np.concatenate((resolved_global_nids, local_mappings))

    #form a dictionary of mappings between orig-node-ids and global-ids
    resolved_mappings = dict(zip(resolved_global_nids[:,0], resolved_global_nids[:,1]))

    #determine global-ids for the orig-src-id and orig-dst-id
    shuffle_global_src_id = [ resolved_mappings[ x ] for x in edge_data[:, 0] ]
    shuffle_global_dst_id = [ resolved_mappings[ x ] for x in edge_data[:, 1] ]

    return np.c_[np.asarray(shuffle_global_src_id, dtype=np.int64), np.c_[ np.asarray(shuffle_global_dst_id, dtype=np.int64), edge_data]]
    #return np.c_[shuffle_global_src_id, np.c_[ shuffle_global_dst_id, edge_data]]


def assign_shuffle_global_nids_nodes(rank, world_size, ntype_counts, node_data):
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
    node_data : numpy ndarray
        node_data, as read from the graph input file (with added additional columns)

    Returns:
    --------
    numpy ndarray
        node_data, as received in the input arguments, and with one additional column added which
        is the locally assigned node-id after data shuffling for its node type
    integer
        this integer indicates the starting id from which global ids are allocated in the current rank
    """
    # sort the list of tuples by nodeType
    ntype_counts.sort(key=lambda x: x[0])

    #add all types of nodes at the current rank
    idx0 = operator.itemgetter(1)
    local_node_count = sum(list(map(idx0, ntype_counts)))

    # Compute prefix sum to determine node-id offsets
    prefix_sum_nodes = allgather_sizes([local_node_count], world_size)

    # assigning node-ids from localNodeStartId to (localNodeEndId - 1)
    # Assuming here that the nodeDataArr is sorted based on the nodeType.
    shuffle_global_nid_start = prefix_sum_nodes[rank]
    shuffle_global_nid_end = prefix_sum_nodes[rank + 1]

    # add a column with global-ids (after data shuffle)
    shuffle_global_nids = np.arange(shuffle_global_nid_start, shuffle_global_nid_end, dtype=np.int64)
    augmted_node_data = np.c_[shuffle_global_nids, node_data]

    #Add a new column, which will mimic the orgi_type_nid, but locally
    #This column will be used to index into the node-features.
    #nodeDataArr is already sorted based on ntype
    part_local_type_nid = []
    for x in ntype_counts:
        ntype = x[0]
        ntype_count = x[1]
        part_local_type_nid.extend([i for i in range(ntype_count)])

    #Add this column to the node_data
    return np.c_[augmted_node_data, np.asarray(part_local_type_nid, dtype=np.int64)], shuffle_global_nid_start


def assign_shuffle_global_nids_edges(rank, world_size, etype_counts, edge_data):
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
    numpy ndarray
        edge_data, as received in the arguments, with one additional column indicating the locally 
        assigned edge id (part_local_type_eid)
    integer
        shuffle_global_eid_start, which indicates the starting value from which shuffle_global-ids are assigned to edges
        on this rank
    """
    #sort the list of tuples by edgeType
    etype_counts.sort(key=lambda x: x[0])

    #compute total no. of edges  
    idx0 = operator.itemgetter(1)
    local_edge_count = sum(list(map(idx0, etype_counts)))
    
    #get prefix sum of edge counts per rank to locate the starting point
    #from which global-ids to edges are assigned in the current rank
    prefix_sum_edges = allgather_sizes([local_edge_count], world_size)
    shuffle_global_eid_start = prefix_sum_edges[rank]
    shuffle_global_eid_end = prefix_sum_edges[rank + 1]

    # assigning edge-ids from localEdgeStart to (localEdgeEndId - 1)
    # Assuming here that the edge_data is sorted by edge_type
    shuffle_global_eids = np.arange(shuffle_global_eid_start, shuffle_global_eid_end, dtype=np.int64)
    return np.c_[ edge_data, shuffle_global_eids], shuffle_global_eid_start
