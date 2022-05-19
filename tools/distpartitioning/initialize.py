import os
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import constants

from timeit import default_timer as timer
from datetime import timedelta
from utils import augment_node_data, augment_edge_data,\
                  read_nodes_file, read_edges_file,\
                  read_node_features_file, read_edge_features_file,\
                  read_partitions_file, read_json,\
                  get_node_types, write_metadata_json, write_dgl_objects
from globalids import assign_shuffle_global_nids_nodes, assign_shuffle_global_nids_edges,\
                      get_shuffle_global_nids_edges
from gloo_wrapper import gather_metadata_json
from convert_partition import create_dgl_object, create_metadata_json

def send_node_data(rank, node_data, part_ids):
    """ 
    Function to send node_data to non-rank-0 processes. 
    
    Parameters:
    -----------
    rank : integer
        rank of the process
    node_data : numpy ndarray
        node_data, in the augmented form
    part_ids : python list 
        list of unique ranks/partition-ids
    """

    for part_id in part_ids: 
        if part_id == rank: 
            continue
        
        #extract <node_type>, <global_type_nid>, <global_nid>
        #which belong to `part_id`
        send_data_idx = (node_data[constants.OWNER_PROCESS] == part_id) 
        idx = send_data_idx.reshape(node_data[constants.GLOBAL_NID].shape[0])
        filt_data = np.column_stack((node_data[constants.NTYPE_ID], \
                                    node_data[constants.GLOBAL_TYPE_NID], \
                                    node_data[constants.GLOBAL_NID]))
        filt_data = filt_data[idx == 1]

        #prepare tensor to send
        send_size = filt_data.shape
        size_tensor = torch.tensor(filt_data.shape, dtype=torch.int64)

        # Send size first, so that the part-id (rank)
        # can create appropriately sized buffers
        dist.send(size_tensor, dst=part_id)

        #send actual node_data to part-id rank
        start = timer()
        send_tensor = torch.from_numpy(filt_data.astype(np.int64))
        dist.send(send_tensor, dst=part_id)
        end = timer()
        print('Rank: ', rank, ' Sent data size: ', filt_data.shape, \
                ', to Process: ', part_id, 'in: ', timedelta(seconds = end - start))

def send_edge_data(rank, edge_data, part_ids): 
    """ 
    Function to send edge data to non-rank-0 processes

    Parameters:
    -----------
    rank : integer
        rank of the process
    edge_data : numpy ndarray
        edge_data, in the augmented form
    part_ids : python list
        list of unique ranks/partition-ids
    """
    for part_id in part_ids: 
        if part_id == rank: 
            continue

        #extract global_sid, global_dit, global_type_eid, etype_id
        send_data = (edge_data[constants.OWNER_PROCESS] == part_id) 
        idx = send_data.reshape(edge_data[constants.GLOBAL_SRC_ID].shape[0])
        filt_data = np.column_stack((edge_data[constants.GLOBAL_SRC_ID][idx == 1], \
                                    edge_data[constants.GLOBAL_DST_ID][idx == 1], \
                                    edge_data[constants.GLOBAL_TYPE_EID][idx == 1], \
                                    edge_data[constants.ETYPE_ID][idx == 1]))

        #send shape
        send_size = filt_data.shape
        size_tensor = torch.tensor(filt_data.shape, dtype=torch.int64)

        # Send size first, so that the rProc can create appropriately sized tensor
        dist.send(size_tensor, dst=part_id)

        start = timer()
        send_tensor = torch.from_numpy(filt_data)
        dist.send(send_tensor, dst=part_id)
        end = timer()

        print('Rank: ', rank, ' Time to send Edges to proc: ', part_id, \
                ' is : ', timedelta(seconds = end - start))

def send_node_features(rank, node_data, node_features, part_ids, ntype_map):
    """ 
    Function to send node_features data to non-rank-0 processes.
    
    Parameters:
    -----------
    rank : integer
        rank of the process
    node_data : numpy ndarray of int64
        node_data, read from the xxx_nodes.txt file
    node_features : numpy ndarray of floats 
        node_features, data from the node_feats.dgl
    part_ids : list
        list of unique ranks/partition-ids
    ntype_map : dictionary 
        mappings between ntype_name -> ntype
    """

    node_features_out = []
    for part_id in part_ids: 
        if part_id == rank: 
            node_features_out.append(None)
            continue

        part_node_features = {}
        for ntype_name, ntype in ntype_map.items():
            
            if (ntype_name +'/feat' in node_features) and (node_features[ntype_name+'/feat'].shape[0] > 0): 
                #extract orig_type_node_id
                idx = (node_data[constants.OWNER_PROCESS] == part_id) & (node_data[constants.NTYPE_ID] == ntype) 
                filt_global_type_nids = node_data[constants.GLOBAL_TYPE_NID][idx] # extract global_ntype_id here
                part_node_features[ntype_name+'/feat'] = node_features[ntype_name+'/feat'][filt_global_type_nids]

        #accumulate subset of node_features targetted for part-id rank
        node_features_out.append(part_node_features)

    #send data
    output_list = [None]
    start = timer ()
    dist.scatter_object_list(output_list, node_features_out, src=0)
    end = timer ()
    print('Rank: ', rank, ', Done sending Node Features to: ', part_id, \
            ' in: ', timedelta(seconds = end - start))

def send_data(rank, node_data, node_features, edge_data, node_part_ids, ntypes_map): 
    """ 
    Wrapper function to send graph data to non-rank-0 processes.
    
    Parameters:
    -----------
    rank : integer
        rank of the process
    node_data : numpy ndarray 
        node_data, augmented, from xxx_nodes.txt file
    node_features : numpy ndarray 
        node_features, data from the node_feats.dgl
    edge_data : numpy ndarray
        edge_data, augmented, from xxx_edges.txt file
    node_part_ids : ndarray  
        array of part_ids indexed by global_nid
    ntype_map : dictionary 
        mappings between ntype_name -> ntype_id
    """

    part_ids = np.unique(node_part_ids)
    part_ids.sort ()
    print('Rank: ', rank, ', Unique partitions: ', part_ids)

    send_node_data(rank, node_data, part_ids)
    send_edge_data(rank, edge_data, part_ids) 
    send_node_features(rank, node_data, node_features, part_ids, ntypes_map)

def recv_data(rank, shape, dtype): 
    """ 
    Auxiliary function to receive a multi-dimensional tensor, used by the
    non-rank-0 processes. 

    Parameters:
    -----------
    rank : integer
        rank of the process
    shape : tuple of integers
        shape of the received data
    dtype : integer
        type of the received data

    Returns: 
    --------
    numpy array
        received data after completing 'recv' gloo primitive.
    """

    #First receive the size of the data to be received from rank-0 process
    recv_tensor_shape = torch.zeros(shape, dtype=torch.int64)
    dist.recv(recv_tensor_shape, src=0)
    recv_shape = list(map(lambda x: int(x), recv_tensor_shape))

    #Receive the data message here for nodes here. 
    recv_tensor_data = torch.zeros(recv_shape, dtype=dtype)
    dist.recv(recv_tensor_data, src=0)
    return recv_tensor_data.numpy()

def recv_node_data(rank, shape, dtype): 
    """ 
    Function to receive node_data, used by non-rank-0 processes.

    Parameters:
    -----------
    rank : integer
        rank of the process
    shape : tuple of integers 
        shape of the received data
    dtype : integer
        type of the received data

    Returns:
    --------
    numpy array
        result of the 'recv' gloo primitive
    """
    return recv_data(rank, shape, dtype)

def recv_edge_data(rank, shape, dtype): 
    """
    Function to receive edge_data, used by non-rank0 processes. 

    Parameters:
    -----------
    rank : integer
        rank of the process
    shape : tuple of integers
        shape of the received data
    dtype : integer
        type of the received data

    Returns:
    --------
    numpy array
        result of the 'recv' operation
    """
    return recv_data(rank, shape, dtype)

def recv_node_features_obj(rank, world_size): 
    """
    Function to receive node_feautres as an object, as read from the node_feats.dgl file.
    This is used by non-rank-0 processes. 

    Parameters:
    -----------
    rank : integer
        rank of the process
    world_size : integer
        no. of processes used

    Returns:
    --------
    numpy ndarray
        node_feature data, of floats, as received by the scatter_object_list function
    """
    send_objs = [None for _ in range(world_size)]
    recv_obj = [None]
    dist.scatter_object_list(recv_obj, send_objs, src=0)

    node_features = recv_obj[0]
    return node_features

def read_graph_files(rank, params, node_part_ids): 
    """
    Read the files and return the data structures
    Node data as read from files, which is in the following format: 
        <node_type> <weight1> <weight2> <weight3> <weight4> <global_type_nid> <attributes>
    is converted to 
        <node_type> <weight1> <weight2> <weight3> <weight4> <global_type_nid> <nid> <recv_proc>
    Edge data as read from files, which is in the following format: 
        <global_src_id> <global_dst_id> <global_type_eid> <edge_type> <attributes>
    is converted to the following format in this function:
        <global_src_id> <global_dst_id> <global_type_eid> <edge_type> <recv_proc>

    Parameters:
    -----------
    rank : integer
        rank of the process
    params : argparser object
        argument parser data structure to access command line arguments
    node_part_ids : numpy array 
        array of part_ids indexed by global_nid

    Returns:
    --------
    numpy ndarray
        integer node_data with additional columns added
    numpy ndarray
        floats, node_features as read from the node features file
    numpy ndarray
        integer, edge_data with additional columns added
    numpy ndarray
        floats, edge_features are read from the edge feature file
    """
    node_data = read_nodes_file(params.input_dir+'/'+params.nodes_file)
    augment_node_data(node_data, node_part_ids)
    print('Rank: ', rank, ', Completed loading nodes data: ', node_data[constants.GLOBAL_TYPE_NID].shape)

    edge_data = read_edges_file(params.input_dir+'/'+params.edges_file, None)
    print('Rank: ', rank, ', Completed loading edge data: ', edge_data[constants.GLOBAL_SRC_ID].shape)
    edge_data = read_edges_file(params.input_dir+'/'+params.removed_edges, edge_data)
    augment_edge_data(edge_data, node_part_ids)
    print('Rank: ', rank, ', Completed adding removed edges : ', edge_data[constants.GLOBAL_SRC_ID].shape)

    node_features = {}
    node_features = read_node_features_file( params.input_dir+'/'+params.node_feats_file )
    print('Rank: ', rank, ', Completed loading node features reading from file ', len(node_features))

    edge_features = {}
    #edge_features = read_edge_features_file( params.input_dir+'/'+params.edge_feats_file )
    #print( 'Rank: ', rank, ', Completed edge features reading from file ', len(edge_features) )

    return node_data, node_features, edge_data, edge_features

def proc_exec(rank, world_size, params):
    """ 
    `main` function for each rank in the distributed implementation.

    This function is used when one-machine is used for executing the entire pipeline. 
    In this case, all the gloo-processes will exist on the same machine. Also, this function
    expects that the graph input files are in single file-format. Nodes, edges, node-features and 
    edge features each will have their own file describing the appropriate parts of the input graph.

    Parameters: 
    -----------
    rank : integer
        rank of the current process
    world_size : integer
        total no. of ranks
    params : argparser object
        argument parser structure to access values passed from command line
    """

    #Read METIS partitions
    node_part_ids = read_partitions_file(params.input_dir+'/'+params.partitions_file)
    print('Rank: ', rank, ', Completed loading metis partitions: ', len(node_part_ids))

    #read graph schema, get ntype_map(dict for ntype to ntype-id lookups) and ntypes list
    schema_map = read_json(params.input_dir+'/'+params.schema)
    ntypes_map, ntypes = get_node_types(schema_map)

    # Rank-0 process will read the graph input files (nodes, edges, node-features and edge-features). 
    # it will uses metis partitions, node-id to partition-id mappings, to determine the node and edge
    # ownership and sends out data to all the other non rank-0 processes.
    if rank == 0: 
        #read input graph files
        node_data, node_features, edge_data, edge_features = read_graph_files(rank, params, node_part_ids)

        # order node_data by node_type before extracting node features. 
        # once this is ordered, node_features are automatically ordered and 
        # can be assigned contiguous ids starting from 0 for each type. 
        #node_data = node_data[node_data[:, 0].argsort()]
        sorted_idx = node_data[constants.NTYPE_ID].argsort()
        for k, v in node_data.items(): 
            node_data[k] = v[sorted_idx]

        print('Rank: ', rank, ', node_data: ', len(node_data))
        print('Rank: ', rank, ', node_features: ', len(node_features))
        print('Rank: ', rank, ', edge_data: ', len(edge_data))
        #print('Rank: ', rank, ', edge_features : ',len( edge_features))
        print('Rank: ', rank, ', partitions : ', len(node_part_ids))

        # shuffle data
        send_data(rank, node_data, node_features, edge_data, node_part_ids, ntypes_map)

        #extract features here for rank-0
        for name, ntype_id in ntypes_map.items(): 
            ntype = name + '/feat'
            if(ntype in node_features): 
                idx = node_data[constants.GLOBAL_TYPE_NID][(node_data[constants.NTYPE_ID] == ntype_id) & (node_data[constants.OWNER_PROCESS] == rank)]
                node_features[ntype] = node_features[ntype][idx]

        # Filter data owned by rank-0
        #extract only ntype, global_type_nid, global_nid 
        idx = np.where(node_data[constants.OWNER_PROCESS] == 0)[0]
        for k, v in node_data.items(): 
            node_data[k] = v[idx]

        #extract only global_src_id, global_dst_id, global_type_eid etype
        idx = np.where(edge_data[constants.OWNER_PROCESS] == 0)[0]
        for k, v in edge_data.items(): 
            edge_data[k] = v[idx]
    else: 
        #Non-rank-0 processes, receives nodes, edges, node-features and edge feautres from rank-0
        # process and creates appropriate data structures. 
        rcvd_node_data = recv_node_data(rank, 2, torch.int64)
        node_data = {}
        node_data[constants.NTYPE_ID] = rcvd_node_data[:,0]
        node_data[constants.GLOBAL_TYPE_NID] = rcvd_node_data[:,1]
        node_data[constants.GLOBAL_NID] = rcvd_node_data[:,2]

        rcvd_edge_data = recv_edge_data(rank, 2, torch.int64)
        edge_data = {}
        edge_data[constants.GLOBAL_SRC_ID] = rcvd_edge_data[:,0]
        edge_data[constants.GLOBAL_DST_ID] = rcvd_edge_data[:,1]
        edge_data[constants.GLOBAL_TYPE_EID] = rcvd_edge_data[:,2]
        edge_data[constants.ETYPE_ID] = rcvd_edge_data[:,3]

        node_features = recv_node_features_obj(rank, world_size)
        edge_features = {}

    # From this point onwards, all the processes will follow the same execution logic and
    # process the data which is owned by the current process. 
    # At this time, all the processes will have all the data which it owns (nodes, edges, 
    # node-features and edge-features).

    #syncronize
    dist.barrier()

    # assign shuffle_global ids to nodes
    assign_shuffle_global_nids_nodes(rank, world_size, node_data)
    print('Rank: ', rank, ' Done assign Global ids to nodes...')

    #sort edge_data by etype
    sorted_idx = edge_data[constants.ETYPE_ID].argsort()
    for k, v in edge_data.items(): 
        edge_data[k] = v[sorted_idx]

    # assign shuffle_global ids to edges
    shuffle_global_eid_start = assign_shuffle_global_nids_edges(rank, world_size, edge_data)
    print('Rank: ', rank, ' Done assign Global ids to edges...')

    # resolve shuffle_global ids for nodes which are not locally owned
    get_shuffle_global_nids_edges(rank, world_size, edge_data, node_part_ids, node_data)
    print('Rank: ', rank, ' Done retrieving Global Node Ids for non-local nodes... ')

    #create dgl objects
    print('Rank: ', rank, ' Creating DGL objects for all partitions')
    num_nodes = 0
    num_edges = shuffle_global_eid_start
    with open('{}/{}'.format(params.input_dir, params.schema)) as json_file: 
        schema = json.load(json_file)
    graph_obj, ntypes_map_val, etypes_map_val, ntypes_map, etypes_map = create_dgl_object(\
            params.graph_name, params.num_parts, \
            schema, rank, node_data, edge_data, num_nodes, num_edges)
    write_dgl_objects(graph_obj, node_features, edge_features, params.output, rank)

    #get the meta-data 
    json_metadata = create_metadata_json(params.graph_name, num_nodes, num_edges, params.num_parts, ntypes_map_val, \
                            etypes_map_val, ntypes_map, etypes_map, params.output)

    if (rank == 0): 
        #get meta-data from all partitions and merge them on rank-0
        metadata_list = gather_metadata_json(json_metadata, rank, world_size)
        metadata_list[0] = json_metadata
        write_metadata_json(metadata_list, params.output, params.graph_name)
    else: 
        #send meta-data to Rank-0 process
        gather_metadata_json(json_metadata, rank, world_size)

def single_dev_init(rank, world_size, func_exec, params, backend="gloo"):
    """
    Init. function which is run by each process in the Gloo ProcessGroup

    Parameters:
    -----------
    rank : integer
        rank of the process
    world_size : integer
        number of processes configured in the Process Group
    proc_exec : function name
        function which will be invoked which has the logic for each process in the group
    params : argparser object
        argument parser object to access the command line arguments
    backend : string
        string specifying the type of backend to use for communication
    """
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '29500'

    #create Gloo Process Group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    #Invoke the main function to kick-off each process
    func_exec(rank, world_size, params)

def multi_dev_init(params):
    """
    Function to be invoked when executing data loading pipeline on multiple machines

    Parameters:
    -----------
    params : argparser object
        argparser object providing access to command line arguments.
    """
    #init the gloo process group here. 
    dist.init_prcess_group("gloo", rank=params.rank, world_size=params.world_size)
    print('[Rank: ', params.rank, '] Done with process group initialization...')

    #invoke the main function here.
    proc_exec(params.rank, params.world_size, params)
    print('[Rank: ', params.rank, '] Done with Distributed data processing pipeline processing.')
