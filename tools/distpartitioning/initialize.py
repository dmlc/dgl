import os
import sys
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from timeit import default_timer as timer
from datetime import timedelta
from utils import augment_node_data, augment_edge_data,\
                  read_nodes_file, read_edges_file,\
                  read_node_features_file, read_edge_features_file,\
                  read_metis_partitions, read_json,\
                  get_node_types, write_metadata_json
from globalids import assign_shuffle_global_nids_nodes, assign_shuffle_global_nids_edges,\
                      get_shuffle_global_nids_edges
from gloo_wrapper import gather_metadata_json
from convert_partition import create_dgl_object

def send_node_data(rank, node_data, part_ids):
    """ Function to send node_data to non-rank-0 processes. 
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
        send_data = (node_data[:, 7] == part_id) 
        idx = send_data.reshape(node_data.shape[0])
        filt_send_data = node_data[:,[0,5,6]][idx == 1] # send ntype, global_type_nid, global_nid

        #prepare tensor to send
        send_size = filt_send_data.shape
        #TODO: check if send_tensor = th.tensor(filt_send_data.shape, dtype=th.int64)
        send_tensor = torch.zeros(len(send_size), dtype=torch.int64)
        for idx in range(len(send_size)): 
            send_tensor[idx] = send_size[idx]

        # Send size first, so that the part-id (rank)
        # can create appropriately sized buffers
        dist.send(send_tensor, dst=part_id)

        #send actual node_data to part-id rank
        start = timer()
        send_tensor = torch.from_numpy(filt_send_data.astype(np.int64))
        dist.send(send_tensor, dst=part_id)
        end = timer()
        print('Rank: ', rank, ' Sent data size: ', filtered_send_data.shape, \
                ', to Process: ', part_id, 'in: ', timedelta(seconds = end - start))

def send_edge_data(rank, edge_data, part_ids): 
    """ Function to send edge data to non-rank-0 processes

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
        send_data = (edge_data[:, 4] == part_id) 
        idx = send_data.reshape(edge_data.shape[0])
        filt_send_data = edge_data[:,[0,1,2,3]][idx == 1]

        #send shape
        send_size = filt_send_data.shape
        #TODO: test send_tensor = th.tensor(filt_send_data.shape, dtype=th.int64)
        send_tensor = torch.zeros(len(send_size), dtype=torch.int64)
        for idx in range(len(send_size)): 
            send_tensor[idx] = send_size[idx]

        # Send size first, so that the rProc can create appropriately sized tensor
        dist.send(send_tensor, dst=part_id)

        start = timer()
        #TODO: remove type-casting again. data is already in np.int64 form.
        send_tensor = torch.from_numpy(filt_send_data.astype(np.int64))
        dist.send(send_tensor, dst=part_id)
        end = timer()

        print('Rank: ', rank, ' Time to send Edges to proc: ', part_id, \
                ' is : ', timedelta(seconds = end - start))

def send_node_features(rank, node_data, node_features, part_ids, ntype_map):
    """ Function to send node_features data to non-rank-0 processes.
    Parameters:
    -----------
    rank : integer
        rank of the process
    node_features : numpy ndarray of floats 
        node_features, data from the node_feats.dgl
    ranks_list : list of unique ranks/partition-ids
    ntype_map : dictionary of mappings between ntype_name -> ntype
    """

    node_features_out = []
    for part_id in part_ids: 
        if part_id == rank: 
            node_features_out.append(None)
            continue

        part_node_features = {}
        #for x in ntype_map.items(): 
        #    ntype_name = x[0]
        #    ntype = x[1]
        for ntype_name, ntype in ntype_map.items():
            
            #extract orig_type_node_id
            idx = (node_data[:,7] == part_id) & (node_data[:,0] == ntype) 
            filt_global_ntype_ids = node_data[:,[5]][idx] # extract global_ntype_id here
            filt_global_ntype_ids = np.concatenate(filt_global_ntype_ids) 

            if (ntype_name +'/feat' in node_features) and (node_features[ntype_name+'/feat'].shape[0] > 0): 
                part_node_features[ntype_name+'/feat'] = node_features[ntype_name+'/feat'][filt_global_ntype_ids]
            #else: 
            #    send_node_features[node_type_name+'/feat'] = None

        #accumulate subset of node_features targetted for part-id rank
        node_features_out.append(part_node_features)

    #send data
    output_list = [None]
    start = timer ()
    dist.scatter_object_list(output_list, node_features_out, src=0)
    end = timer ()

    print('Rank: ', rank, ', Done sending Node Features to: ', part_id, \
            ' in: ', timedelta(seconds = end - start))

def send_data(rank, node_data, node_features, edge_data, metis_partitions, ntypes_map): 
    """ Wrapper function to send graph data to non-rank-0 processes.
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
    metis_partitions : dictionary
        orig_node_id -> partition_id mappings as defined by METIS
    ntype_map : dictionary 
        mappings between ntype_name -> ntype_id
    """

    part_ids = np.unique(list(metis_partitions.values()))
    part_ids.sort ()
    print('Rank: ', rank, ', Unique partitions: ', part_ids)

    send_node_data(rank, node_data, part_ids)
    send_edge_data(rank, edge_data, part_ids) 
    send_node_features(rank, node_data, node_features, part_ids, ntypes_map)

def recv_data(rank, shape, dtype): 
    """ Auxiliary function to receive a multi-dimensional tensor, used by the
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
    """ Function to receive node_data, used by non-rank-0 processes.
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
    """Function to receive edge_data, used by non-rank0 processes. 
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
    """Function to receive node_feautres as an object, as read from the node_feats.dgl file.
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

def read_graph_files(rank, params, metis_partitions): 
    """Read the files and return the data structures
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
    metis_partisions : dictionary
        global_node_id -> partition_id/rank mappings as determined by METIS
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
    augmted_node_data = []
    node_data = read_nodes_file(params.input_dir+'/'+params.nodes_file)
    augmted_node_data = augment_node_data(node_data, metis_partitions)
    print('Rank: ', rank, ', Completed loading nodes data: ', augmted_node_data.shape)

    augmted_edge_data = []
    edge_data = read_edges_file(params.input_dir+'/'+params.edges_file)
    removed_edge_data = read_edges_file(params.input_dir+'/'+params.removed_edges)
    edge_data = np.vstack((edge_data, removed_edge_data))
    augmted_edge_data = augment_edge_data(edge_data, metis_partitions)
    print('Rank: ', rank, ', Completed loading edges data: ', augmted_edge_data.shape)

    node_features = []
    node_features = read_node_features_file( params.input_dir+'/'+params.node_feats_file )
    print('Rank: ', rank, ', Completed loading node features reading from file ', len(node_features))

    edge_features = []
    #edge_features = read_edge_features_file( params.input_dir+'/'+params.edge_feats_file )
    #print( 'Rank: ', rank, ', Completed edge features reading from file ', len(edge_features) )

    return augmted_node_data, node_features, augmted_edge_data, edge_features

def proc_exec(rank, world_size, params):
    """ `main` function for each rank in the distributed implementation.
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
    metis_partitions = read_metis_partitions(params.input_dir+'/'+params.metis_partitions)
    print('Rank: ', rank, ', Completed loading metis partitions: ', len(metis_partitions))

    #read graph schema, get ntype_map(dict for ntype to ntype-id lookups) and ntypes list
    schema_map = read_json(params.input_dir+'/'+params.schema)
    ntypes_map, ntypes = get_node_types(schema_map)

    if rank == 0: 
        #read input graph files
        node_data, node_features, edge_data, edge_features = read_graph_files(rank, params, metis_partitions)

        # order node_data by node_type before extracting node features. 
        # once this is ordered, node_features are automatically ordered and 
        # can be assigned contiguous ids starting from 0 for each type. 
        node_data = node_data[node_data[:, 0].argsort()]

        print('Rank: ', rank, ', node_data: ', node_data.shape)
        print('Rank: ', rank, ', node_features: ', len(node_features))
        print('Rank: ', rank, ', edge_data: ', edge_data.shape)
        #print('Rank: ', rank, ', edge_features : ',len( edge_features))
        print('Rank: ', rank, ', partitions : ', len(metis_partitions))

        # shuffle data
        send_data(rank, node_data, node_features, edge_data, metis_partitions, ntypes_map)

        #extract features here for rank-0
        for name, ntype_id in ntypes_map.items(): 
            ntype = name + '/feat'
            if(ntype in node_features): 
                idx = node_data[:,5][(node_data[:,0] == ntype_id) & (node_data[:,7] == rank)]
                node_features[ntype] = node_features[ntype][idx]

        # Filter data owned by rank-0
        #extract only ntype, global_type_nid, global_nid 
        node_data = node_data[:,[0,5,6]][node_data[:,7] == 0] 

        #extract only global_src_id, global_dst_id, global_type_eid etype
        edge_data = edge_data[:,[0,1,2,3]][edge_data[:,4] == 0] 

    else: 
        #non-rank-0 processes receive data from rank-0
        node_data = recv_node_data(rank, 2, torch.int64)
        edge_data = recv_edge_data(rank, 2, torch.int64)
        node_features = recv_node_features_obj(rank, world_size)

    #syncronize
    dist.barrier()

    #determine ntypes present in the graph
    ntypes = np.unique(node_data[:,0])
    ntypes.sort ()

    #for a list of tuples (ntype, count)
    ntype_counts = []
    bins = np.bincount(node_data[:,0])
    for ntype in ntypes: 
        ntype_counts.append((ntype, bins[ntype]))
        
    # after this call node_data changes to [globalId, node_type, orig_node_type_id, orig_node_id, local_type_id]
    # note that orig_node_id is the line no. of a node in the file xxx_nodes.txt
    node_data, shuffle_global_nid_start = assign_shuffle_global_nids_nodes(rank, world_size, ntype_counts, node_data)
    print('Rank: ', rank, ' Done assign Global ids to nodes...')

    #Work on the edge and assign GlobalIds
    etypes = np.unique(edge_data[:,3])
    etypes.sort()
    
    #sort edge_data by etype
    edge_data = edge_data[edge_data[:,3].argsort()]

    etype_counts = []
    bins = np.bincount(edge_data[:,3])
    for etype in etypes: 
        etype_counts.append((etype, bins[etype]) )

    edge_data, shuffle_global_eid_start = assign_shuffle_global_nids_edges(rank, world_size, etype_counts, edge_data)
    print('Rank: ', rank, ' Done assign Global ids to edges...')

    edge_data = get_shuffle_globalids_edges(rank, world_size, edge_data, metis_partitions, node_data)
    print('Rank: ', rank, ' Done retrieving Global Node Ids for non-local nodes... ')

    #call convert_parititio.py for serialization 
    print('Rank: ', rank, ' Creating DGL objects for all partitions')
    #json_metadata, output_dir, graph_name = gen_dgl_objs(False, pipeline_args, params)
    num_nodes = 0
    num_edges = 0
    ntypes_map = None
    ntypes_map_val = None
    etypes_map = None
    etypes_map_val = None

    #create dgl objects
    create_dgl_object(params.input_dir, params.graph_name, params.num_parts, params.num_node_weights, params.node_attr_dtype, \
                            params.edge_attr_dtype, params.workspace_dir, params.output_dir, params.removed_edges, params.schema, \
                            rank, node_data, node_features, edge_data, edge_features, num_nodes, num_edges, \
                            ntypes_map_val, etypes_map_val, ntypes_map, etypes_map)
    #get the meta-data 
    json_metadata = create_metadata_json(params.graph_name, num_nodes, num_edges, params.num_parts, ntypes_map_val, \
                            etypes_map_val, ntypes_map, etypes_map, params.output_dir, False)

    if (rank == 0): 
        #get meta-data from all partitions and merge them on rank-0
        metadata_list = gather_metadata_json(json_metadata, rank, world_size)
        metadata_list[0] = json_metadata
        write_metadata_json(metadata_list, output_dir, graph_name)
    else: 
        #send meta-data to Rank-0 process
        gather_metadata_json(json_metadata, rank, world_size)

def init_process(rank, world_size, proc_exec, params, backend="gloo"):
    """Init. function which is run by each process in the Gloo ProcessGroup
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
    proc_exec(rank, world_size, params)
