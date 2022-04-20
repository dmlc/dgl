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
from globalids import assign_globalids_nodes, assign_globalids_edges,\
                      get_globalids_edges
from gloo_wrapper import gather_metadata_json
from convert_partition import gen_dgl_objs

def send_node_data(rank, node_data, part_list):
    """
    Function to send node_data to the non-rank-0 processes. 

    Parameters:
    -----------
    rank : rank of the process
    node_data : node_data, in the augmented form
    ranks_list : list of unique ranks/partition-ids
    """

    for part_id in part_list: 
        if part_id == rank: 
            continue
        
        #extract <node_type>, <orig_type_id>, <global_id>
        #which belong to `part_id`
        send_data = (node_data[:, 7] == part_id) 
        idx = send_data.reshape(node_data.shape[0])
        filtered_send_data = node_data[:,[0,5,6]][idx == 1] # send node_type, orig_type_id, line_id (global_id)

        #
        send_size = filtered_send_data.shape
        send_tensor = torch.zeros(len(send_size), dtype=torch.int)
        for idx in range(len(send_size)): 
            send_tensor[idx] = send_size[idx]

        # Send size first, so that the part-id (rank)
        # can create appropriately sized buffers
        dist.send(send_tensor, dst=part_id)

        #send actual node_data to part-id rank
        start = timer ()
        send_tensor = torch.from_numpy( filtered_send_data.astype(np.int32) )
        dist.send(send_tensor, dst=part_id)
        end = timer ()
        print( 'Rank: ', rank, ' Sent data size: ', filtered_send_data.shape, ', to Process: ', part_id, 'in: ', timedelta(seconds = end - start) )


def send_edge_data(rank, edge_data, rank_list): 
    """
    Function to send edge data to non-rank-0 processes

    Parameters:
    -----------
    rank : rank of the process
    node_data : node_data, in the augmented form
    ranks_list : list of unique ranks/partition-ids
    """
    for part_id in rank_list: 
        if part_id == rank: 
            continue

        send_data = (edge_data[:, 4] == part_id) 
        idx = send_data.reshape(edge_data.shape[0])
        filtered_send_data = edge_data[:,[0,1,2,3]][idx == 1]

        send_size = filtered_send_data.shape
        send_tensor = torch.zeros(len(send_size), dtype=torch.int32)
        for idx in range(len(send_size)): 
            send_tensor[idx] = send_size[idx]

        # Send size first, so that the rProc can create appropriately sized tensor
        dist.send(send_tensor, dst=part_id)

        start = timer()
        send_tensor = torch.from_numpy(filtered_send_data.astype(np.int32))
        dist.send(send_tensor, dst=part_id)
        end = timer()

        print('Rank: ', rank, ' Time to send Edges to proc: ', part_id, ' is : ', timedelta(seconds = end - start))


def send_node_features(rank, node_data, node_features, rank_list, ntype_map):
    """
    Function to send node_features data to non-rank-0 processes.

    Parameters:
    -----------
    rank : rank of the process
    node_features : node_features, data from the node_feats.dgl
    ranks_list : list of unique ranks/partition-ids
    ntype_map : dictionary of mappings between ntype_name -> ntype_id
    """

    node_features_rank_lst = []
    for part_id in rank_list: 
        if part_id == rank: 
            node_features_rank_lst.append( None )
            continue

        send_node_features = {}
        for x in ntype_map.items(): 
            node_type_name = x[0]
            node_type_id = x[1]
            
            #extract orig_type_node_id
            idx = (node_data[:,7] == part_id) & (node_data[:,0] == node_type_id) 
            filtered_orig_node_type_ids = node_data[:,[5]][idx] # extract orig_type_node_id here
            filtered_orig_node_type_ids = np.concatenate(filtered_orig_node_type_ids) 

            if (node_type_name +'/feat' in node_features) and (node_features[node_type_name+'/feat'].shape[0] > 0): 
                send_node_features[node_type_name+'/feat'] = node_features[node_type_name+'/feat'][filtered_orig_node_type_ids]
            #else: 
            #    send_node_features[node_type_name+'/feat'] = None

        #accumulate subset of node_features targetted for part-id rank
        node_features_rank_lst.append(send_node_features)

    #send data
    output_list = [None]
    start = timer ()
    dist.scatter_object_list(output_list, node_features_rank_lst, src=0)
    end = timer ()

    print( 'Rank: ', rank, ', Done sending Node Features to: ', part_id, ' in: ', timedelta(seconds = end - start))
        

def send_data(rank, node_data, node_features, edge_data, metis_partitions, ntypes_map): 
    """
    Wrapper function to send graph data to non-rank-0 processes.

    Parameters:
    -----------
    rank : rank of the process
    ranks_list : list of unique ranks/partition-ids
    node_data : node_data, augmented, from xxx_nodes.txt file
    node_features : node_features, data from the node_feats.dgl
    edge_data : edge_data, augmented, from xxx_edges.txt file
    metis_partitions : orig_node_id -> partition_id mappings as defined by METIS
    ntype_map : dictionary of mappings between ntype_name -> ntype_id
    """

    partition_ids = np.unique(list(metis_partitions.values()))
    partition_ids.sort ()
    print( 'Rank: ', rank, ', Unique partitions: ', partition_ids)

    send_node_data(rank, node_data, partition_ids)
    send_edge_data(rank, edge_data, partition_ids) 
    send_node_features(rank, node_data, node_features, partition_ids, ntypes_map)


def recv_data(rank, dimensions, dtype): 
    """
    Auxiliary function to receive a multi-dimensional tensor, used by the
    non-rank-0 processes. 

    Parameters:
    -----------
    rank : rank of the process
    dimensions : dimensions of the received data
    dtype : type of the received data

    """

    #First receive the size of the data to be received from rank-0 process
    recv_tensor_shape = torch.zeros(dimensions, dtype = torch.int32)
    dist.recv(recv_tensor_shape, src=0)
    recv_shape = list( map( lambda x: int(x), recv_tensor_shape) )

    #Receive the data message here for nodes here. 
    recv_tensor_data = torch.zeros( recv_shape, dtype=dtype)
    dist.recv( recv_tensor_data, src=0 )
    return recv_tensor_data.numpy ()


def recv_node_data(rank, dimensions, dtype ): 
    """
    Function to receive node_data, used by non-rank-0 processes.

    Parameters:
    -----------
    rank : rank of the process
    dimensions : dimensions of the received data
    dtype : type of the received data
    """
    return recv_data(rank, dimensions, dtype)

def recv_edge_data(rank, dimensions, dtype): 
    """
    Function to receive edge_data, used by non-rank0 processes. 

    Parameters:
    -----------
    rank : rank of the process
    dimensions : dimensions of the received data
    dtype : type of the received data
    """
    return recv_data (rank, dimensions, dtype)

def recv_node_features_obj(rank, world_size): 
    """
    Function to receive node_feautres as an object, as read from the node_feats.dgl file.
    This is used by non-rank-0 processes. 

    Parameters:
    -----------
    rank : rank of the process
    world_size : no. of processes used
    """
    send_objs = [None for _ in range(world_size)]
    recv_obj = [None]
    dist.scatter_object_list(recv_obj, send_objs, src=0)

    node_features = recv_obj[0]
    return node_features


def read_graph_files( rank, params, metis_partitions ): 
    '''
    Read the files and return the data structures
    Node data as read from files, which is in the following format: 
        <node_type> <weight1> <weight2> <weight3> <weight4> <orig_type_node_id> <attributes>
    is converted to 
        <node_type> <weight1> <weight2> <weight3> <weight4> <orig_type_node_id> <nid> <recv_proc>
    Edge data as read from files, which is in the following format: 
        <src_id> <dst_id> <type_edge_id> <edge_type> <attributes>
    is converted to the following format in this function:
        <src_id> <dst_id> <type_edge_id> <edge_type> <recv_proc>
        
    Parameters:
    -----------
    rank : rank of the process
    params : argument parser data structure to access command line arguments
    metis_partisions : orig_node_id -> partition_id/rank mappings as determined by METIS
    '''
    augmted_node_data = []
    node_data = read_nodes_file(params.input_dir+'/'+params.nodes_file)
    augmted_node_data = augment_node_data(node_data, metis_partitions)
    print( 'Rank: ', rank, ', Completed loading nodes data: ', augmted_node_data.shape )

    augmted_edge_data = []
    edge_data = read_edges_file(params.input_dir+'/'+params.edges_file)
    removed_edge_data = read_edges_file(params.input_dir+'/'+params.removed_edges)
    edge_data = np.vstack((edge_data, removed_edge_data))
    augmted_edge_data = augment_edge_data(edge_data, metis_partitions)
    print( 'Rank: ', rank, ', Completed loading edges data: ', augmted_edge_data.shape )

    node_features = []
    node_features = read_node_features_file( params.input_dir+'/'+params.node_feats_file )
    print( 'Rank: ', rank, ', Completed loading node features reading from file ', len(node_features))

    edge_features = []
    #edge_features = read_edge_features_file( params.input_dir+'/'+params.edge_feats_file )
    #print( 'Rank: ', rank, ', Completed edge features reading from file ', len(edge_features) )

    return augmted_node_data, node_features, augmted_edge_data, edge_features

def proc_exec(rank, world_size, params):
    """
    `main` function for each rank in the distributed implementation.

    Parameters: 
    -----------
    rank : rank of the current process
    world_size : total no. of ranks
    params : argument parser structure to access values passed from command line
    """

    #Read METIS partitions
    metis_partitions = read_metis_partitions(params.input_dir+'/'+params.metis_partitions)
    print( 'Rank: ', rank, ', Completed loading metis partitions: ', len(metis_partitions))

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
        #extract only orig_ntype, orig_type_nid, orig_node_id 
        node_data = node_data[:,[0,5,6]][node_data[:,7] == 0] 

        #extract only orig_src_id, orig_dst_id, orig_etype_id orig_etype
        edge_data = edge_data[:,[0,1,2,3]][edge_data[:,4] == 0] 

    else: 
        #non-rank-0 processes receive data from rank-0
        node_data = recv_node_data(rank, 2, torch.int32)
        edge_data = recv_edge_data(rank, 2, torch.int32 )
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
    node_data, node_offset_globalid = assign_globalids_nodes(rank, world_size, ntype_counts, node_data)
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

    edge_data, edge_offset_globalid = assign_globalids_edges(rank, world_size, etype_counts, edge_data)
    print('Rank: ', rank, ' Done assign Global ids to edges...')

    edge_data = get_globalids_edges(rank, world_size, edge_data, metis_partitions, node_data)
    print('Rank: ', rank, ' Done retrieving Global Node Ids for non-local nodes... ')

    #Here use the functionality of the convert partition.py to store the dgl objects
    #for the node_features, edge_features and graph itself. 
    #Also generate the json file for the entire graph. 
    pipeline_args = {}
    pipeline_args["rank"] = rank
    pipeline_args["node-global-id"] = node_data[: ,0]
    pipeline_args["node-ntype"] = node_data[: ,1]
    pipeline_args["node-ntype-orig-ids"] = node_data[:, 2]
    pipeline_args["node-orig-id"] = node_data[:, 3]
    pipeline_args["node-local-node-type-id"] = node_data[:, 4]
    pipeline_args["node-global-node-id-offset"] = node_offset_globalid
    pipeline_args["node-features"] = node_features

    pipeline_args["edge-src-id"] = edge_data[:,0]
    pipeline_args["edge-dst-id"] = edge_data[:,1]
    pipeline_args["edge-orig-src-id"] = edge_data[:,2]
    pipeline_args["edge-orig-dst-id"] = edge_data[:,3]
    pipeline_args["edge-orig-edge-id"] = edge_data[:,4]
    pipeline_args["edge-etype-ids"] = edge_data[:,5]
    pipeline_args["edge-global-ids"] = edge_data[:,6]
    pipeline_args["edge-global-edge-id-offset"] = edge_offset_globalid
    #pipeline_args["edge-features"] = edge_features

    #call convert_parititio.py for serialization 
    print('Rank: ', rank, 'Starting to serialize necessary files per Metis partitions')
    json_metadata, output_dir, graph_name = gen_dgl_objs(False, pipeline_args, params)

    if (rank == 0): 
        metadata_list = gather_metadata_json(json_metadata, rank, world_size)
        metadata_list[0] = json_metadata
        write_metadata_json(metadata_list, output_dir, graph_name)
    else: 
        gather_metadata_json(json_metadata, rank, world_size)

def init_process(rank, world_size, proc_exec, params, backend="gloo"):
    """
    Init. function which is run by each process in the Gloo ProcessGroup
    """
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    proc_exec(rank, world_size, params)
