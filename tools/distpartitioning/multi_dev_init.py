import os
import sys
import constants
import numpy as np
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl

from timeit import default_timer as timer
from datetime import timedelta
from dataset_utils import get_dataset
from utils import read_partitions_file, read_json, get_node_types, \
                    augment_node_data, augment_edge_data, get_ntypes_map, \
                    write_dgl_objects, write_metadata_json
from gloo_wrapper import alltoall_cpu_object_lst, alltoallv_cpu, \
                    alltoall_cpu, allgather_sizes, gather_metadata_json
from globalids import assign_shuffle_global_nids_nodes, \
                    assign_shuffle_global_nids_edges, get_shuffle_global_nids_edges
from convert_partition import create_dgl_object, create_metadata_json, validateDGLObjects

def exchange_node_data(rank, world_size, node_data):
    """
    Exchange node_data among the processes in the world
    Prepare the list of slices targeting each of the process and
    trigger alltoallv_cpu for the message exchange.

    Parameters:
    -----------
    rank : int
        rank of the current process
    world_size : int
        total no. of participating processes
    node_data : dictionary
        nodes data dictionary with keys as column names and values as
        columns from the nodes csv file
    """
    input_list = []
    send_sizes = []
    recv_sizes = []
    start = timer()
    for i in np.arange(world_size):
        send_idx = (node_data[constants.OWNER_PROCESS] == i)
        idx = send_idx.reshape(node_data[constants.GLOBAL_NID].shape[0])
        filt_data = np.column_stack((node_data[constants.NTYPE_ID][idx == 1], \
                                node_data[constants.GLOBAL_TYPE_NID][idx == 1], \
                                node_data[constants.GLOBAL_NID][idx == 1]))
        if(filt_data.shape[0] <= 0): 
            input_list.append(torch.empty((0,), dtype=torch.int64))
            send_sizes.append(torch.empty((0,), dtype=torch.int64))
        else:
            input_list.append(torch.from_numpy(filt_data))
            send_sizes.append(torch.tensor(filt_data.shape, dtype=torch.int64))
        recv_sizes.append(torch.zeros((2,), dtype=torch.int64))
    end = timer()
    print('[Rank: ', rank, '] Preparing node_data to send out: ', timedelta(seconds=end - start))

    #exchange sizes first followed by data. 
    dist.barrier()
    start = timer()
    alltoall_cpu(rank, world_size, recv_sizes, send_sizes)

    output_list = []
    for s in recv_sizes: 
        output_list.append(torch.zeros(s.tolist(), dtype=torch.int64))
    
    dist.barrier()
    alltoallv_cpu(rank, world_size, output_list, input_list)
    end = timer()
    print('[Rank: ', rank, '] Time to exchange node data : ', timedelta(seconds=end - start))

    #stitch together the received data to form a consolidated data-structure
    rcvd_node_data = torch.cat(output_list).numpy()
    print('[Rank: ', rank, '] Received node data shape ', rcvd_node_data.shape)

    #Replace the node_data values with the received node data and the OWNER_PROCESS key-value
    #pair is removed after the data communication
    node_data[constants.NTYPE_ID] = rcvd_node_data[:,0]
    node_data[constants.GLOBAL_TYPE_NID] = rcvd_node_data[:,1]
    node_data[constants.GLOBAL_NID] = rcvd_node_data[:,2]
    node_data.pop(constants.OWNER_PROCESS)

def exchange_edge_data(rank, world_size, edge_data):
    """
    Exchange edge_data among processes in the world.
    Prepare list of sliced data targeting each process and trigger
    alltoallv_cpu to trigger messaging api

    Parameters:
    -----------
    rank : int
        rank of the process
    world_size : int
        total no. of processes
    edge_data : dictionary
        edge information, as a dicitonary which stores column names as keys and values
        as column data. This information is read from the edges.txt file.
    """
    input_list = []
    send_sizes = []
    recv_sizes = []
    start = timer()
    for i in np.arange(world_size):
        send_idx = (edge_data[constants.OWNER_PROCESS] == i)
        send_idx = send_idx.reshape(edge_data[constants.GLOBAL_SRC_ID].shape[0])
        filt_data = np.column_stack((edge_data[constants.GLOBAL_SRC_ID][send_idx == 1], \
                                    edge_data[constants.GLOBAL_DST_ID][send_idx == 1], \
                                    edge_data[constants.GLOBAL_TYPE_EID][send_idx == 1], \
                                    edge_data[constants.ETYPE_ID][send_idx == 1], \
                                    edge_data[constants.GLOBAL_EID][send_idx == 1]))
        if(filt_data.shape[0] <= 0):
            input_list.append(torch.empty((0,), dtype=torch.int64))
            send_sizes.append(torch.empty((0,), dtype=torch.int64))
        else:
            input_list.append(torch.from_numpy(filt_data))
            send_sizes.append(torch.tensor(filt_data.shape, dtype=torch.int64))
        recv_sizes.append(torch.zeros((2,), dtype=torch.int64))
    end = timer()
    
    dist.barrier ()
    start = timer()
    alltoall_cpu(rank, world_size, recv_sizes, send_sizes)
    output_list = []
    for s in recv_sizes: 
        output_list.append(torch.zeros(s.tolist(), dtype=torch.int64))

    dist.barrier ()
    alltoallv_cpu(rank, world_size, output_list, input_list)
    end = timer()
    print('[Rank: ', rank, '] Time to send/rcv edge data: ', timedelta(seconds=end-start))

    #Replace the values of the edge_data, with the received data from all the other processes.
    rcvd_edge_data = torch.cat(output_list).numpy()
    edge_data[constants.GLOBAL_SRC_ID] = rcvd_edge_data[:,0]
    edge_data[constants.GLOBAL_DST_ID] = rcvd_edge_data[:,1]
    edge_data[constants.GLOBAL_TYPE_EID] = rcvd_edge_data[:,2]
    edge_data[constants.ETYPE_ID] = rcvd_edge_data[:,3]
    edge_data[constants.GLOBAL_EID] = rcvd_edge_data[:,4]
    edge_data.pop(constants.OWNER_PROCESS)


def exchange_node_features(rank, world_size, node_data, node_features, ntypes_map, \
        ntypes_nid_map, ntype_id_count, node_part_ids):
    """
    This function is used to shuffle node features so that each process will receive
    all the node features whose corresponding nodes are owned by the same process. 
    The mapping procedure to identify the owner process is not straight forward. The
    following steps are used to identify the owner processes for the locally read node-
    features. 
    a. Compute the global_nids for the locally read node features. Here metadata json file
        is used to identify the corresponding global_nids. Please note that initial graph input
        nodes.txt files are sorted based on node_types. 
    b. Using global_nids and metis partitions owner processes can be easily identified. 
    c. Now each process sends the global_nids for which shuffle_global_nids are needed to be 
        retrieved. 
    d. After receiving the corresponding shuffle_global_nids these ids are added to the 
        node_data and edge_data dictionaries

    Parameters: 
    -----------
    rank : int
        rank of the current process
    world_size : int
        total no. of participating processes. 
    node_data : dictionary
        dictionary where node data is stored, which is initially read from the nodes txt file mapped
        to the current process
    node_feautres: dicitonary
        dictionry where node_features are stored and this information is read from the appropriate
        node features file which belongs to the current process
    ntypes_map : dictionary
        mappings between node type names and node type ids
    ntypes_nid_map : dictionary
        mapping between node type names and global_nids which belong to the keys in this dictionary
    ntype_id_count : dictionary
        mapping between node type id and no of nodes which belong to each node_type_id
    node_part_ids : numpy array
        numpy array which store the partition-ids and indexed by global_nids
    """

    #determine Global_type_nid for the residing features 
    start = timer()
    node_features_rank_lst = []
    global_nid_rank_lst = []
    for part_id in np.arange(world_size):

        #form outgoing features to each process
        send_node_features = {}
        send_global_nids = {}
        for ntype_name, ntype_id in ntypes_map.items(): 

            #check if features exist for this node_type
            if (ntype_name+'/feat' in node_features) and (node_features[ntype_name+'/feat'].shape[0] > 0):
                feature_count = node_features[ntype_name+'/feat'].shape[0]
                global_feature_count = ntype_id_count[str(ntype_id)]

                #determine the starting global_nid for this node_type_id
                feat_per_proc = math.ceil(global_feature_count / world_size)
                global_type_nid_start = feat_per_proc * rank
                global_type_nid_end = global_type_nid_start
                if((global_type_nid_start + feat_per_proc) > global_feature_count):
                    global_type_nid_end += (ntype_id_count[str(ntype_id)] - global_type_nid_start)
                    type_nid = np.arange(0, (ntype_id_count[str(ntype_id)] - global_type_nid_start))
                else: 
                    global_type_nid_end += feat_per_proc 
                    type_nid = np.arange(0, feat_per_proc)

                #now map the global_ntype_id to global_nid 
                global_nid_offset = ntypes_nid_map[ntype_name][0]
                global_nid_start = global_type_nid_start + global_nid_offset
                global_nid_end = global_type_nid_end + global_nid_offset

                #assert (global_nid_end - global_nid_start) == feature_count
                global_nids = np.arange(global_nid_start, global_nid_end, dtype=np.int64)

                #determine node feature ownership 
                #TODO: a Bug here. 
                '''
                part_ids = node_part_ids[global_nids] 
                idx = (part_ids == part_id)
                out_global_nid = global_nids[idx == 1]
                out_type_nid = type_nid[idx == 1]
                out_features = node_features[ntype_name+'/feat'][out_type_nid]
                send_node_features[ntype_name+'/feat'] = out_features
                send_global_nids[ntype_name+'/feat'] = out_global_nid
                '''
                part_ids_slice = node_part_ids[global_nid_start:global_nid_end]
                idx = (part_ids_slice == part_id)
                out_global_nid = global_nids[idx == 1]
                out_type_nid = type_nid[idx == 1]
                out_features = node_features[ntype_name+'/feat'][out_type_nid]
                send_node_features[ntype_name+'/feat'] = out_features
                send_global_nids[ntype_name+'/feat'] = out_global_nid

        node_features_rank_lst.append(send_node_features)
        global_nid_rank_lst.append(send_global_nids)

    dist.barrier ()
    output_list = alltoall_cpu_object_lst(rank, world_size, node_features_rank_lst)
    output_list[rank] = node_features_rank_lst[rank]

    output_nid_list = alltoall_cpu_object_lst(rank, world_size, global_nid_rank_lst)
    output_nid_list[rank] = global_nid_rank_lst[rank]
            
    #stitch node_features together to form one large feature tensor
    rcvd_node_features = {}
    rcvd_global_nids = {}
    for idx in range(world_size):
        for ntype_name, ntype_id in ntypes_map.items():
            if ((output_list[idx] is not None) and (ntype_name+'/feat' in output_list[idx])):
                if (ntype_name+'/feat' not in rcvd_node_features):
                    rcvd_node_features[ntype_name+'/feat'] = torch.empty((0,), dtype=torch.float)
                    rcvd_global_nids[ntype_name+'/feat'] = torch.empty((0,), dtype=torch.int64)
                rcvd_node_features[ntype_name+'/feat'] = \
                    torch.cat((rcvd_node_features[ntype_name+'/feat'], output_list[idx][ntype_name+'/feat']))
                rcvd_global_nids[ntype_name+'/feat'] = \
                    np.concatenate((rcvd_global_nids[ntype_name+'/feat'], output_nid_list[idx][ntype_name+'/feat']))
    end = timer()
    print('[Rank: ', rank, '] Total time for node feature exchange: ', timedelta(seconds = end - start))

    return rcvd_node_features, rcvd_global_nids

def exchange_graph_data(rank, world_size, node_data, node_features, edge_data,
        node_part_ids, ntypes_map, ntypes_nid_map, ntype_id_count):
    """
    Wrapper function which is used to shuffle graph data on all the processes. 

    Parameters: 
    -----------
    rank : int
        rank of the current process
    world_size : int
        total no. of participating processes. 
    node_data : dictionary
        dictionary where node data is stored, which is initially read from the nodes txt file mapped
        to the current process
    node_feautres: dicitonary
        dictionry where node_features are stored and this information is read from the appropriate
        node features file which belongs to the current process
    edge_data : dictionary
        dictionary which is used to store edge information as read from the edges.txt file assigned
        to each process.
    node_part_ids : numpy array
        numpy array which store the partition-ids and indexed by global_nids
    ntypes_map : dictionary
        mappings between node type names and node type ids
    ntypes_nid_map : dictionary
        mapping between node type names and global_nids which belong to the keys in this dictionary
    ntype_id_count : dictionary
        mapping between node type id and no of nodes which belong to each node_type_id
    """
    rcvd_node_features, rcvd_global_nids = exchange_node_features(rank, world_size, node_data, \
            node_features, ntypes_map, ntypes_nid_map, ntype_id_count, node_part_ids)
    print( 'Rank: ', rank, ' Done with node features exchange.')

    exchange_node_data(rank, world_size, node_data)
    exchange_edge_data(rank, world_size, edge_data)
    return rcvd_node_features, rcvd_global_nids

def read_dataset(rank, world_size, node_part_ids, params):
    """
    After reading input graph files, add additional information(columns) are added
    to these data structures, as discussed in detail in the initialize.py file for the
    case of single-file-format dataset use-case. 

    Parameters:
    -----------
    rank : int
        rank of the current process
    worls_size : int
        total no. of processes instantiated
    node_part_ids : numpy array
        metis partitions which are the output of partitioning algorithm
    params : argparser object 
        argument parser object to access command line arguments

    Returns : 
    ---------
    dictionary
        node data information is read from nodes.txt and additionnal columns are added such as 
        owner process for each node.
    dictionary
        node features which is a dictionary where keys are feature names and values are feature
        data as multi-dimensional tensors 
    dictionary
        edge data information is read from edges.txt and additional columns are added such as 
        owner process for each edge. 
    dictionary
        edge features which is also a dictionary, similar to node features dictionary
    """
    edge_features = {}
    node_data, node_features, edge_data = \
        get_dataset(params.input_dir, params.graph_name, rank, params.num_node_weights)

    prefix_sum_nodes = allgather_sizes([node_data[constants.NTYPE_ID].shape[0]], world_size)
    augment_node_data(node_data, node_part_ids, prefix_sum_nodes[rank])
    print('[Rank: ', rank, '] Done augmenting node_data: ', len(node_data), node_data[constants.GLOBAL_TYPE_NID].shape)
    print('[Rank: ', rank, '] Done assigning Global_NIDS: ', prefix_sum_nodes[rank], prefix_sum_nodes[rank+1], prefix_sum_nodes[rank]+node_data[constants.GLOBAL_TYPE_NID].shape[0])

    prefix_sum_edges = allgather_sizes([edge_data[constants.ETYPE_ID].shape[0]], world_size)
    augment_edge_data(edge_data, node_part_ids, prefix_sum_edges[rank])
    print('[Rank: ', rank, '] Done augmenting edge_data: ', len(edge_data), edge_data[constants.GLOBAL_SRC_ID].shape)

    return node_data, node_features, edge_data, edge_features

def gen_dist_partitions(rank, world_size, params):
    """
    Function which will be executed by all Gloo processes to
    begin execution of the pipeline. This function expects the input dataset is split
    across multiple file format. Directory structure is described below in detail:
    input_dir/
        <graph-name>_nodes00.txt
        ....
        <graph-name>_nodes<world_size-1>.txt
        <graph-name>_edges00.txt
        ....
        <graph-name>_edges<world_size-1>.txt
        <graph-name>_metadata.json
        nodes-ntype0-XY/ #XY = no. of features to read for this ntype
            node-feat-0/
                0.npy
                1.npy
                ....
                <world_size-1>.npy
            ....
            node-feat-<XY-1>/
                0.npy
                1.npy
                ....
                <world_size-1>.npy
        nodes-ntype1-XY/ #XY = no. of features to read for this ntype
            node-feat-0/
                0.npy
                1.npy
                ....
                <world_size-1>.npy
            ....
            node-feat-<XY-1>/
                0.npy
                1.npy
                ....
                <world_size-1>.npy

    Basically, each individual file is split into "p" files, where "p" is the no. of processes in the
    world. Directory names are encoded strings which consist of prefix and suffix strings. Suffix strings
    indicate the no. of items present inside that directory. For instance, "nodes-ntype0-2" directory has 
    "2" node type sub-directories within it. And each feature file, whether it is node features file or edge
    feature file, is split into "p" numpy files named as 0.npy, 1.npy, ..., <p-1>.npy. 

    The function performs the following steps: 
    1. Reads the metis partitions to identify the owner process of all the nodes in the entire graph.
    2. Reads the input data set, each partitipating process will map to a single file for the nodes, edges, 
        node-features and edge-features for each node-type and edge-types respectively.
    3. Now each process shuffles the data by identifying the respective owner processes using metis
        partitions. 
        a. To identify owner processes for nodes, metis partitions will be used. 
        b. For edges, the owner process of the destination node will be the owner of the edge as well. 
        c. For node and edge features, identifying the owner process is a little bit involved. 
            For this purpose, graph metadata json file is used to first map the locally read node features
            to their global_nids. Now owner process is identified using metis partitions for these global_nids
            to retrieve shuffle_global_nids. A similar process is used for edge_features as well. 
        d. After all the data shuffling is done, the order of node-features may be different when compared to 
            their global_type_nids. Node- and edge-data are ordered by node-type and edge-type respectively. 
            And now node features and edge features are re-ordered to match the order of their node- and edge-types. 
    4. Last step is to create the DGL objects with the data present on each of the processes. 
        a. DGL objects for nodes, edges, node- and edge- features. 
        b. Metadata is gathered from each process to create the global metadata json file, by process rank = 0. 

    Parameters:
    ----------
    rank : int
        integer representing the rank of the current process in a typical distributed implementation
    world_size : int
        integer representing the total no. of participating processes in a typical distributed implementation
    params : argparser object
        this object, key value pairs, provides access to the command line arguments from the runtime environment
    """
    global_start = timer()
    print('[Rank: ', rank, '] Starting distributed data processing pipeline...')

    #init processing
    node_part_ids = read_partitions_file(params.input_dir+'/'+params.partitions_file)
    schema_map = read_json(params.input_dir+'/'+params.schema)
    ntypes_map, ntypes = get_node_types(schema_map)
    print('[Rank: ', rank, '] Initialized metis partitions and node_types map...')

    #read input graph files and augment these datastructures with
    #appropriate information (global_nid and owner process) for node and edge data
    node_data, node_features, edge_data, edge_features = read_dataset(rank, world_size, node_part_ids, params)
    print('[Rank: ', rank, '] Done augmenting file input data with auxilary columns')

    #send out node and edge data --- and appropriate features. 
    #this function will also stitch the data recvd from other processes
    #and return the aggregated data
    ntypes_nid_map, ntype_id_count = get_ntypes_map(schema_map)
    rcvd_node_features, rcvd_global_nids  = exchange_graph_data(rank, world_size, node_data, \
            node_features, edge_data, node_part_ids, ntypes_map, ntypes_nid_map, ntype_id_count)
    print('[Rank: ', rank, '] Done with data shuffling...')

    #sort node_data by ntype
    idx = node_data[constants.NTYPE_ID].argsort()
    for k, v in node_data.items():
        node_data[k] = v[idx]
    print('[Rank: ', rank, '] Sorted node_data by node_type')

    #resolve global_ids for nodes
    assign_shuffle_global_nids_nodes(rank, world_size, node_data)
    print('[Rank: ', rank, '] Done assigning global-ids to nodes...')

    #shuffle node feature according to the node order on each rank. 
    for ntype_name in ntypes: 
        if (ntype_name+'/feat' in rcvd_global_nids):
            global_nids = rcvd_global_nids[ntype_name+'/feat']

            common, idx1, idx2 = np.intersect1d(node_data[constants.GLOBAL_NID], global_nids, return_indices=True)
            shuffle_global_ids = node_data[constants.SHUFFLE_GLOBAL_NID][idx1]
            feature_idx = shuffle_global_ids.argsort()
            rcvd_node_features[ntype_name+'/feat'] = rcvd_node_features[ntype_name+'/feat'][feature_idx]

    #sort edge_data by etype
    sorted_idx = edge_data[constants.ETYPE_ID].argsort()
    for k, v in edge_data.items():
        edge_data[k] = v[sorted_idx]

    shuffle_global_eid_start = assign_shuffle_global_nids_edges(rank, world_size, edge_data)
    print('[Rank: ', rank, '] Done assigning global_ids to edges ...')

    #determine global-ids for edge end-points
    get_shuffle_global_nids_edges(rank, world_size, edge_data, node_part_ids, node_data)
    print('[Rank: ', rank, '] Done resolving orig_node_id for local node_ids...')

    #create dgl objects here
    start = timer()
    num_nodes = 0
    num_edges = shuffle_global_eid_start
    graph_obj, ntypes_map_val, etypes_map_val, ntypes_map, etypes_map = create_dgl_object(\
            params.graph_name, params.num_parts, \
            schema_map, rank, node_data, edge_data, num_nodes, num_edges)
    write_dgl_objects(graph_obj, rcvd_node_features, edge_features, params.output, rank)

    #get the meta-data 
    json_metadata = create_metadata_json(params.graph_name, len(node_data[constants.NTYPE_ID]), len(edge_data[constants.ETYPE_ID]), \
                            rank, world_size, ntypes_map_val, \
                            etypes_map_val, ntypes_map, etypes_map, params.output)

    if (rank == 0):
        #get meta-data from all partitions and merge them on rank-0
        metadata_list = gather_metadata_json(json_metadata, rank, world_size)
        metadata_list[0] = json_metadata
        write_metadata_json(metadata_list, params.output, params.graph_name)
    else:
        #send meta-data to Rank-0 process
        gather_metadata_json(json_metadata, rank, world_size)
    end = timer()
    print('[Rank: ', rank, '] Time to create dgl objects: ', timedelta(seconds = end - start))

    global_end = timer()
    print('[Rank: ', rank, '] Total execution time of the program: ', timedelta(seconds = global_end - global_start))
