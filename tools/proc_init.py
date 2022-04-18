import os
import sys
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from timeit import default_timer as timer
from datetime import timedelta

from read_nodes import include_recv_proc_nodes, include_recv_proc_edges, \
                        read_nodes_file, read_edge_file, \
                        read_node_features_file, read_edge_features_file
from read_metis_partitions import read_metis_partitions
from proc_nodecount import assignGlobalNodeIds, assignGlobalEdgeIds
from proc_globalids import getGlobalIdsForEdges
from read_schema import read_json, get_node_types, \
                        write_metadata_json, get_metadata_json, send_metadata_json

from convert_partition import processMetisPartitions

'''
This function is run by rank-0 process. 
Rank-0 process reads input files and sends out data to other processes. 
It sends nodes, edges, node_features to respective Gloo processes as per metis partitions
'''
def sendOutData ( rank, node_data, node_features, edge_data, metis_partitions, ntypes_map ): 
    '''
    sending nodes data to other processes. 
    <node_type> <weight1> <weight2> <weight3> <weight4> <orig_type_node_id> <line(global)_id> <recv_proc_id>
    '''
    recvProcs = np.unique( list( metis_partitions.values () ) )
    recvProcs.sort ()
    print( 'Rank: ', rank, ', Unique partitions: ', recvProcs )

    for rProc in recvProcs: 
        if rProc == rank: 
            continue
        
        sendData = (node_data[:, 7] == rProc) 
        idx = sendData.reshape( node_data.shape[0] )
        sendDataFiltered = node_data[:,[0,5,6]][idx == 1] # send node_type, orig_type_id, line_id (global_id)

        sendDataSize = sendDataFiltered.shape
        sendTensor = torch.zeros(len(sendDataSize), dtype=torch.int)
        for idx in range(len(sendDataSize)): 
            sendTensor[idx] = sendDataSize[idx]
        # Send size first, so that the rProc can create appropriately sized tensor
        dist.send( sendTensor, dst=rProc )

        start = timer ()
        sendTensor = torch.from_numpy( sendDataFiltered.astype(np.int32) )
        dist.send( sendTensor, dst=rProc )
        end = timer ()
        print( 'Rank: ', rank, ' Sent data size: ', sendDataFiltered.shape, ', to Process: ', rProc, 'in: ', timedelta(seconds = end - start) )

    print( 'Rank: ', rank, ' Done sending Node information to non-rank-0 processes' )

    node_features_rank_lst = []
    for rProc in recvProcs: 
        if rProc == rank: 
            node_features_rank_lst.append( None )
            continue

        send_node_features = {}
        for x in ntypes_map.items(): 
            node_type_name = x[0]
            node_type = x[1]
            
            sendData = (node_data[:, 7] == rProc) & (node_data[ :, 0] == node_type) 
            origTypeNodeIdFiltered = node_data[:,[5]][sendData] # extract orig_type_node_id here
            origTypeNodeIdFiltered = np.concatenate( origTypeNodeIdFiltered ) 

            if (node_type_name +'/feat' in node_features) and (node_features[node_type_name+'/feat'].shape[0] > 0): 
                send_node_features[node_type_name+'/feat'] = node_features[node_type_name+'/feat'][origTypeNodeIdFiltered]
            else: 
                send_node_features[node_type_name+'/feat'] = None

        node_features_rank_lst.append( send_node_features )

    output_list = [None]
    start = timer ()
    dist.scatter_object_list(output_list, node_features_rank_lst, src=0)
    end = timer ()
    print( 'Rank: ', rank, ', Done sending Node Features in: ', timedelta(seconds = end - start))

    for rProc in recvProcs: 
        if rProc == rank: 
            continue

        sendData = (edge_data[:, 4] == rProc) 
        idx = sendData.reshape( edge_data.shape[0] )
        sendDataFiltered = edge_data[:,[0,1,2,3]][idx == 1]

        sendDataSize = sendDataFiltered.shape
        sendTensor = torch.zeros(len(sendDataSize), dtype=torch.int32)
        for idx in range(len(sendDataSize)): 
            sendTensor[idx] = sendDataSize[idx]
        # Send size first, so that the rProc can create appropriately sized tensor
        dist.send( sendTensor, dst=rProc )

        start = timer ()
        sendTensor = torch.from_numpy( sendDataFiltered.astype( np.int32 ))
        dist.send( sendTensor, dst=rProc )
        end = timer ()
        print( 'Rank: ', rank, ' Time to send Edges to proc: ', rProc, ' is : ', timedelta(seconds = end - start) )
    print( 'Rank: ', rank, ' Done sending Edge information to non-rank-0 processes' )
        

'''
This function is run by non-rank-0 processes. 
This function receives data from rank-0 processes and stores them locally in 
local data structures. 
'''
def recvData(rank, dimensions, dtype): 

    '''
    First receive the size of the data to be received from rank-0 process
    '''
    recv_tensor_shape = torch.zeros(dimensions, dtype = torch.int32)
    dist.recv(recv_tensor_shape, src=0)
    recv_shape = list( map( lambda x: int(x), recv_tensor_shape) )

    '''
    Receive the data message here for nodes here. 
    '''
    recv_tensor_data = torch.zeros( recv_shape, dtype=dtype)
    dist.recv( recv_tensor_data, src=0 )
    return recv_tensor_data.numpy ()

'''
Function to receive node data
'''
def recvNodeData( rank, dimensions, dtype ): 
    return recvData( rank, dimensions, dtype )

'''
Invoke the Node Data here as well. it will serve the purpose. 
'''
def recvEdgeData (rank, dimensions, dtype): 
    return recvNodeData (dimensions, dtype)

'''
This function is for receiving node features. 
Note that the node features are indexed by the orig_node_type_id. 
for the mag-dataset, only node_type = 3 has node features and others node types does not
have any node features. 
'''
def recvNodeFeatures (rank, dtype): 
    globalIds = recvNodeData( rank, 1, torch.int32 ) ## to receive the origTypeNodeIDs
    node_features = recvNodeData( rank, 2, torch.float32 ) ## to receive actual node feature data
    print( 'Rank: ', rank, ', Done receiving node feature data... ' )
    return globalIds, node_features

def recv_node_features_obj(rank, size): 
    send_objs = [None for _ in range(size)]
    recv_obj = [None]
    dist.scatter_object_list(recv_obj, send_objs, src=0)

    node_features = recv_obj[0]
    return node_features


def readInputFiles( rank, params, metis_partitions ): 

    '''
    Node data is structured as follows: 
    <node_type> <weight1> <weight2> <weight3> <weight4> <orig_type_node_id> <attributes>
    is converted to 
    <node_type> <weight1> <weight2> <weight3> <weight4> <orig_type_node_id> <nid> <recv_proc>
    '''
    rcvProc_nodes_data = []
    nodes_data = read_nodes_file( params.input_dir+'/'+params.nodes_file )
    rcvProc_nodes_data = include_recv_proc_nodes( nodes_data, metis_partitions )
    print( 'Rank: ', rank, ', Completed loading nodes data: ', rcvProc_nodes_data.shape )

    rcvProc_edge_data = []
    edges_data = read_edge_file( params.input_dir+'/'+params.edges_file )
    rcvProc_edge_data = include_recv_proc_edges( edges_data, metis_partitions )
    print( 'Rank: ', rank, ', Completed loading edges data: ', rcvProc_edge_data.shape )

    node_features = []
    node_features = read_node_features_file( params.input_dir+'/'+params.node_feats_file )
    print( 'Rank: ', rank, ', Completed loading node features reading from file ', len(node_features))

    edge_features = []
    #edge_features = read_edge_features_file( params.input_dir+'/'+params.edge_feats_file )
    #print( 'Rank: ', rank, ', Completed edge features reading from file ', len(edge_features) )

    return rcvProc_nodes_data, node_features, rcvProc_edge_data, edge_features

def run(rank, size, params):

    metis_partitions = read_metis_partitions(params.input_dir+'/'+params.metis_partitions)
    print( 'Rank: ', rank, ', Completed loading metis partitions: ', len(metis_partitions))

    schema_map = read_json(params.input_dir+'/'+params.schema)
    ntypes_map, ntypes = get_node_types(schema_map)

    if rank == 0: 
        node_data, node_features, edge_data, edge_features = readInputFiles( rank, params, metis_partitions )

        # order node_data by node_type before extracting node features. 
        # once this is ordered, node_features are automatically ordered and 
        # can be assigned contiguous ids starting from 0 for each type. 
        node_data = node_data[ node_data[:, 0].argsort() ]

        print( 'Rank: ', rank, ', node_data: ', node_data.shape )
        print( 'Rank: ', rank, ', node_features: ', len(node_features))
        print( 'Rank: ', rank, ', edge_data: ', edge_data.shape )
        #print( 'Rank: ', rank, ', edge_features : ',len( edge_features) )
        print( 'Rank: ', rank, ', partitions : ', len(metis_partitions ))

        # shuffle data
        sendOutData( rank, node_data, node_features, edge_data, metis_partitions, ntypes_map)

        # Filter data owned by rank-0
        node_data = node_data[ :, [0,5,6] ][ node_data[ :, 7 ] == 0 ] #extract only ntype, orig_type_nid, line_id
        edge_data = edge_data[ :, [0,1,2,3]][edge_data[:,4] == 0 ] #extract only orig-src-id, orig-dst-id, orig-type-id orig-type
    else: 
        node_data = recvNodeData (rank, 2, torch.int32)
        node_features = recv_node_features_obj(rank, size)
        edge_data = recvData( rank, 2, torch.int32 )

    '''
    sort data by type-id, and follow through with assigning globalIds
    '''
    type_ids = np.unique( node_data[ :, 0 ] )
    type_ids.sort ()

    lstNodeTypeCounts = []
    typeCounts = np.bincount( node_data[ :, 0 ] )
    for typeId in type_ids: 
        lstNodeTypeCounts.append( (typeId, typeCounts[ typeId ]) )
    print( 'Rank: ', rank, ', LstNodeTypeCounts: ', lstNodeTypeCounts )
        
    # after this call node_data = [ globalId, node_type, orig_node_type_id, line_id, local_type_id ]
    node_data, node_offset_globalid = assignGlobalNodeIds ( rank, size, lstNodeTypeCounts, node_data )
    print( 'Rank: ', rank, ' Done assign Global ids to nodes...')

    #Work on the edge and assign GlobalIds
    etype_ids = np.unique( edge_data[:, 3] )
    etype_ids.sort ()
    print('Rank: ', rank, ' etype-ids: ', etype_ids )
    
    edge_data = edge_data[ edge_data[ :, 3 ].argsort() ]
    lstEdgeTypeCounts = []
    edgeCounts = np.bincount( edge_data[ :,3 ] )

    for etype in etype_ids: 
        lstEdgeTypeCounts.append( (etype, edgeCounts[etype]) )
    print('Rank: ', rank, ', LstEdgeTypeCounts: ', lstEdgeTypeCounts )

    edge_data, edge_offset_globalid = assignGlobalEdgeIds ( rank, size, lstEdgeTypeCounts, edge_data )
    print( 'Rank: ', rank, ' Done assign Global ids to edges...')

    edge_data = getGlobalIdsForEdges( rank, size, edge_data, metis_partitions, node_data)
    print( 'Rank: ', rank, ' Done retrieving Global Node Ids for non-local nodes... ' )
    print( 'Rank: ', rank, ' Done with distributed processing... Starting to serialize necessary files per Metis partitions' )

    '''
    Here use the functionality of the convert partition.py to store the dgl objects
    for the node_features, edge_features and graph itself. 
    Also generate the json file for the entire graph. 
    '''
    pipelineArgs = {}
    pipelineArgs[ "rank" ] = rank
    pipelineArgs["node-global-id" ] = node_data[: ,0]
    pipelineArgs["node-ntype"] = node_data[: ,1]
    pipelineArgs["node-ntype-orig-ids" ] = node_data[:, 2]
    pipelineArgs["node-orig-id" ] = node_data[:, 3]
    pipelineArgs["node-local-node-type-id" ] = node_data[:, 4]
    pipelineArgs["node-global-node-id-offset"] = node_offset_globalid
    pipelineArgs["node-features"] = node_features

    #pipelineArgs[ "edge_data" ] = edge_data
    pipelineArgs[ "edge-src-id" ] = edge_data[:,0]
    pipelineArgs[ "edge-dst-id" ] = edge_data[:,1]
    pipelineArgs[ "edge-orig-src-id" ] = edge_data[:,2]
    pipelineArgs[ "edge-orig-dst-id" ] = edge_data[:,3]
    pipelineArgs[ "edge-orig-edge-id" ] = edge_data[:,4]
    pipelineArgs[ "edge-etype-ids" ] = edge_data[:,5]
    pipelineArgs[ "edge-global-ids" ] = edge_data[:,6]
    pipelineArgs[ "edge-global-edge-id-offset" ] = edge_offset_globalid

    #call convert_parititio.py for serialization 
    json_metadata, output_dir, graph_name = processMetisPartitions (False, pipelineArgs, params)

    if (rank == 0): 
        metadata_list = get_metadata_json(size)
        metadata_list[0] = json_metadata
        write_metadata_json( metadata_list, output_dir, graph_name)
    else: 
        send_metadata_json(json_metadata, size)

'''
Initialization func. to create process group
'''
def init_process(rank, size, fn, params, backend="gloo"):
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, params)
