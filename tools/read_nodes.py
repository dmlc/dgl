
import sys
import torch
import numpy as np

import dgl

def include_recv_proc_edges( edges, partitions ): 
    rcvIdx = [ partitions[x[1]] for x in edges ]
    edges_aug = np.c_[ edges, rcvIdx ]
    return edges_aug

def include_recv_proc_nodes( node_data, partitions ): 
    gIdx = np.arange(node_data.shape[0])
    node_data_aug = np.c_[ node_data, gIdx ]

    rcvIdx = [ partitions[ x[6]] for x in node_data_aug ]
    node_data_aug = np.c_[ node_data_aug, rcvIdx ]

    return node_data_aug

def read_nodes_file(nodesFile): 
    if nodesFile == "" or nodesFile == None: 
        return None

    # Read the file from here. 
    # Assuming the nodes file is a numpy file 
    # nodes.txt file is of the following format
    # <node_type> <weight1> <weight2> <weight3> <weight4> <orig_type_node_id> <attributes>
    # For the ogb-mag dataset, nodes.txt is of the above format. 
    #

    nodes_data = np.loadtxt( nodesFile, delimiter=' ', dtype='int')
    print(f'Dimensions of the nodes file is: {nodes_data.shape }')
    return nodes_data

def read_edge_file( edge_file ): 
    if edge_file == "" or edge_file == None: 
        return None

    #Read the file from here. 
    #<src_id> <dst_id> <type_edge_id> <edge_type> <attributes>
    # src_id -- global idx for the source node ... line # in the graph_nodes.txt
    # dst_id -- global idx for the destination id node ... line # in the graph_nodes.txt

    edge_data = np.loadtxt( edge_file , delimiter=' ', dtype = 'int' )
    print( f'Dimesions of the edge file: {edge_data.shape}')
    return edge_data


def read_node_features_file (nodes_features_file):
    node_features = dgl.data.utils.load_tensors( nodes_features_file, False)
    return node_features


def read_edge_features_file( edge_features_file ):
    edge_features = dgl.data.utils.load_tensors( edge_features_file, True )
    return edge_features

def read_json(json_file): 
    with open(json_file) as schema: 
        val = json.load(schema)

    return val
