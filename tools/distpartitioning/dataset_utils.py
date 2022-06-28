import os
import numpy as np
import constants
import torch

def get_dataset(input_dir, graph_name, rank, num_node_weights):
    """
    Function to read the multiple file formatted dataset. 

    Parameters:
    -----------
    input_dir : string
        root directory where dataset is located.
    graph_name : string
        graph name string
    rank : int
        rank of the current process
    num_node_weights : int
        integer indicating the no. of weights each node is attributed with

    Return:
    -------
    dictionary
        Data read from nodes.txt file and used to build a dictionary with keys as column names
        and values as columns in the csv file.
    dictionary
        Data read from numpy files for all the node features in this dataset. Dictionary built 
        using this data has keys as node feature names and values as tensor data representing 
        node features
    dictionary
        Data read from edges.txt file and used to build a dictionary with keys as column names 
        and values as columns in the csv file. 
    """
    #node features dictionary
    node_features = {}

    #iterate over the sub-dirs and extract the nodetypes
    #in each nodetype folder read all the features assigned to 
    #current rank
    siblings = os.listdir(input_dir)
    for s in siblings:
        if s.startswith("nodes-"):
            tokens = s.split("-")
            ntype = tokens[1]
            num_feats = tokens[2]
            for idx in range(int(num_feats)):
                feat_file = s +'/node-feat-'+'{:02d}'.format(idx) +'/'+ str(rank)+'.npy'
                if (os.path.exists(input_dir+'/'+feat_file)):
                    features = np.load(input_dir+'/'+feat_file)
                    node_features[ntype+'/feat'] = torch.tensor(features)

    #done build node_features locally. 
    if len(node_features) <= 0: 
        print('[Rank: ', rank, '] This dataset does not have any node features')
    else: 
        for k, v in node_features.items():
            print('[Rank: ', rank, '] node feature name: ', k, ', feature data shape: ', v.size())

    #read (split) xxx_nodes.txt file
    node_file = input_dir+'/'+graph_name+'_nodes'+'_{:02d}.txt'.format(rank)
    node_data = np.loadtxt(node_file, delimiter=' ', dtype='int64')
    nodes_datadict = {}
    nodes_datadict[constants.NTYPE_ID] = node_data[:,0]
    type_idx = 0 + num_node_weights + 1
    nodes_datadict[constants.GLOBAL_TYPE_NID] = node_data[:,type_idx]
    print('[Rank: ', rank, '] Done reading node_data: ', len(nodes_datadict), nodes_datadict[constants.NTYPE_ID].shape)

    #read (split) xxx_edges.txt file
    edge_datadict = {}
    edge_file = input_dir+'/'+graph_name+'_edges'+'_{:02d}.txt'.format(rank)
    edge_data = np.loadtxt(edge_file, delimiter=' ', dtype='int64')
    edge_datadict[constants.GLOBAL_SRC_ID] = edge_data[:,0]
    edge_datadict[constants.GLOBAL_DST_ID] = edge_data[:,1]
    edge_datadict[constants.GLOBAL_TYPE_EID] = edge_data[:,2]
    edge_datadict[constants.ETYPE_ID] = edge_data[:,3]
    print('[Rank: ', rank, '] Done reading edge_file: ', len(edge_datadict), edge_datadict[constants.GLOBAL_SRC_ID].shape)

    return nodes_datadict, node_features, edge_datadict
