import os
import numpy as np
import constants

def get_dataset(input_dir, graph_name, rank):
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
    dicitonary
        Data read from removed_edges txt file and a dictionary is built with column names as keys
        and columns in the csv file as values. 
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
    for k, v in node_features.items():
        print('[Rank: ', rank, '] node feature name: ', k, ', feature data shape: ', v.size())

    #read (split) xxx_nodes.txt file
    node_file = input_dir+'/'+graph_name+'_nodes'+'{:02d}.txt'.format(rank)
    node_data = np.loadtxt(node_file, delimiter=' ', dtype='int64')
    nodes_datadict = {}
    nodes_datadict[constants.NTYPE_ID] = node_data[:,0]
    nodes_datadict[constants.GLOBAL_TYPE_NID] = node_data[:,5]
    print('[Rank: ', rank, '] Done reading node_data: ', len(nodes_datadict), nodes_datadict[constants.NTYPE_ID].shape)

    #read (split) xxx_edges.txt file
    edge_datadict = {}
    edge_file = input_dir+'/'+graph_name+'_edges'+'{:02d}.txt'.format(rank)
    edge_data = np.loadtxt(edge_file, delimiter=' ', dtype='int64')
    edge_datadict[constants.GLOBAL_SRC_ID] = edge_data[:,0]
    edge_datadict[constants.GLOBAL_DST_ID] = edge_data[:,1]
    edge_datadict[constants.GLOBAL_TYPE_EID] = edge_data[:,2]
    edge_datadict[constants.ETYPE_ID] = edge_data[:,3]
    print('[Rank: ', rank, '] Done reading edge_file: ', len(edge_datadict), edge_datadict[constants.GLOBAL_SRC_ID].shape)

    #read (single) file xxx_removed_edges.txt file
    redge_datadict = {}
    removed_edges_file = input_dir+'/'+graph_name+'_removed_edges'+'{:02d}.txt'.format(rank)
    removed_edges = np.loadtxt(removed_edges_file, delimiter=' ', dtype='int64')
    redge_datadict[constants.GLOBAL_SRC_ID] = removed_edges[:,0]
    redge_datadict[constants.GLOBAL_DST_ID] = removed_edges[:,1]
    redge_datadict[constants.GLOBAL_TYPE_EID] = removed_edges[:,2]
    redge_datadict[constants.ETYPE_ID] = removed_edges[:,3]
    print('[Rank: ', rank, '] Done reading removed_edge_file: ', len(redge_datadict), redge_datadict[constants.GLOBAL_SRC_ID].shape)

    return nodes_datadict, node_features, edge_datadict, redge_datadict 
