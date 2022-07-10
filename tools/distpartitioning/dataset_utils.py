import os
import numpy as np
import constants
import torch

def get_dataset(input_dir, graph_name, rank, world_size, schema_map):
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
    node_feature_tids = {}

    #iterate over the "node_data" dictionary in the schema_map
    #read the node features if exists
    #also keep track of the type_nids for which the node_features are read.
    dataset_features = schema_map["node_data"]
    for ntype_name, ntype_feature_data in dataset_features.items():
        #ntype_feature_data is a dictionary
        #where key: feature_name, value: list of lists
        node_feature_tids[ntype_name] = []
        for feat_name, feat_data in ntype_feature_data.items():
            assert len(feat_data) == world_size
            my_feat_data = feat_data[rank]
            if (os.path.isabs(my_feat_data[0])):
                node_features[ntype_name+'/'+feat_name] = torch.from_numpy(np.load(my_feat_data[0]))
            else:
                node_features[ntype_name+'/'+feat_name] = torch.from_numpy(np.load(input_dir+my_feat_data[0]))

            v = [feat_name, my_feat_data[1], my_feat_data[2]]
            l = node_feature_tids[ntype_name]
            l.append(v)
            node_feature_tids[ntype_name] = l

    #read my nodes for each node type
    node_tids = {}
    node_data = schema_map["nid"]
    for ntype_name, ntype_info in node_data.items():
        v = []
        node_file_info = ntype_info["data"]
        for idx in range(len(node_file_info)):
            v.append((node_file_info[idx][1], node_file_info[idx][2]))
        node_tids[ntype_name] = v

    #read my edges for each edge type
    edge_tids = {}
    edge_datadict = {}
    edge_data = schema_map["eid"]
    for etype_name, etype_info in edge_data.items():
        if(etype_info["format"] == "csv"):
            edge_info = etype_info["data"]
            assert len(edge_info) == world_size

            data = np.loadtxt(edge_info[rank][0], delimiter=' ', dtype=np.int64)
            for idx, k in enumerate([constants.GLOBAL_SRC_ID, constants.GLOBAL_DST_ID, constants.GLOBAL_TYPE_EID, constants.ETYPE_ID]):
                if k in edge_datadict:
                    v = edge_datadict[k]
                    v = np.concatenate((v, data[:,idx]))
                    edge_datadict[k] = v
                else:
                    edge_datadict[k] = data[:,idx]

        v = []
        edge_file_info = etype_info["data"]
        for idx in range(len(edge_file_info)):
            v.append((edge_file_info[idx][1], edge_file_info[idx][2]))
        edge_tids[etype_name] = v
    print('[Rank: ', rank, '] Done reading edge_file: ', len(edge_datadict), edge_datadict[constants.GLOBAL_SRC_ID].shape)

    return node_tids, node_features, node_feature_tids, edge_datadict, edge_tids

