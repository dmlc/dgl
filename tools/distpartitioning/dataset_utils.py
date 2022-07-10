import os
import numpy as np
import constants
import torch

def get_ci_nodefeats(rank, feat_name):
    '''
    Function to generate dummpy node features for the CI test graph, which
    is described in constants.py in detail. 

    Parameters:
    -----------
    rank : int
        unique integer assigned to each process in a distributed execution.
    feat_name : string
        this string is used as feature name 

    Returns:
    --------
    numpy ndarray : 
        an array of pre-defined dimensions with predetermined data, based on 
        the rank of the process
    '''
    if (feat_name == "nodes-feat-01"):
        return np.ones((10,20), dtype=np.float)*(rank+1)
    elif (feat_name == "nodes-feat-02"):
        return np.ones((10,20), dtype=np.float)*(rank+1)*10

def get_ci_edges(rank, etype_name):
    '''
    Function to generated edges for the CI test graph. 

    Parameters:
    -----------
    rank : int
        unique integer assigned to each process in a distributed execution.
    etype_name : string
        this string is used as feature name 

    Returns:
    --------
    numpy ndarray : 
        an array is generated which represents the edges for the CI test graph
    '''
    nt1 = np.array([x for x in range(0, 20)]).astype(np.int64)
    nt2 = np.array([x for x in range(20, 40)]).astype(np.int64)
    if(etype_name == "etype-1"):
        # Here edges are between ntype-1 and ntype-1
        # ntype-1: gnids - [0, 20)
        # count rank0:10 , rank1:10 
        idx1 = np.random.random_integers(0, 19, (10,))
        idx2 = np.random.random_integers(0, 19, (10,))

        gsrc_id = nt1[idx1]
        gdst_id = nt1[idx2]

        if(rank == 0):
            type_eid = np.arange(10)
        else:
            type_eid = np.arange(10, 20)

        etype_id = np.ones((10,)) * 0
        return np.column_stack([gsrc_id, gdst_id, type_eid, etype_id]).astype(np.int64)

    elif(etype_name == "etype-2"):
        # Here edges are between ntype-2 and ntype-2
        # for ntype-2 gnids - [20, 40)
        # count rank0:10 rank1:10
        idx1 = np.random.random_integers(0, 19, (10,))
        idx2 = np.random.random_integers(0, 19, (10,))

        gsrc_id = nt2[idx1]
        gdst_id = nt2[idx2]

        if(rank == 0):
            type_eid = np.arange(10)
        else:
            type_eid = np.arange(10, 20)

        etype_id = np.ones((10,)) * 1
        return np.column_stack([gsrc_id, gdst_id, type_eid, etype_id]).astype(np.int64)

    elif(etype_name == "etype-3"):
        # Here edges are between ntype-1 and ntype-2
        # for ntype-1 [0, 20) and [20, 40)
        # count rank0:10 rank1:10
        idx1 = np.random.random_integers(0, 19, (10,))
        idx2 = np.random.random_integers(0, 19, (10,))

        gsrc_id = nt1[idx1]
        gdst_id = nt2[idx2]

        if(rank == 0):
            type_eid = np.arange(10)
        else:
            type_eid = np.arange(10, 20)

        etype_id = np.ones((10,)) * 2
        return np.column_stack([gsrc_id, gdst_id, type_eid, etype_id]).astype(np.int64)

def get_dataset(input_dir, graph_name, rank, num_node_weights, cimode):
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
            if(cimode == True):
                feats = get_ci_nodefeats(rank, feat_name)
                print('[Rank: ', rank, '] CI node features generated: ', feats.shape)
                node_features[ntype_name+'/'+feat_name] = torch.from_numpy(get_ci_nodefeats(rank, feat_name))
            else:
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

            if(cimode == True):
                data = get_ci_edges(rank, etype_name)
            else:
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

