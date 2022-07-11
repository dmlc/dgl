import os
import numpy as np
import constants
import torch

import pyarrow
from pyarrow import csv

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
    world_size : int
        total number of process in the current execution
    num_node_weights : int
        integer indicating the no. of weights each node is attributed with

    Return:
    -------
    dictionary
        where keys are node-type names and values are tuples. Each tuple represents the
        range of type ids read from a file by the current process. Please note that node
        data for each node type is split into "p" files and each one of these "p" files are
        read a process in the distributed graph partitioning pipeline
    dictionary
        Data read from numpy files for all the node features in this dataset. Dictionary built 
        using this data has keys as node feature names and values as tensor data representing 
        node features
    dictionary
        in which keys are node-type and values are a triplet. This triplet has node-feature name, 
        and range of tids for the node feature data read from files by the current process. Each
        node-type may have mutiple feature(s) and associated tensor data.
    dictionary
        Data read from edges.txt file and used to build a dictionary with keys as column names 
        and values as columns in the csv file. 
    dictionary
        in which keys are edge-type names and values are triplets. This triplet has edge-feature name, 
        and range of tids for theedge feature data read from the files by the current process. Each
        edge-type may have several edge features and associated tensor data.

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

            node_feature_tids[ntype_name].append([feat_name, my_feat_data[1], my_feat_data[2]])

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
        assert etype_info["format"] == "csv"

        edge_info = etype_info["data"]
        assert len(edge_info) == world_size

        data_df = csv.read_csv(edge_info[rank][0], read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True), 
                                    parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        edge_datadict[constants.GLOBAL_SRC_ID] = data_df['f0'].to_numpy()
        edge_datadict[constants.GLOBAL_DST_ID] = data_df['f1'].to_numpy()
        edge_datadict[constants.GLOBAL_TYPE_EID] = data_df['f2'].to_numpy()
        edge_datadict[constants.ETYPE_ID] = data_df['f3'].to_numpy()

        v = []
        edge_file_info = etype_info["data"]
        for idx in range(len(edge_file_info)):
            v.append((edge_file_info[idx][1], edge_file_info[idx][2]))
        edge_tids[etype_name] = v
    print('[Rank: ', rank, '] Done reading edge_file: ', len(edge_datadict), edge_datadict[constants.GLOBAL_SRC_ID].shape)

    return node_tids, node_features, node_feature_tids, edge_datadict, edge_tids

