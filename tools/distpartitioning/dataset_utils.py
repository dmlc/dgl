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
    schema_map : dictionary
        this is the dictionary created by reading the graph metadata json file
        for the input graph dataset

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
    
    '''
    The structure of the node_data is as follows, which is present in the input metadata json file. 
       "node_data" : {
            "ntype0-name" : {
                "feat0-name" : {
                    "format" : {"name": "numpy"},
                    "data" :   [ #list of lists
                        ["<path>/feat-0.npy", 0, id_end0],
                        ["<path>/feat-1.npy", id_start1, id_end1],
                        ....
                        ["<path>/feat-<p-1>.npy", id_start<p-1>, id_end<p-1>]                
                    ]
                },
                "feat1-name" : {
                    "format" : {"name": "numpy"}, 
                    "data" : [ #list of lists
                        ["<path>/feat-0.npy", 0, id_end0],
                        ["<path>/feat-1.npy", id_start1, id_end1],
                        ....
                        ["<path>/feat-<p-1>.npy", id_start<p-1>, id_end<p-1>]                
                    ]
                }
            }
       }

    As shown above, the value for the key "node_data" is a dictionary object, which is 
    used to describe the feature data for each of the node-type names. Keys in this top-level
    dictionary are node-type names and value is a dictionary which captures all the features
    for the current node-type. Feature data is captured with keys being the feature-names and
    value is a dictionary object which has 2 keys namely format and data. Format entry is used
    to mention the format of the storage used by the node features themselves and "data" is used
    to mention all the files present for this given node feature.
    '''

    #iterate over the "node_data" dictionary in the schema_map
    #read the node features if exists
    #also keep track of the type_nids for which the node_features are read.
    dataset_features = schema_map[constants.STR_NODE_DATA]
    if((dataset_features is not None) and (len(dataset_features) > 0)):
        for ntype_name, ntype_feature_data in dataset_features.items():
            #ntype_feature_data is a dictionary
            #where key: feature_name, value: dictionary in which keys are "format", "data"
            node_feature_tids[ntype_name] = []
            for feat_name, feat_data in ntype_feature_data.items():
                assert len(feat_data[constants.STR_DATA]) == world_size
                assert feat_data[constants.STR_FORMAT][constants.STR_NAME] == constants.STR_NUMPY
                my_feat_data_fname = feat_data[constants.STR_DATA][rank] #this will be just the file name
                if (os.path.isabs(my_feat_data_fname)):
                    node_features[ntype_name+'/'+feat_name] = \
                            torch.from_numpy(np.load(my_feat_data_fname))
                else:
                    node_features[ntype_name+'/'+feat_name] = \
                            torch.from_numpy(np.load(os.path.join(input_dir, my_feat_data_fname)))

                node_feature_tids[ntype_name].append([feat_name, -1, -1])

    '''
        "node_type" : ["ntype0-name", "ntype1-name", ....], #m node types
        "num_nodes_per_chunk" : [
            [a0, a1, ...a<p-1>], #p partitions
            [b0, b1, ... b<p-1>], 
            ....
            [c0, c1, ..., c<p-1>] #no, of node types
        ],

    The "node_type" points to a list of all the node names present in the graph
    And "num_nodes_per_chunk" is used to mention no. of nodes present in each of the
    input nodes files. These node counters are used to compute the type_node_ids as
    well as global node-ids by using a simple cumulative summation and maitaining an
    offset counter to store the end of the current.

    '''

    #read my nodes for each node type
    node_tids = {}
    ntype_gnid_offset = {}
    gnid_offset = 0
    ntype_names = schema_map[constants.STR_NODE_TYPE]
    for idx, counts in enumerate(schema_map[constants.STR_NUM_NODES_PER_CHUNK]):
        ntype_name = ntype_names[idx]
        ntype_gnid_offset[ntype_name] = gnid_offset 
        type_nid_start = np.cumsum([0] + counts[:-1])
        type_nid_end = np.cumsum(counts)
        type_nid_ranges = list(zip(type_nid_start, type_nid_end))
        node_tids[ntype_name] = type_nid_ranges

        #go back and updated the tids for this ntype_name
        if (ntype_name in node_feature_tids):
            for item in node_feature_tids[ntype_name]:
                item[1] = type_nid_ranges[rank][0]
                item[2] = type_nid_ranges[rank][1]

        gnid_offset += type_nid_ranges[-1][1]

    #done build node_features locally. 
    if len(node_features) <= 0:
        print('[Rank: ', rank, '] This dataset does not have any node features')
    else:
        for k, v in node_features.items():
            print('[Rank: ', rank, '] node feature name: ', k, ', feature data shape: ', v.size())

    '''
    As shown in the case of nodes, edges also have very similar structures in 
    the dictionary.
    '''

    #read my edges for each edge type
    edge_tids = {}
    etype_geid_offset = {}
    geid_offset = 0
    etype_names = schema_map[constants.STR_EDGE_TYPE]
    etype_name_idmap = {e : idx for idx, e in enumerate(etype_names)}
    for idx, counts in enumerate(schema_map[constants.STR_NUM_EDGES_PER_CHUNK]):
        etype_name = etype_names[idx]
        etype_geid_offset[etype_name] = geid_offset
        type_eid_start = np.cumsum([0] + counts[:-1])
        type_eid_end = np.cumsum(counts)
        type_eid_ranges = list(zip(type_eid_start, type_eid_end))
        edge_tids[etype_name] = type_eid_ranges
        geid_offset += type_eid_ranges[-1][1]

    edge_datadict = {}
    edge_data = schema_map[constants.STR_EDGES]

    #read the edges files and store this data in memory.
    for col in [constants.GLOBAL_SRC_ID, constants.GLOBAL_DST_ID, \
            constants.GLOBAL_TYPE_EID, constants.ETYPE_ID]:
        edge_datadict[col] = []

    for etype_name, etype_info in edge_data.items():
        assert etype_info[constants.STR_FORMAT][constants.STR_NAME] == constants.STR_CSV

        edge_info = etype_info[constants.STR_DATA]
        assert len(edge_info) == world_size

        #edgetype strings are in canonical format, src_node_type:edge_type:dst_node_type
        tokens = etype_name.split(":")
        assert len(tokens) == 3

        src_ntype_name = tokens[0]
        rel_name = tokens[1]
        dst_ntype_name = tokens[2]

        data_df = csv.read_csv(edge_info[rank], read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True), 
                                    parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        #currently these are just type_edge_ids... which will be converted to global ids
        edge_datadict[constants.GLOBAL_SRC_ID].append(data_df['f0'].to_numpy() + ntype_gnid_offset[src_ntype_name])
        edge_datadict[constants.GLOBAL_DST_ID].append(data_df['f1'].to_numpy() + ntype_gnid_offset[dst_ntype_name])
        edge_datadict[constants.GLOBAL_TYPE_EID].append(np.arange(edge_tids[etype_name][rank][0],\
                edge_tids[etype_name][rank][1] ,dtype=np.int64))
        edge_datadict[constants.ETYPE_ID].append(etype_name_idmap[etype_name] * \
                np.ones(shape=(data_df['f0'].to_numpy().shape), dtype=np.int64))

    #stitch together to create the final data on the local machine
    for col in [constants.GLOBAL_SRC_ID, constants.GLOBAL_DST_ID, constants.GLOBAL_TYPE_EID, constants.ETYPE_ID]:
        edge_datadict[col] = np.concatenate(edge_datadict[col])

    assert edge_datadict[constants.GLOBAL_SRC_ID].shape == edge_datadict[constants.GLOBAL_DST_ID].shape
    assert edge_datadict[constants.GLOBAL_DST_ID].shape == edge_datadict[constants.GLOBAL_TYPE_EID].shape
    assert edge_datadict[constants.GLOBAL_TYPE_EID].shape == edge_datadict[constants.ETYPE_ID].shape
    print('[Rank: ', rank, '] Done reading edge_file: ', len(edge_datadict), edge_datadict[constants.GLOBAL_SRC_ID].shape)

    return node_tids, node_features, node_feature_tids, edge_datadict, edge_tids

