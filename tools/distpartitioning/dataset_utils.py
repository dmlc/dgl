import logging
import os

import numpy as np
import pyarrow
import torch
from pyarrow import csv

import constants
from utils import get_idranges, map_partid_rank


def get_dataset(input_dir, graph_name, rank, world_size, num_parts, schema_map):
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
    num_parts : int
        total number of output graph partitions
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
    dictionary
        Data read from numpy files for all the edge features in this dataset. This dictionary's keys
        are feature names and values are tensors data representing edge feature data.
    dictionary
        This dictionary is used for identifying the global-id range for the associated edge features
        present in the previous return value. The keys are edge-type names and values are triplets.
        Each triplet consists of edge-feature name and starting and ending points of the range of 
        tids representing the corresponding edge feautres.
    """

    #node features dictionary
    #TODO: With the new file format, It is guaranteed that the input dataset will have 
    #no. of nodes with features (node-features) files and nodes metadata will always be the same.
    #This means the dimension indicating the no. of nodes in any node-feature files and the no. of
    #nodes in the corresponding nodes metadata file will always be the same. With this guarantee, 
    #we can eliminate the `node_feature_tids` dictionary since the same information is also populated
    #in the `node_tids` dictionary. This will be remnoved in the next iteration of code changes.
    node_features = {}
    node_feature_tids = {}
    
    '''
    The structure of the node_data is as follows, which is present in the input metadata json file. 
       "node_data" : {
            "ntype0-name" : {
                "feat0-name" : {
                    "format" : {"name": "numpy"},
                    "data" :   [ #list
                        "<path>/feat-0.npy",
                        "<path>/feat-1.npy",
                        ....
                        "<path>/feat-<p-1>.npy"
                    ]
                },
                "feat1-name" : {
                    "format" : {"name": "numpy"}, 
                    "data" : [ #list 
                        "<path>/feat-0.npy",
                        "<path>/feat-1.npy",
                        ....
                        "<path>/feat-<p-1>.npy"
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

    Data read from each of the node features file is a multi-dimensional tensor data and is read
    in numpy format, which is also the storage format of node features on the permanent storage.

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

    Since nodes are NOT actually associated with any additional metadata, w.r.t to the processing
    involved in this pipeline this information is not needed to be stored in files. This optimization
    saves a considerable amount of time when loading massively large datasets for paritioning. 
    As opposed to reading from files and performing shuffling process each process/rank generates nodes
    which are owned by that particular rank. And using the "num_nodes_per_chunk" information each
    process can easily compute any nodes per-type node_id and global node_id.
    The node-ids are treated as int64's in order to support billions of nodes in the input graph.
    '''

    #read my nodes for each node type
    node_tids, ntype_gnid_offset = get_idranges(schema_map[constants.STR_NODE_TYPE], 
                                    schema_map[constants.STR_NUM_NODES_PER_CHUNK],
                                    num_chunks=num_parts)

    #iterate over the "node_data" dictionary in the schema_map
    #read the node features if exists
    #also keep track of the type_nids for which the node_features are read.
    dataset_features = schema_map[constants.STR_NODE_DATA]
    if((dataset_features is not None) and (len(dataset_features) > 0)):
        for ntype_name, ntype_feature_data in dataset_features.items():
            for feat_name, feat_data in ntype_feature_data.items():
                assert feat_data[constants.STR_FORMAT][constants.STR_NAME] == constants.STR_NUMPY

                # It is guaranteed that num_chunks is always greater 
                # than num_partitions. 
                num_chunks = len(feat_data[constants.STR_DATA])
                read_list = np.array_split(np.arange(num_chunks), num_parts)
                for local_part_id in range(num_parts):
                    if map_partid_rank(local_part_id, world_size) == rank:
                        nfeat = []
                        nfeat_tids = []
                        for idx in read_list[local_part_id]:
                            nfeat_file = feat_data[constants.STR_DATA][idx]
                            if not os.path.isabs(nfeat_file):
                                nfeat_file = os.path.join(input_dir, nfeat_file)
                            logging.info(f'Loading node feature[{feat_name}] of ntype[{ntype_name}] from {nfeat_file}')
                            nfeat.append(np.load(nfeat_file))
                        nfeat = np.concatenate(nfeat) if len(nfeat) != 0 else np.array([])
                        node_features[ntype_name+"/"+feat_name+"/"+str(local_part_id//world_size)] = torch.from_numpy(nfeat)
                        nfeat_tids.append(node_tids[ntype_name][local_part_id])
                        node_feature_tids[ntype_name+"/"+feat_name+"/"+str(local_part_id//world_size)] = nfeat_tids

    #done building node_features locally. 
    if len(node_features) <= 0:
        logging.info(f'[Rank: {rank}] This dataset does not have any node features')
    else:
        assert len(node_features) == len(node_feature_tids)

        # Note that the keys in the node_features dictionary are as follows:
        # `ntype_name/feat_name/local_part_id`. 
        #   where ntype_name and feat_name are self-explanatory, and 
        #   local_part_id indicates the partition-id, in the context of current
        #   process which take the values 0, 1, 2, ....
        for feat_name, feat_info  in node_features.items():
            logging.info(f'[Rank: {rank}] node feature name: {feat_name}, feature data shape: {feat_info.size()}')

            tokens = feat_name.split("/")
            assert len(tokens) == 3

            # Get the range of type ids which are mapped to the current node.
            tids = node_feature_tids[feat_name]

            # Iterate over the range of type ids for the current node feature
            # and count the number of features for this feature name.
            count = tids[0][1] - tids[0][0]
            assert count == feat_info.size()[0]


    '''
    Reading edge features now.
    The structure of the edge_data is as follows, which is present in the input metadata json file. 
       "edge_data" : {
            "etype0-name" : {
                "feat0-name" : {
                    "format" : {"name": "numpy"},
                    "data" :   [ #list
                        "<path>/feat-0.npy",
                        "<path>/feat-1.npy",
                        ....
                        "<path>/feat-<p-1>.npy"
                    ]
                },
                "feat1-name" : {
                    "format" : {"name": "numpy"}, 
                    "data" : [ #list 
                        "<path>/feat-0.npy",
                        "<path>/feat-1.npy",
                        ....
                        "<path>/feat-<p-1>.npy"
                    ]
                }
            }
       }

    As shown above, the value for the key "edge_data" is a dictionary object, which is 
    used to describe the feature data for each of the edge-type names. Keys in this top-level
    dictionary are edge-type names and value is a dictionary which captures all the features
    for the current edge-type. Feature data is captured with keys being the feature-names and
    value is a dictionary object which has 2 keys namely `format` and `data`. Format entry is used
    to mention the format of the storage used by the node features themselves and "data" is used
    to mention all the files present for this given node feature.

    Data read from each of the node features file is a multi-dimensional tensor data and is read
    in numpy format, which is also the storage format of node features on the permanent storage.
    '''
    edge_features = {}
    edge_feature_tids = {}

    # Read edges for each edge type that are processed by the currnet process.
    edge_tids, _ = get_idranges(schema_map[constants.STR_EDGE_TYPE], 
                                    schema_map[constants.STR_NUM_EDGES_PER_CHUNK], num_parts)

    # Iterate over the "edge_data" dictionary in the schema_map.
    # Read the edge features if exists.
    # Also keep track of the type_eids for which the edge_features are read.
    dataset_features = schema_map[constants.STR_EDGE_DATA]
    if dataset_features and (len(dataset_features) > 0):
        for etype_name, etype_feature_data in dataset_features.items():
            for feat_name, feat_data in etype_feature_data.items():
                assert feat_data[constants.STR_FORMAT][constants.STR_NAME] == constants.STR_NUMPY
                num_chunks = len(feat_data[constants.STR_DATA])
                read_list = np.array_split(np.arange(num_chunks), num_parts)
                for local_part_id in range(num_parts):
                    if map_partid_rank(local_part_id, world_size) == rank:
                        efeats = []
                        efeat_tids = []
                        for idx in read_list[local_part_id]:
                            feature_fname = feat_data[constants.STR_DATA][idx]
                            if (os.path.isabs(feature_fname)):
                                logging.info(f'Loading numpy from {feature_fname}')
                                efeats.append(torch.from_numpy(np.load(feature_fname)))
                            else:
                                numpy_path = os.path.join(input_dir, feature_fname)
                                logging.info(f'Loading numpy from {numpy_path}')
                                efeats.append(torch.from_numpy(np.load(numpy_path)))
                        efeat_tids.append(edge_tids[etype_name][local_part_id])
                        edge_features[etype_name+'/'+feat_name+"/"+str(local_part_id//world_size)] = torch.from_numpy(np.concatenate(efeats))
                        edge_feature_tids[etype_name+"/"+feat_name+"/"+str(local_part_id//world_size)] = efeat_tids

    # Done with building node_features locally. 
    if len(edge_features) <= 0:
        logging.info(f'[Rank: {rank}] This dataset does not have any edge features')
    else:
        assert len(edge_features) == len(edge_feature_tids)

        for k, v in edge_features.items():
            logging.info(f'[Rank: {rank}] edge feature name: {k}, feature data shape: {v.shape}')
            tids = edge_feature_tids[k]
            count = tids[0][1] - tids[0][0]
            assert count == v.size()[0]

    '''
    Code below is used to read edges from the input dataset with the help of the metadata json file
    for the input graph dataset. 
    In the metadata json file, we expect the following key-value pairs to help read the edges of the 
    input graph. 

    "edge_type" : [ # a total of n edge types
        canonical_etype_0, 
        canonical_etype_1, 
        ..., 
        canonical_etype_n-1
    ]

    The value for the key is a list of strings, each string is associated with an edgetype in the input graph.
    Note that these strings are in canonical edgetypes format. This means, these edge type strings follow the
    following naming convention: src_ntype:etype:dst_ntype. src_ntype and dst_ntype are node type names of the 
    src and dst end points of this edge type, and etype is the relation name between src and dst ntypes. 

    The files in which edges are present and their storage format are present in the following key-value pair: 
    
    "edges" : {
        "canonical_etype_0" : {
            "format" : { "name" : "csv", "delimiter" : " " }, 
            "data" : [
                filename_0, 
                filename_1, 
                filename_2, 
                ....
                filename_<p-1>
            ]
        },
    }

    As shown above the "edges" dictionary value has canonical edgetypes as keys and for each canonical edgetype
    we have "format" and "data" which describe the storage format of the edge files and actual filenames respectively. 
    Please note that each edgetype data is split in to `p` files, where p is the no. of partitions to be made of
    the input graph.

    Each edge file contains two columns representing the source per-type node_ids and destination per-type node_ids
    of any given edge. Since these are node-ids as well they are read in as int64's.
    '''

    #read my edges for each edge type
    etype_names = schema_map[constants.STR_EDGE_TYPE]
    etype_name_idmap = {e : idx for idx, e in enumerate(etype_names)}
    edge_tids, _ = get_idranges(schema_map[constants.STR_EDGE_TYPE],
                    schema_map[constants.STR_NUM_EDGES_PER_CHUNK],
                    num_chunks=num_parts)

    edge_datadict = {}
    edge_data = schema_map[constants.STR_EDGES]

    #read the edges files and store this data in memory.
    for col in [constants.GLOBAL_SRC_ID, constants.GLOBAL_DST_ID, \
            constants.GLOBAL_TYPE_EID, constants.ETYPE_ID]:
        edge_datadict[col] = []

    for etype_name, etype_info in edge_data.items():
        assert etype_info[constants.STR_FORMAT][constants.STR_NAME] == constants.STR_CSV

        edge_info = etype_info[constants.STR_DATA]

        #edgetype strings are in canonical format, src_node_type:edge_type:dst_node_type
        tokens = etype_name.split(":")
        assert len(tokens) == 3

        src_ntype_name = tokens[0]
        dst_ntype_name = tokens[2]

        num_chunks = len(edge_info)
        read_list = np.array_split(np.arange(num_chunks), num_parts)
        src_ids = []
        dst_ids = []

        curr_partids = []
        for part_id in range(num_parts):
            if map_partid_rank(part_id, world_size) == rank:
                curr_partids.append(read_list[part_id])

        for idx in np.concatenate(curr_partids):
            edge_file = edge_info[idx]
            if not os.path.isabs(edge_file):
                edge_file = os.path.join(input_dir, edge_file)
            logging.info(f'Loading edges of etype[{etype_name}] from {edge_file}')

            read_options=pyarrow.csv.ReadOptions(use_threads=True, block_size=4096, autogenerate_column_names=True)
            parse_options=pyarrow.csv.ParseOptions(delimiter=' ')
            with pyarrow.csv.open_csv(edge_file, read_options=read_options, parse_options=parse_options) as reader:
                for next_chunk in reader:
                    if next_chunk is None:
                        break

                    next_table = pyarrow.Table.from_batches([next_chunk])
                    src_ids.append(next_table['f0'].to_numpy())
                    dst_ids.append(next_table['f1'].to_numpy())

        src_ids = np.concatenate(src_ids)
        dst_ids = np.concatenate(dst_ids)

        #currently these are just type_edge_ids... which will be converted to global ids
        edge_datadict[constants.GLOBAL_SRC_ID].append(src_ids + ntype_gnid_offset[src_ntype_name][0, 0])
        edge_datadict[constants.GLOBAL_DST_ID].append(dst_ids + ntype_gnid_offset[dst_ntype_name][0, 0])
        edge_datadict[constants.ETYPE_ID].append(etype_name_idmap[etype_name] * \
            np.ones(shape=(src_ids.shape), dtype=np.int64))

        for local_part_id in range(num_parts):
            if (map_partid_rank(local_part_id, world_size) == rank):
                edge_datadict[constants.GLOBAL_TYPE_EID].append(np.arange(edge_tids[etype_name][local_part_id][0],\
                    edge_tids[etype_name][local_part_id][1] ,dtype=np.int64))

    #stitch together to create the final data on the local machine
    for col in [constants.GLOBAL_SRC_ID, constants.GLOBAL_DST_ID, constants.GLOBAL_TYPE_EID, constants.ETYPE_ID]:
        edge_datadict[col] = np.concatenate(edge_datadict[col])

    assert edge_datadict[constants.GLOBAL_SRC_ID].shape == edge_datadict[constants.GLOBAL_DST_ID].shape
    assert edge_datadict[constants.GLOBAL_DST_ID].shape == edge_datadict[constants.GLOBAL_TYPE_EID].shape
    assert edge_datadict[constants.GLOBAL_TYPE_EID].shape == edge_datadict[constants.ETYPE_ID].shape
    logging.info(f'[Rank: {rank}] Done reading edge_file: {len(edge_datadict)}, {edge_datadict[constants.GLOBAL_SRC_ID].shape}')
    logging.info(f'Rank: {rank} edge_feat_tids: {edge_feature_tids}')

    return node_tids, node_features, node_feature_tids, edge_datadict, edge_tids, edge_features, edge_feature_tids

