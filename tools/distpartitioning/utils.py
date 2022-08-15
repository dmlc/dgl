import os
import torch
import numpy as np
import json
import dgl
import constants
import logging
import platform
from pathlib import Path

import pyarrow
from pyarrow import csv

def read_ntype_partition_files(schema_map, input_dir):
    """
    Utility method to read the partition id mapping for each node.
    For each node type, there will be an file, in the input directory argument
    containing the partition id mapping for a given nodeid. 

    Parameters:
    -----------
    schema_map : dictionary
        dictionary created by reading the input metadata json file
    input_dir : string
        directory in which the node-id to partition-id mappings files are 
        located for each of the node types in the input graph

    Returns:
    --------
    numpy array : 
        array of integers representing mapped partition-ids for a given node-id.
        The line number, in these files, are used as the type_node_id in each of 
        the files. The index into this array will be the homogenized node-id and 
        value will be the partition-id for that node-id (index). Please note that 
        the partition-ids of each node-type are stacked together vertically and 
        in this way heterogenous node-ids are converted to homogenous node-ids. 
    """
    assert os.path.isdir(input_dir)

    #iterate over the node types and extract the partition id mappings
    part_ids = []
    ntype_names = schema_map[constants.STR_NODE_TYPE]
    for ntype in ntype_names:
        df = csv.read_csv(os.path.join(input_dir, '{}.txt'.format(ntype)), \
                read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True), \
                parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        ntype_partids = df['f0'].to_numpy()
        part_ids.append(ntype_partids)
    return np.concatenate(part_ids)

def read_json(json_file):
    """
    Utility method to read a json file schema
    
    Parameters:
    -----------
    json_file : string
        file name for the json schema

    Returns:
    --------
    dictionary, as serialized in the json_file
    """
    with open(json_file) as schema:
        val = json.load(schema)

    return val

def get_ntype_featnames(ntype_name, schema_map): 
    """
    Retrieves node feature names for a given node_type

    Parameters:
    -----------
    ntype_name : string
        a string specifying a node_type name

    schema : dictionary
        metadata json object as a dictionary, which is read from the input
        metadata file from the input dataset

    Returns:
    --------
    list : 
        a list of feature names for a given node_type
    """
    ntype_featdict = schema_map[constants.STR_NODE_DATA]
    if (ntype_name in ntype_featdict):
        featnames = []
        ntype_info = ntype_featdict[ntype_name]
        for k, v in ntype_info.items(): 
            featnames.append(k)
        return featnames
    else: 
        return []

def get_node_types(schema_map):
    """ 
    Utility method to extract node_typename -> node_type mappings
    as defined by the input schema

    Parameters:
    -----------
    schema_map : dictionary
        Input schema from which the node_typename -> node_type
        dictionary is created.

    Returns:
    --------
    dictionary
        with keys as node type names and values as ids (integers)
    list
        list of ntype name strings
    dictionary
        with keys as ntype ids (integers) and values as node type names
    """
    ntypes = schema_map[constants.STR_NODE_TYPE]
    ntype_ntypeid_map = {e : i for i, e in enumerate(ntypes)}
    ntypeid_ntype_map = {i : e for i, e in enumerate(ntypes)}
    return ntype_ntypeid_map, ntypes, ntypeid_ntype_map

def get_gnid_range_map(node_tids): 
    """
    Retrieves auxiliary dictionaries from the metadata json object

    Parameters:
    -----------
    node_tids: dictionary
        This dictionary contains the information about nodes for each node_type.
        Typically this information contains p-entries, where each entry has a file-name, 
        starting and ending type_node_ids for the nodes in this file. Keys in this dictionary
        are the node_type and value is a list of lists. Each individual entry in this list has
        three items: file-name, starting type_nid and ending type_nid

    Returns:
    --------
    dictionary : 
        a dictionary where keys are node_type names and values are global_nid range, which is a tuple.

    """
    ntypes_gid_range = {} 
    offset = 0
    for k, v in node_tids.items(): 
        ntypes_gid_range[k] = [offset + int(v[0][0]), offset + int(v[-1][1])]
        offset += int(v[-1][1])

    return ntypes_gid_range

def write_metadata_json(metadata_list, output_dir, graph_name):
    """
    Merge json schema's from each of the rank's on rank-0. 
    This utility function, to be used on rank-0, to create aggregated json file.

    Parameters:
    -----------
    metadata_list : list of json (dictionaries)
        a list of json dictionaries to merge on rank-0
    output_dir : string
        output directory path in which results are stored (as a json file)
    graph-name : string
        a string specifying the graph name
    """
    #Initialize global metadata
    graph_metadata = {}

    #Merge global_edge_ids from each json object in the input list
    edge_map = {}
    x = metadata_list[0]["edge_map"]
    for k in x:
        edge_map[k] = []
        for idx in range(len(metadata_list)):
            edge_map[k].append([int(metadata_list[idx]["edge_map"][k][0][0]),int(metadata_list[idx]["edge_map"][k][0][1])])
    graph_metadata["edge_map"] = edge_map

    graph_metadata["etypes"] = metadata_list[0]["etypes"]
    graph_metadata["graph_name"] = metadata_list[0]["graph_name"]
    graph_metadata["halo_hops"] = metadata_list[0]["halo_hops"]

    #Merge global_nodeids from each of json object in the input list
    node_map = {}
    x = metadata_list[0]["node_map"]
    for k in x:
        node_map[k] = []
        for idx in range(len(metadata_list)):
            node_map[k].append([int(metadata_list[idx]["node_map"][k][0][0]), int(metadata_list[idx]["node_map"][k][0][1])])
    graph_metadata["node_map"] = node_map

    graph_metadata["ntypes"] = metadata_list[0]["ntypes"]
    graph_metadata["num_edges"] = int(sum([metadata_list[i]["num_edges"] for i in range(len(metadata_list))]))
    graph_metadata["num_nodes"] = int(sum([metadata_list[i]["num_nodes"] for i in range(len(metadata_list))]))
    graph_metadata["num_parts"] = metadata_list[0]["num_parts"]
    graph_metadata["part_method"] = metadata_list[0]["part_method"]

    for i in range(len(metadata_list)):
        graph_metadata["part-{}".format(i)] = metadata_list[i]["part-{}".format(i)]

    with open('{}/metadata.json'.format(output_dir), 'w') as outfile: 
        json.dump(graph_metadata, outfile, sort_keys=False, indent=4)

def augment_edge_data(edge_data, part_ids, edge_tids, rank, world_size):
    """
    Add partition-id (rank which owns an edge) column to the edge_data.
    
    Parameters:
    -----------
    edge_data : numpy ndarray
        Edge information as read from the xxx_edges.txt file
    part_ids : numpy array
        array of part_ids indexed by global_nid
    """
    #add global_nids to the node_data
    etype_offset = {}
    offset = 0
    for etype_name, tid_range in edge_tids.items(): 
        assert int(tid_range[0][0]) == 0
        assert len(tid_range) == world_size
        etype_offset[etype_name] = offset + int(tid_range[0][0])
        offset += int(tid_range[-1][1])

    global_eids = []
    for etype_name, tid_range in edge_tids.items(): 
        global_eid_start = etype_offset[etype_name]
        begin = global_eid_start + int(tid_range[rank][0])
        end = global_eid_start + int(tid_range[rank][1])
        global_eids.append(np.arange(begin, end, dtype=np.int64))
    global_eids = np.concatenate(global_eids)
    assert global_eids.shape[0] == edge_data[constants.ETYPE_ID].shape[0]
    edge_data[constants.GLOBAL_EID] = global_eids

    #assign the owner process/rank for each edge 
    edge_data[constants.OWNER_PROCESS] = part_ids[edge_data[constants.GLOBAL_DST_ID]]

def read_edges_file(edge_file, edge_data_dict):
    """ 
    Utility function to read xxx_edges.txt file

    Parameters:
    -----------
    edge_file : string
        Graph file for edges in the input graph

    Returns:
    --------
    dictionary
        edge data as read from xxx_edges.txt file and columns are stored
        in a dictionary with key-value pairs as column-names and column-data. 
    """
    if edge_file == "" or edge_file == None:
        return None

    #Read the file from here.
    #<global_src_id> <global_dst_id> <type_eid> <etype> <attributes>
    # global_src_id -- global idx for the source node ... line # in the graph_nodes.txt
    # global_dst_id -- global idx for the destination id node ... line # in the graph_nodes.txt

    edge_data_df = csv.read_csv(edge_file, read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True), 
                                    parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
    edge_data_dict = {}
    edge_data_dict[constants.GLOBAL_SRC_ID] = edge_data_df['f0'].to_numpy()
    edge_data_dict[constants.GLOBAL_DST_ID] = edge_data_df['f1'].to_numpy()
    edge_data_dict[constants.GLOBAL_TYPE_EID] = edge_data_df['f2'].to_numpy()
    edge_data_dict[constants.ETYPE_ID] = edge_data_df['f3'].to_numpy()
    return edge_data_dict

def read_node_features_file(nodes_features_file):
    """
    Utility function to load tensors from a file

    Parameters:
    -----------
    nodes_features_file : string
        Features file for nodes in the graph

    Returns:
    --------
    dictionary
        mappings between ntype and list of features
    """

    node_features = dgl.data.utils.load_tensors(nodes_features_file, False)
    return node_features

def read_edge_features_file(edge_features_file):
    """ 
    Utility function to load tensors from a file

    Parameters:
    -----------
    edge_features_file : string
        Features file for edges in the graph

    Returns:
    --------
    dictionary
        mappings between etype and list of features
    """
    edge_features = dgl.data.utils.load_tensors(edge_features_file, True)
    return edge_features

def write_node_features(node_features, node_file):
    """
    Utility function to serialize node_features in node_file file

    Parameters:
    -----------
    node_features : dictionary
        dictionary storing ntype <-> list of features
    node_file     : string 
        File in which the node information is serialized
    """
    dgl.data.utils.save_tensors(node_file, node_features)

def write_edge_features(edge_features, edge_file): 
    """
    Utility function to serialize edge_features in edge_file file

    Parameters:
    -----------
    edge_features : dictionary
        dictionary storing etype <-> list of features
    edge_file     : string 
        File in which the edge information is serialized
    """
    dgl.data.utils.save_tensors(edge_file, edge_features)

def write_graph_dgl(graph_file, graph_obj): 
    """
    Utility function to serialize graph dgl objects

    Parameters:
    -----------
    graph_obj : dgl graph object
        graph dgl object, as created in convert_partition.py, which is to be serialized
    graph_file : string
        File name in which graph object is serialized
    """
    dgl.save_graphs(graph_file, [graph_obj])

def write_dgl_objects(graph_obj, node_features, edge_features, output_dir, part_id): 
    """
    Wrapper function to create dgl objects for graph, node-features and edge-features

    Parameters:
    -----------
    graph_obj : dgl object
        graph dgl object as created in convert_partition.py file
    node_features : dgl object
        Tensor data for node features
    edge_features : dgl object
        Tensor data for edge features
    output_dir : string
        location where the output files will be located
    part_id : int
        integer indicating the partition-id
    """

    part_dir = output_dir + '/part' + str(part_id)
    os.makedirs(part_dir, exist_ok=True)
    write_graph_dgl(os.path.join(part_dir ,'graph.dgl'), graph_obj)

    if node_features != None:
        write_node_features(node_features, os.path.join(part_dir, "node_feat.dgl"))

    if (edge_features != None):
        write_edge_features(edge_features, os.path.join(part_dir, "edge_feat.dgl"))

def get_idranges(names, counts): 
    """
    Utility function to compute typd_id/global_id ranges for both nodes and edges. 

    Parameters:
    -----------
    names : list of strings
        list of node/edge types as strings
    counts : list of lists
        each list contains no. of nodes/edges in a given chunk

    Returns:
    --------
    dictionary
        dictionary where the keys are node-/edge-type names and values are
        list of tuples where each tuple indicates the range of values for 
        corresponding type-ids. 
    dictionary
        dictionary where the keys are node-/edge-type names and value is a tuple.
        This tuple indicates the global-ids for the associated node-/edge-type.
    """
    gnid_start = 0
    gnid_end = gnid_start
    tid_dict = {}
    gid_dict = {}
    for idx, typename in enumerate(names): 
        type_counts = counts[idx]
        tid_start = np.cumsum([0] + type_counts[:-1])
        tid_end = np.cumsum(type_counts)
        tid_ranges = list(zip(tid_start, tid_end))

        type_start = tid_ranges[0][0]
        type_end = tid_ranges[-1][1]

        gnid_end += tid_ranges[-1][1]

        tid_dict[typename] = tid_ranges
        gid_dict[typename] = np.array([gnid_start, gnid_end]).reshape([1,2])

        gnid_start = gnid_end

    return tid_dict, gid_dict

def validate_json_file(rank, fname, num_parts):

    ff = Path(fname)
    assert ff.is_file()
    assert rank < num_parts

    metadata = read_json(fname)

    #node_type
    assert constants.STR_NODE_TYPE in metadata
    ntype_names = list(metadata[constants.STR_NODE_TYPE])

    #num_nodes_per_chunk
    assert constants.STR_NUM_NODES_PER_CHUNK in metadata

    #check for each node type chunks_per_nodetype are defined
    assert len(metadata[constants.STR_NUM_NODES_PER_CHUNK]) == len(ntype_names)

    #edge_type
    assert constants.STR_EDGE_TYPE in metadata
    etype_names = list(metadata[constants.STR_EDGE_TYPE])

    #verify the canonocal edge_types, with `:` as the separator
    for ename in etype_names:
        assert len(ename.split(':')) == 3

    #num_edges_per_chunk
    assert constants.STR_NUM_EDGES_PER_CHUNK in metadata
    assert len(metadata[constants.STR_NUM_EDGES_PER_CHUNK]) == len(etype_names)

    assert constants.STR_EDGES in metadata
    for etype, etype_val in metadata[constants.STR_EDGES].items():
        #etype_name and etype_val is a dictionary
        assert etype in etype_names
        assert constants.STR_DATA in etype_val
        my_edges_file = etype_val[constants.STR_DATA][rank]
        ff = Path(my_edges_file)
        assert ff.is_file()

        assert constants.STR_FORMAT in etype_val
        assert constants.STR_DELIMITER in etype_val[constants.STR_FORMAT]
        assert constants.STR_NAME in etype_val[constants.STR_FORMAT]

        assert ff.suffix in [
                            '{}{}'.format('.', constants.STR_CSV),
                            '{}{}'.format('.', constants.STR_TXT)
                            ]
        assert etype_val[constants.STR_FORMAT][constants.STR_NAME] in [
                            constants.STR_CSV, constants.STR_TXT
                            ]


    #assert node_data
    #Node features are optional
    if constants.STR_NODE_DATA in metadata:
        featinfo = metadata[constants.STR_NODE_DATA]
        featinfo_dict = {}

        for node_type, node_type_val in featinfo.items():
            for feat_name, feat_val in node_type_val.items():
                for k, v in feat_val.items():
                    if k == constants.STR_DATA:
                        my_node_features_file = v[rank]
                        assert isinstance(my_node_features_file, str)
                        file_path = Path(my_node_features_file)
                        assert file_path.is_file()
                        assert file_path.suffix in ['{}{}'.format('.', constants.STR_NUMPY)]
    logging.info(f'Validated the input json file successfully !!!')

if __name__ == "__main__":
    logging.basicConfig(level='INFO', format=f"[{platform.node()} %(levelname)s %(asctime)s PID:%(process)d] %(message)s")
    validate_json_file(15, "/fsx-dev/m5gnn/data/core_graph/metadata-0808.json", 16)
