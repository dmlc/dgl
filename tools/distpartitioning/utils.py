import os
import torch
import numpy as np
import json
import dgl
import constants

import pyarrow
from pyarrow import csv

def read_partitions_file(part_file, cimode):
    """
    Utility method to read metis partitions, which is the output of 
    pm_dglpart2

    Parameters:
    -----------
    part_file : string
        file name which is the output of metis partitioning
        algorithm (pm_dglpart2, in the METIS installation).
        This function expects each line in `part_file` to be formatted as 
        <global_nid> <part_id>
        and the contents of this file are sorted by <global_nid>. 

    Returns:
    --------
    numpy array
        array of part_ids and the idx is the <global_nid>
    """
    if(cimode):
        np.random.seed(0)
        arr = np.random.random_integers(0, 1, size=(constants.CI_GRAPH_NUM_NODES))
        print(arr)
        return arr

    partitions_map = np.loadtxt(part_file, delimiter=' ', dtype=np.int64)
    #as a precaution sort the lines based on the <global_nid>
    partitions_map = partitions_map[partitions_map[:,0].argsort()]
    return partitions_map[:,1]

def read_json(json_file, cimode):
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
    if(cimode):
        return json.loads(constants.CI_JSON_STRING)

    with open(json_file) as schema:
        val = json.load(schema)

    return val

def get_ntype_featnames(ntype_name, schema): 
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
    ntype_dict = schema["node_data"]
    if (ntype_name in ntype_dict):
        featnames = []
        ntype_info = ntype_dict[ntype_name]
        for k, v in ntype_info.items(): 
            featnames.append(k)
        return featnames
    else: 
        return []

def get_node_types(schema):
    """ 
    Utility method to extract node_typename -> node_type mappings
    as defined by the input schema

    Parameters:
    -----------
    schema : dictionary
        Input schema from which the node_typename -> node_type
        dictionary is created.

    Returns:
    --------
    dictionary, list 
        dictionary with ntype <-> type_nid mappings
        list of ntype strings
    """
    ntype_info = schema["nid"]
    ntypes = []
    for k in ntype_info.keys(): 
        ntypes.append(k)
    ntype_ntypeid_map = {e: i for i, e in enumerate(ntypes)}
    ntypeid_ntype_map = {str(i): e for i, e in enumerate(ntypes)}
    return ntype_ntypeid_map, ntypes, ntypeid_ntype_map

def get_edge_types(schema): 
    """
    schema : dictionary
        metadata json object as a dictionary which is part of the input dataset
        to this pipeline

    Returns:
    --------
    dictionary: 
        a dictionary where keys are edgetype names and values are edgetype_ids
    list : 
        a list of edge type names
    """

    global_eid_ranges = schema['eid']
    global_eid_ranges = {key: np.array(global_eid_ranges[key]).reshape(1,2)
                    for key in global_eid_ranges}
    etypes = [(key, global_eid_ranges[key][0, 0]) for key in global_eid_ranges]
    etypes.sort(key=lambda e: e[1])

    etypes = [e[0] for e in etypes]
    etypes_map = {e: i for i, e in enumerate(etypes)}

    return etypes_map, etypes

def get_ntypes_map(node_tids): 
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
    dictionary : 
        a dictionary where kesy are node_type names and values are total count of nodes for this 
        node_type

    """
    ntypes_gid_range = {} 
    offset = 0
    for k, v in node_tids.items(): 
        ntypes_gid_range[k] = [offset + int(v[0][0]), offset + int(v[-1][1])]
        offset += int(v[-1][1])

    node_type_id_count = {}
    ntypes = []
    for k in node_tids.keys(): 
        ntypes.append(k)

    idx = 0
    for k, v in node_tids.items(): 
        node_type_id_count[ str(idx) ] = int(v[-1][1])
        idx += 1

    return ntypes_gid_range, node_type_id_count

def write_metadata_json(metadata_list, output_dir, graph_name, cimode):
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

    if (cimode == False):
        with open('{}/{}.json'.format(output_dir, graph_name), 'w') as outfile:
            json.dump(graph_metadata, outfile, sort_keys=True, indent=4)
        return None
    else:
        return graph_metadata


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
    #global_eids = np.arange(id_offset, id_offset + len(edge_data[constants.GLOBAL_TYPE_EID]), dtype=np.int64)
    #edge_data[constants.GLOBAL_EID] = global_eids
    etype_offset = {}
    offset = 0
    for etype_name, tid_range in edge_tids.items(): 
        assert int(tid_range[0][0]) == 0
        assert len(tid_range) == world_size
        etype_offset[etype_name] = offset + int(tid_range[0][0])
        offset += int(tid_range[-1][1])

    global_eids = np.array([], dtype=np.int64)
    for etype_name, tid_range in edge_tids.items(): 
        global_eid_start = etype_offset[etype_name]
        begin = global_eid_start + int(tid_range[rank][0])
        end = global_eid_start + int(tid_range[rank][1])
        global_eids = np.concatenate((global_eids, np.arange(begin, end)))
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
    graph_obj : dgl object
        graph dgl object as created in convert_partition.py file

    node_features : dgl object
        Tensor data for node features

    edge_features : dgl object
        Tensor data for edge features
    """

    part_dir = output_dir + '/part' + str(part_id)
    os.makedirs(part_dir, exist_ok=True)
    #write_graph_dgl(os.path.join(part_dir ,'part'+str(part_id)), graph_obj)
    write_graph_dgl(os.path.join(part_dir ,'graph.dgl'), graph_obj)

    if node_features != None:
        write_node_features(node_features, os.path.join(part_dir, "node_feat.dgl"))

    if (edge_features != None):
        write_edge_features(edge_features, os.path.join(part_dir, "edge_feat.dgl"))
