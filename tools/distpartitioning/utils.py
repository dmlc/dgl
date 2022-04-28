import torch
import numpy as np
import json
import dgl

def read_metis_partitions(part_file):
    """Utility method to read metis partitions, which is the output of 
    pm_dglpart2

    Parameters:
    -----------
    part_file : string
        file name which is the output of metis partitioning
        algorithm (pm_dglpart2, in the METIS installation).

    Returns:
    --------
    dictionary, mapping between global_nid <-> partition_id
    """
    metis_map = np.loadtxt(part_file, delimiter=' ', dtype=np.int64)
    return dict(metis_map)

def read_json(json_file):
    """Utility method to read a json file schema
    
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

def get_node_types(schema):
    """ Utility method to extract node_typename -> node_type mappings
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
    #Get the node_id ranges from the schema
    global_nid_ranges = schema['nid']
    global_nid_ranges = {key: np.array(global_nid_ranges[key]).reshape(1,2)
                    for key in global_nid_ranges}

    #Create an array with the starting id for each node_type and sort
    ntypes = [(key, global_nid_ranges[key][0,0]) for key in global_nid_ranges]
    ntypes.sort(key=lambda e: e[1])

    #Create node_typename -> node_type dictionary
    ntypes = [e[0] for e in ntypes]
    ntypes_map = {e: i for i, e in enumerate(ntypes)}

    return ntypes_map, ntypes

def write_metadata_json(metadata_list, output_dir, graph_name):
    """Merge json schema's from each of the rank's on rank-0. 
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
            edge_map[k].append(metadata_list[idx]["edge_map"][k][0])
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
            node_map[k].append(metadata_list[idx]["node_map"][k][0])
    graph_metadata["node_map"] = node_map

    graph_metadata["ntypes"] = metadata_list[0]["ntypes"]
    graph_metadata["num_edges"] = sum([metadata_list[i]["num_edges"] for i in range(len(metadata_list))])
    graph_metadata["num_nodes"] = sum([metadata_list[i]["num_nodes"] for i in range(len(metadata_list))])
    graph_metadata["num_parts"] = metadata_list[0]["num_parts"]
    graph_metadata["part_method"] = metadata_list[0]["part_method"]

    for i in range(len(metadata_list)):
        graph_metadata["part-{}".format(i)] = metadata_list[i]["part-{}".format(i)]

    with open('{}/{}.json'.format(output_dir, graph_name), 'w') as outfile: 
        json.dump(graph_metadata, outfile, sort_keys=True, indent=4)

def augment_edge_data(edge_data, partitions):
    """Add partition-id (rank which owns an edge) column to the edge_data.
    
    Parameters:
    -----------
    edge_data : numpy ndarray
        Edge information as read from the xxx_edges.txt file
    partitions : dictionary
        METIS assigned partitions to each of the nodes in the input graph

    Returns:
    --------
    numpy ndarray
        original edge_data with one additional column
    """
    tgtrank = [partitions[x[1]] for x in edge_data]
    return np.c_[edge_data, tgtrank]

def augment_node_data(node_data, partitions): 
    """Utility function to add auxilary columns to the node_data numpy ndarray.

    Parameters:
    -----------
    node_data : numpy ndarray
        Node information as read from xxx_nodes.txt file
    partitions : dictionary 
        METIS assigned partitions for a graph

    Returns:
    --------
    numpy ndarray
        original node_data, numpy ndarray, with two additional columns
    """
    global_nids = np.arange(node_data.shape[0])
    node_data_aug = np.c_[node_data, global_nids]

    tgtrank = [partitions[x[6]] for x in node_data_aug]
    return np.c_[node_data_aug, tgtrank]

def read_nodes_file(nodes_file):
    """Utility function to read xxx_nodes.txt file
    
    Parameters:
    -----------
    nodesfile : string
        Graph file for nodes in the input graph
    
    Returns:
    --------
    numpy ndarray
        node data as in the xxx_nodes.txt file
    """
    if nodes_file == "" or nodes_file == None:
        return None

    # Read the file from here.
    # Assuming the nodes file is a numpy file
    # nodes.txt file is of the following format
    # <node_type> <weight1> <weight2> <weight3> <weight4> <global_type_nid> <attributes>
    # For the ogb-mag dataset, nodes.txt is of the above format.
    #
    nodes_data = np.loadtxt(nodes_file, delimiter=' ', dtype='int64')
    return nodes_data

def read_edges_file(edge_file):
    """ Utility function to read xxx_edges.txt file

    Parameters:
    -----------
    edge_file : string
        Graph file for edges in the input graph

    Returns:
    --------
    numpy ndarray
        edge data as in the xxx_edges.txt file

    """
    if edge_file == "" or edge_file == None:
        return None

    #Read the file from here.
    #<global_src_id> <global_dst_id> <type_eid> <etype> <attributes>
    # global_src_id -- global idx for the source node ... line # in the graph_nodes.txt
    # global_dst_id -- global idx for the destination id node ... line # in the graph_nodes.txt

    edge_data = np.loadtxt(edge_file , delimiter=' ', dtype = 'int64')
    return edge_data

def read_node_features_file (nodes_features_file):
    """Utility function to load tensors from a file

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

def read_edge_features_file( edge_features_file ):
    """ Utility function to load tensors from a file

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
    """Utility function to serialize node_features in node_file file

    Parameters:
    -----------
    node_features : dictionary
        dictionary storing ntype <-> list of features
    node_file     : string 
        File in which the node information is serialized
    """
    dgl.data.utils.save_tensors(node_file, node_features)
