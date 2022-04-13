import sys
import json
import torch
import numpy as np
import torch.distributed as dist

import dgl

def read_json(json_file):
    with open(json_file) as schema:
        val = json.load(schema)

    return val

def get_node_types(schema): 
    nid_ranges = schema['nid']
    nid_ranges = {key: np.array(nid_ranges[key]).reshape(1,2)
                    for key in nid_ranges}
    ntypes = [(key, nid_ranges[key][0,0]) for key in nid_ranges]
    ntypes.sort(key=lambda e: e[1])
    ntypes = [e[0] for e in ntypes]
    ntypes_map = {e: i for i, e in enumerate(ntypes)}

    return ntypes_map, ntypes

def send_metadata_json(metadata, size): 
    dist.gather_object(metadata, None,dst=0)

def get_metadata_json(size): 
    output = [None for _ in range(size)]
    dist.gather_object(None,output,dst=0)
    return output

def write_metadata_json(metadata_list, output_dir, graph_name):

    graph_metadata = {}

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

    with open('{}/{}.json'.format(output_dir, graph_name ), 'w') as outfile:
        json.dump(graph_metadata, outfile, sort_keys=True, indent=4)

