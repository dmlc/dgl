import os
import sys
import constants
import numpy as np
import torch
import torch.distributed as dist
import argparse
from pathlib import Path

from utils import read_json, get_node_types, get_idranges

import pyarrow
import pyarrow.csv as csv

def get_proc_info():
    """
    helper function to get the rank, world_size parameters from the
    environment when `mpirun` is used to run this python program

    Returns:
    --------
    integer :
        rank of the current process
    integer :
        total no. of process used to run this program
    """
    local_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
    return local_rank, world_size


def gen_edge_files(schema_map, output):
    """
    Function to create edges files to be consumed by ParMETIS
    for partitioning purposes.

    Parameters:
    -----------
    schema_map : json dictionary
        dictionary created by reading the metadata.json file for the input dataset
    output : string
        location of storing the node-weights and edge files for ParMETIS
    """
    rank, world_size = get_proc_info()
    type_nid_dict, ntype_gnid_offset = get_idranges(schema_map[constants.STR_NODE_TYPE],
                                        schema_map[constants.STR_NUM_NODES_PER_CHUNK])

    #Regenerate edge files here...▒
    # in the following format....
    edge_data = schema_map[constants.STR_EDGES]
    etype_names = schema_map[constants.STR_EDGE_TYPE]
    etype_name_idmap = {e : idx for idx, e in enumerate(etype_names)}
    edge_tids, _ = get_idranges(schema_map[constants.STR_EDGE_TYPE],
                                    schema_map[constants.STR_NUM_EDGES_PER_CHUNK])

    outdir = Path(params.output)
    os.makedirs(outdir, exist_ok=True)
    edge_files = []
    num_parts = len(schema_map[constants.STR_NUM_EDGES_PER_CHUNK][0])
    for etype_name, etype_info in edge_data.items():

        edge_info = etype_info[constants.STR_DATA]

        #edgetype strings are in canonical format, src_node_type:edge_type:dst_node_type
        tokens = etype_name.split(":")
        assert len(tokens) == 3

        src_ntype_name = tokens[0]
        rel_name = tokens[1]
        dst_ntype_name = tokens[2]

        data_df = csv.read_csv(edge_info[rank],
                      read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
                      parse_options=pyarrow.csv.ParseOptions(delimiter=' '))
        data_f0 = data_df['f0'].to_numpy()
        data_f1 = data_df['f1'].to_numpy()

        global_src_id = data_f0 + ntype_gnid_offset[src_ntype_name][0, 0]
        global_dst_id = data_f1 + ntype_gnid_offset[dst_ntype_name][0, 0]
        cols = [global_src_id, global_dst_id]
        col_names = ["global_src_id", "global_dst_id"]
        #global_type_eid = np.arange(edge_tids[etype_name][rank][0],\
        #     edge_tids[etype_name][rank][1] ,dtype=np.int64)
        #etype_id = etype_name_idmap[etype_name] * \
        #        np.ones(shape=(data_df['f0'].to_numpy().shape), dtype=np.int64)

        out_file = edge_info[rank].split("/")[-1]
        out_file = os.path.join(outdir, 'edges_{}'.format(out_file))
        options = csv.WriteOptions(include_header=False, delimiter=" ")
        options.delimiter = " "

        csv.write_csv(pyarrow.Table.from_arrays(cols, names=col_names), out_file, options)

        edge_files.append(out_file)

    return edge_files

def read_node_features(schema_map, tgt_ntype_name, feat_names):

    rank, world_size = get_proc_info()
    node_features = {}
    if constants.STR_NODE_DATA in schema_map:
        dataset_features = schema_map[constants.STR_NODE_DATA]
        if((dataset_features is not None) and (len(dataset_features) > 0)):
            for ntype_name, ntype_feature_data in dataset_features.items():
                if ntype_name != tgt_ntype_name : 
                    continue
                #ntype_feature_data is a dictionary
                #where key: feature_name, value: dictionary in which keys are "format", "data"
                for feat_name, feat_data in ntype_feature_data.items():
                    if feat_name in feat_names:
                        my_feat_data_fname = feat_data[constants.STR_DATA][rank]
                        print('Reading: ', my_feat_data_fname)
                        if (os.path.isabs(my_feat_data_fname)):
                            node_features[feat_name] = np.load(my_feat_data_fname)
                        else:
                            node_features[feat_name] = np.load(os.path.join(input_dir, my_feat_data_fname))
    return node_features


def gen_node_weights_files(schema_map, output):
    """
    Function to create node weight files for ParMETIS along with the edge files.

    Parameters:
    -----------
    schema_map : json dictionary
        dictionary created by reading the metadata.json file for the input dataset
    output : string
        location of storing the node-weights and edge files for ParMETIS

    Returns:
    --------
    list : 
        list of filenames for nodes of the input graph
    list : 
        list o ffilenames for edges of the input graph
    """
    rank, world_size = get_proc_info()
    ntypes_ntypeid_map, ntypes, ntid_ntype_map = get_node_types(schema_map)
    type_nid_dict, ntype_gnid_offset = get_idranges(schema_map[constants.STR_NODE_TYPE],
                                        schema_map[constants.STR_NUM_NODES_PER_CHUNK])

    node_files = []
    outdir = Path(params.output)
    os.makedirs(outdir, exist_ok=True)

    for ntype_id, ntype_name in ntid_ntype_map.items():
        type_start, type_end = type_nid_dict[ntype_name][rank][0], type_nid_dict[ntype_name][rank][1]
        count = type_end - type_start
        sz = (count,)

        cols = []
        col_names = []

        cols.append(pyarrow.array(np.ones(sz, dtype=np.int64) * np.int64(ntype_id)))
        col_names.append("ntype")

        for i in range(len(ntypes)):
            if (i == ntype_id):
                cols.append(pyarrow.array(np.ones(sz, dtype=np.int64)))
            else:
                cols.append(pyarrow.array(np.zeros(sz, dtype=np.int64)))
            col_names.append("w{}".format(i))


        #
        #Add train/test/validation masks if present. node-degree will be added when this file
        #is read by ParMETIS to mimic the exisiting single process pipeline present in dgl
        #
        node_feats = read_node_features(schema_map,ntype_name, set(["train", "test", "valid"]))
        for k, v in node_feats.items():
            assert sz == v.shape
            cols.append(pyarrow.array(v))
            col_names.append(k)
            #print('Adding data for the col: ', k)

        #
        #type_nid should be the very last column in the node weights files.
        #
        cols.append(pyarrow.array(np.arange(count, dtype=np.int64) + np.int64(type_start)))
        col_names.append("type_nid")

        out_file = os.path.join(outdir, 'node_weights_{}_{}.txt'.format(ntype_name, rank))
        options = csv.WriteOptions(include_header=False, delimiter=" ")
        options.delimiter = " "

        csv.write_csv(pyarrow.Table.from_arrays(cols, names=col_names), out_file, options)
        node_files.append((ntype_gnid_offset[ntype_name][0,0] + type_start, \
                                ntype_gnid_offset[ntype_name][0,0] + type_end, out_file))

    '''
    edge_files = []
    edge_data = schema_map[constants.STR_EDGES]
    for etype_name, etype_info in edge_data.items():
        files_data = etype_info["data"]
        for idx, x in enumerate(files_data):
            if (idx == rank):
                edge_files.append(x)
    '''

    return node_files


def gen_parmetis_input_args(params, schema_map):
    """
    Function to create two input arguments which will be passed to the parmetis.
    first argument is a text file which has a list of node-weights files, and
    second argument is a text file which has a list of edge files. 
    ParMETIS uses these two files to read/load the graph and partition the graph

    Parameters:
    -----------
    params : argparser instance
        instance of ArgParser class, which has all the input arguments passed to 
        run this program
    schema_map : json dictionary
        dictionary object created after reading the graph metadata.json file
    """

    num_nodes_per_chunk = schema_map[constants.STR_NUM_NODES_PER_CHUNK]
    num_parts = len(num_nodes_per_chunk[0])
    ntypes_ntypeid_map, ntypes, ntid_ntype_map = get_node_types(schema_map)
    type_nid_dict, ntype_gnid_offset = get_idranges(schema_map[constants.STR_NODE_TYPE],
                                        schema_map[constants.STR_NUM_NODES_PER_CHUNK])

    node_files = []
    outdir = Path(params.output)
    os.makedirs(outdir, exist_ok=True)
    for ntype_id, ntype_name in ntid_ntype_map.items():
        global_nid_offset = ntype_gnid_offset[ntype_name][0,0]
        for r in range(num_parts):
            type_start, type_end = type_nid_dict[ntype_name][r][0], type_nid_dict[ntype_name][r][1]
            out_file = os.path.join(outdir, 'node_weights_{}_{}.txt'.format(ntype_name, r))
            node_files.append((out_file, global_nid_offset + type_start, global_nid_offset + type_end))

    nfile = open(os.path.join(params.output, 'parmetis_nfiles.txt'), "w")
    for f in node_files:
        #filename global_node_id_start global_node_id_end(exclusive)
        nfile.write('{} {} {}\n'.format(f[0], f[1], f[2]))
    nfile.close()

    #Regenerate edge files here...▒
    edge_data = schema_map[constants.STR_EDGES]
    edge_files = []
    for etype_name, etype_info in edge_data.items():
        edge_info = etype_info[constants.STR_DATA]
        for r in range(num_parts):
            out_file = edge_info[r].split("/")[-1]
            out_file = os.path.join(outdir, 'edges_{}'.format(out_file))
            edge_files.append(out_file)

    efile = open(os.path.join(params.output, 'parmetis_efiles.txt'), "w")
    for f in edge_files:
        efile.write('{}\n'.format(f))
    efile.close()


def run_wrapper(params):
    print("Starting to generate ParMETIS files...")

    rank, world_size = get_proc_info()
    schema = read_json(params.schema)
    num_nodes_per_chunk = schema[constants.STR_NUM_NODES_PER_CHUNK]
    num_parts = len(num_nodes_per_chunk[0])
    n = gen_node_weights_files(schema, params.output)
    print('Done with node weights....')

    e = gen_edge_files(schema, params.output)
    print('Done with edge weights...')

    if rank == 0:
        gen_parmetis_input_args(params, schema)
    print('Done generating files for ParMETIS run ..')

    '''
    nfiles = []
    efiles = []
    for i in range(num_parts):
        n = gen_node_weights_files(i, schema, params.output)
        e = gen_edge_files(i, schema, params.output)
        nfiles.append(n)
        efiles.append(e)

    nodes = []
    outer_len = len(nfiles)
    inner_len = len(nfiles[0])
    for x in range(inner_len):
        for y in range(outer_len):
            nodes.append(nfiles[y][x])

    edges = []
    outer_len = len(efiles)
    inner_len = len(efiles[0])
    for x in range(inner_len):
        for y in range(outer_len):
            edges.append(efiles[y][x])

    gen_parmetis_input_args(params, nodes, edges)
    '''

if __name__ == "__main__":
    """
    Main function used to generate temporary files needed for ParMETIS execution.
    This function generates node-weight files and edges files which are consumed by ParMETIS.

    Example usage:
    --------------
    mpirun -np 4 python3 parmetis_wrapper --schema <file> --output <target-output-dir>
    """
    parser = argparse.ArgumentParser(description='Generate ParMETIS files for input dataset')
    parser.add_argument('--schema', required=True, type=str,
                     help='The schema of the input graph')
    parser.add_argument('--output', required=True, type=str,
                    help='The output directory for the node weights files and auxiliary files for ParMETIS.')
    params = parser.parse_args()

    #Invoke the function to generate files for parmetis
    run_wrapper(params)
