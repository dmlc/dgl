import argparse
import dgl
import json
import numpy as np
import os
import sys
import tempfile
import torch

from dgl.data.utils import load_tensors, load_graphs
from chunk_graph import chunk_graph
from create_chunked_dataset import create_chunked_dataset

def test_parmetis_preprocessing():
    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks = 2
        create_chunked_dataset(root_dir, num_chunks, include_masks=True)

        # Trigger ParMETIS pre-processing here. 
        env = dict(os.environ)
        dgl_home = env['DGL_HOME']
        if dgl_home[-1] != "/":
            dgl_home += "/"
        print(dgl_home)
        schema_path = os.path.join(root_dir, 'chunked-data/metadata.json')
        results_dir = os.path.join(root_dir, 'parmetis-data')
        os.system(f'mpirun -np 2 python3 {dgl_home}tools/distpartitioning/parmetis_preprocess.py '\
                  f'--schema {schema_path} --output {results_dir}')

        # Now add all the tests and check whether the test has passed or failed.
        # Read parmetis_nfiles and ensure all files are present.
        parmetis_data_dir = os.path.join(root_dir, 'parmetis-data')
        assert os.path.isdir(parmetis_data_dir)
        parmetis_nodes_file = os.path.join(parmetis_data_dir, 'parmetis_nfiles.txt')
        assert os.path.isfile(parmetis_nodes_file)

        # `parmetis_nfiles.txt` should have each line in the following format.
        # <filename> <global_id_start> <global_id_end>
        with open(parmetis_nodes_file, 'r') as nodes_metafile:
            lines = nodes_metafile.readlines()
            total_node_count = 0
            for line in lines:
                tokens = line.split(" ")
                assert len(tokens) == 3
                assert os.path.isfile(tokens[0])
                assert int(tokens[1]) == total_node_count

                #check contents of each of the nodes files here
                with open(tokens[0], 'r') as nodes_file:
                    node_lines = nodes_file.readlines()
                    for line in node_lines:
                        val = line.split(" ")
                        # <ntype_id> <weight_list> <mask_list> <type_node_id>
                        assert len(val) == 8 
                    node_count = len(node_lines)
                    total_node_count += node_count
                assert int(tokens[2]) == total_node_count

        # Meta_data object.
        output_dir = os.path.join(root_dir, 'chunked-data')
        json_file = os.path.join(output_dir, 'metadata.json')
        assert os.path.isfile(json_file)
        with open(json_file, 'rb') as f:
            meta_data = json.load(f) 
        
        # Count the total no. of nodes.
        true_node_count = 0
        num_nodes_per_chunk = meta_data['num_nodes_per_chunk']
        for i in range(len(num_nodes_per_chunk)):
            node_per_part = num_nodes_per_chunk[i]
            for j in range(len(node_per_part)):
                true_node_count += node_per_part[j]
        assert total_node_count == true_node_count

        # Read parmetis_efiles and ensure all files are present.
        # This file contains a list of filenames.
        parmetis_edges_file = os.path.join(parmetis_data_dir, 'parmetis_efiles.txt')
        assert os.path.isfile(parmetis_edges_file)

        with open(parmetis_edges_file, 'r') as edges_metafile:
            lines = edges_metafile.readlines()
            total_edge_count = 0
            for line in lines:
                edges_filename = line.strip()
                assert os.path.isfile(edges_filename)

                with open(edges_filename, 'r') as edges_file:
                    edge_lines = edges_file.readlines()
                    total_edge_count += len(edge_lines)
                    for line in edge_lines:
                        val = line.split(" ")
                        assert len(val) == 2

        # Count the total no. of edges
        true_edge_count = 0
        num_edges_per_chunk = meta_data['num_edges_per_chunk']
        for i in range(len(num_edges_per_chunk)):
            edges_per_part = num_edges_per_chunk[i]
            for j in range(len(edges_per_part)):
                true_edge_count += edges_per_part[j]
        assert true_edge_count == total_edge_count

if __name__ == '__main__':
    test_parmetis_preprocessing()
