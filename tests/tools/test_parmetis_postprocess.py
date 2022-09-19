import argparse
import dgl
import json
import numpy as np
import os
import sys
import tempfile
import torch
import logging
import platform

from dgl.data.utils import load_tensors, load_graphs
from chunk_graph import chunk_graph

def test_parmetis_preprocessing():
    # Step0: prepare chunked graph data format.

    # Create a synthetic mini graph (similar to MAG240 dataset).
    num_institutions = 20
    num_authors = 100
    num_papers = 600
    num_cite_edges = 2000
    num_write_edges = 1000
    num_affiliate_edges = 200
    num_nodes = num_institutions + num_authors + num_papers

    def rand_edges(num_src, num_dst, num_edges):
        eids = np.random.choice(num_src * num_dst, num_edges, replace=False)
        src = torch.from_numpy(eids // num_dst)
        dst = torch.from_numpy(eids % num_dst)
        return src, dst

    # Create the no. of edges and build a dictioinary to store them. 
    data_dict = {
        ('paper', 'cites', 'paper'): rand_edges(num_papers, num_papers, num_cite_edges),
        ('author', 'writes', 'paper'): rand_edges(num_authors, num_papers, num_write_edges),
        ('author', 'affiliated_with', 'institution'): rand_edges(num_authors, num_institutions, num_affiliate_edges)
    }
    src, dst = data_dict[('author', 'writes', 'paper')]
    data_dict[('paper', 'rev_writes', 'author')] = (dst, src)
    g = dgl.heterograph(data_dict)

    # Save node features in appropriate files.
    with tempfile.TemporaryDirectory() as root_dir:
        print('root_dir:', root_dir)
        input_dir = os.path.join(root_dir, 'data_test')
        os.makedirs(input_dir)
        for sub_d in ['paper', 'cites', 'writes', 'author', 'institution']:
            os.makedirs(os.path.join(input_dir, sub_d))

        output_dir = os.path.join(root_dir, 'chunked-data')
        num_chunks = 2
        chunk_graph(
            g,
            'mag240m',
            {
                'paper': {}, 
                'author': {},
                'institution': {}
            },
            {
                'cites': { },
                'writes': { },
                'rev_writes': { }
            },
            num_chunks=num_chunks,
            output_path=output_dir)

        # Check metadata.json.
        json_file = os.path.join(output_dir, 'metadata.json')
        assert os.path.isfile(json_file)
        with open(json_file, 'rb') as f:
            meta_data = json.load(f)
        assert meta_data['graph_name'] == 'mag240m'
        assert len(meta_data['num_nodes_per_chunk'][0]) == num_chunks

        # Generate random parmetis partition ids for the nodes in the graph.
        # Replace this code with actual ParMETIS executable when it is ready
        parmetis_file = os.path.join(output_dir, 'parmetis_output.txt')
        node_ids = np.arange(num_nodes)
        partition_ids = np.random.randint(0, 2, (num_nodes,))
        parmetis_output = np.column_stack([node_ids, partition_ids])
        
        # Create parmetis output, this is mimicking running actual parmetis.
        with open(parmetis_file, 'w') as f:
            np.savetxt(f, parmetis_output)

        # Check the post processing script here. 
        results_dir = os.path.join(output_dir, 'partitions_dir')
        print(json_file)
        print(results_dir)
        print(parmetis_file)
        os.system(f'python3 tools/distpartitioning/parmetis_postprocess.py '\
                f'--schema {json_file} '\
                f'--parmetis_output {parmetis_file} '\
                f'--output_dir {results_dir}')

        ntype_count = {
                'author':num_authors, 
                'paper':num_papers, 
                'institution':num_institutions
                }
        for ntype_name in ['author', 'paper', 'institution']:
            fname = os.path.join(results_dir, f'{ntype_name}.txt')
            print(fname)
            assert os.path.isfile(fname)
            
            # Load and check the partition ids in this file.
            part_ids = np.loadtxt(fname)
            assert part_ids.shape[0] == ntype_count[ntype_name]
            assert np.min(part_ids) == 0
            assert np.max(part_ids) == 1

if __name__ == '__main__':
    #Configure logging.
    logging.basicConfig(level='INFO', format=f"[{platform.node()} \
        %(levelname)s %(asctime)s PID:%(process)d] %(message)s")
    test_parmetis_preprocessing()
