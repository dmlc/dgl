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

def test_part_pipeline():
    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks = 2
        all_ntypes, all_etypes = create_chunked_dataset(root_dir, num_chunks)
        
        # Step1: graph partition
        in_dir = os.path.join(root_dir, 'chunked-data')
        output_dir = os.path.join(root_dir, '2parts')
        os.system('python3 tools/partition_algo/random_partition.py '\
                  '--in_dir {} --out_dir {} --num_partitions {}'.format(
                    in_dir, output_dir, num_chunks))
        for ntype in ['author', 'institution', 'paper']:
            fname = os.path.join(output_dir, '{}.txt'.format(ntype))
            with open(fname, 'r') as f:
                header = f.readline().rstrip()
                assert isinstance(int(header), int)

        # Step2: data dispatch
        partition_dir = os.path.join(root_dir, '2parts')
        out_dir = os.path.join(root_dir, 'partitioned')
        ip_config = os.path.join(root_dir, 'ip_config.txt')
        with open(ip_config, 'w') as f:
            f.write('127.0.0.1\n')
            f.write('127.0.0.2\n')

        cmd = 'python3 tools/dispatch_data.py'
        cmd += f' --in-dir {in_dir}'
        cmd += f' --partitions-dir {partition_dir}'
        cmd += f' --out-dir {out_dir}'
        cmd += f' --ip-config {ip_config}'
        cmd += ' --process-group-timeout 60'
        os.system(cmd)

        # check metadata.json
        meta_fname = os.path.join(out_dir, 'metadata.json')
        with open(meta_fname, 'rb') as f:
            meta_data = json.load(f)

        #all_etypes = ['affiliated_with', 'writes', 'cites', 'rev_writes']
        for etype in all_etypes:
            assert len(meta_data['edge_map'][etype]) == num_chunks
        assert meta_data['etypes'].keys() == set(all_etypes)
        assert meta_data['graph_name'] == 'mag240m'

        #all_ntypes = ['author', 'institution', 'paper']
        for ntype in all_ntypes:
            assert len(meta_data['node_map'][ntype]) == num_chunks
        assert meta_data['ntypes'].keys() == set(all_ntypes)
        assert meta_data['num_edges'] == 4200
        assert meta_data['num_nodes'] == 720
        assert meta_data['num_parts'] == num_chunks

        for i in range(num_chunks):
            sub_dir = 'part-' + str(i)
            assert meta_data[sub_dir]['node_feats'] == 'part{}/node_feat.dgl'.format(i)
            assert meta_data[sub_dir]['edge_feats'] == 'part{}/edge_feat.dgl'.format(i)
            assert meta_data[sub_dir]['part_graph'] == 'part{}/graph.dgl'.format(i)

            # check data
            sub_dir = os.path.join(out_dir, 'part' + str(i))

            # graph.dgl
            fname = os.path.join(sub_dir, 'graph.dgl')
            assert os.path.isfile(fname)
            g_list, data_dict = load_graphs(fname)
            g = g_list[0]
            assert isinstance(g, dgl.DGLGraph)

            # node_feat.dgl
            fname = os.path.join(sub_dir, 'node_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            all_tensors = ['paper/feat', 'paper/label', 'paper/year']
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)

            # edge_feat.dgl
            fname = os.path.join(sub_dir, 'edge_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)

if __name__ == '__main__':
    test_part_pipeline()
