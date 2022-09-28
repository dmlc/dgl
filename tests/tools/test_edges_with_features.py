import argparse
import json
import logging
import os
import platform
import sys
import tempfile

import dgl
import numpy as np
import torch
from chunk_graph import chunk_graph
from dgl.data.utils import load_graphs, load_tensors

from create_chunked_dataset import create_chunked_dataset

def test_edges_with_features():
    """
    This function is a unit test for testing edges with features data.
    This will be triggered by the CI test framework or can be launched
    individually as follows:

    python3 -m pytest tests/tools/test_edges_with_features.py

    Please note that, this is based on the test_dist_part.py unit test.
    This file also uses the same dataset, in chunked format. But, while validating
    the results in this unit test additional checks are made in each of the graph
    partitions to validate the feature data stored in each partition to make sure
    that it matches with the data stored in the original graph.
    """

    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks=2
        all_ntypes, all_etypes, edge_data_gold = create_chunked_dataset(root_dir, num_chunks)

        # Step1: graph partition
        in_dir = os.path.join(root_dir, 'chunked-data')
        output_dir = os.path.join(root_dir, '2parts')
        os.system(
            'python3 tools/partition_algo/random_partition.py '
            '--in_dir {} --out_dir {} --num_partitions {}'.format(
                in_dir, output_dir, num_chunks
            )
        )
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

        os.system(
            'python3 tools/dispatch_data.py '
            '--in-dir {} --partitions-dir {} --out-dir {} --ip-config {}'.format(
                in_dir, partition_dir, out_dir, ip_config
            )
        )
        print('Graph partitioning pipeline complete...')

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

        # Create Id Map here.
        edge_dict = {
            "author:affiliated_with:institution": np.array([0, 200]).reshape(
                1, 2
            ),
            "author:writes:paper": np.array([200, 1200]).reshape(1, 2),
            "paper:cites:paper": np.array([1200, 3200]).reshape(1, 2),
            "paper:rev_writes:author": np.array([3200, 4200]).reshape(1, 2),
        }
        id_map = dgl.distributed.id_map.IdMap(edge_dict)
        etype_id, type_eid = id_map(np.arange(4200))

        for i in range(num_chunks):
            sub_dir = 'part-' + str(i)
            sub_dir = os.path.join(out_dir, 'part' + str(i))

            # graph.dgl
            fname = os.path.join(sub_dir, 'graph.dgl')
            assert os.path.isfile(fname)
            g_list, data_dict = load_graphs(fname)
            g = g_list[0]

            # edge_feat.dgl
            fname = os.path.join(sub_dir, 'edge_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)

            # get orig_eids
            orig_type_eids = g.edata['orig_id'].numpy()
            orig_etype_ids = g.edata[dgl.ETYPE].numpy()
            print(f'[Rank: {i}] orig_etype_ids: {np.bincount(orig_etype_ids)}')

        for i in range(num_chunks):
            sub_dir = 'part-' + str(i)
            assert meta_data[sub_dir][
                'node_feats'
            ] == 'part{}/node_feat.dgl'.format(i)
            assert meta_data[sub_dir][
                'edge_feats'
            ] == 'part{}/edge_feat.dgl'.format(i)
            assert meta_data[sub_dir][
                'part_graph'
            ] == 'part{}/graph.dgl'.format(i)

            # check data
            sub_dir = os.path.join(out_dir, 'part' + str(i))

            # Read graph.dgl
            fname = os.path.join(sub_dir, 'graph.dgl')
            assert os.path.isfile(fname)
            g_list, data_dict = load_graphs(fname)
            g = g_list[0]
            assert isinstance(g, dgl.DGLGraph)

            # Read node_feat.dgl
            fname = os.path.join(sub_dir, 'node_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            all_tensors = ['paper/feat', 'paper/label', 'paper/year']
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)

            # Read edge_feat.dgl
            fname = os.path.join(sub_dir, 'edge_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            all_tensors = [
                'paper:cites:paper/count',
                'author:writes:paper/year',
                'paper:rev_writes:author/year',
            ]
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)

            # Get orig_eids
            orig_type_eids = g.edata['orig_id'].numpy()
            orig_etype_ids = g.edata[dgl.ETYPE].numpy()

            # Compare the data stored as edge features in this partition with the data
            # from the original graph.
            etype_names = list(edge_dict.keys())
            for idx, etype_name in enumerate(etype_names):
                part_data = None
                key = None
                if etype_name + '/count' in tensor_dict:
                    key = etype_name + '/count'
                    part_data = tensor_dict[etype_name + '/count'].numpy()
                if etype_name + '/year' in tensor_dict:
                    key = etype_name + '/year'
                    part_data = tensor_dict[etype_name + '/year'].numpy()

                if part_data is None:
                    continue

                gold_type_ids = orig_type_eids[orig_etype_ids == idx]
                gold_data = edge_data_gold[key][gold_type_ids]
                assert np.all(gold_data == part_data)


if __name__ == '__main__':
    test_edges_with_features()
