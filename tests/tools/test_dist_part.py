import json
import os
import tempfile
import unittest

import dgl
import numpy as np
import pytest
import torch
from chunk_graph import chunk_graph
from dgl.data.utils import load_graphs, load_tensors

from create_chunked_dataset import create_chunked_dataset

@pytest.mark.parametrize("num_chunks", [1, 8])
def test_chunk_graph(num_chunks):

    with tempfile.TemporaryDirectory() as root_dir:

        g = create_chunked_dataset(root_dir, num_chunks, include_edge_data=True)

        num_cite_edges = g.number_of_edges('cites')
        num_write_edges = g.number_of_edges('writes')
        num_affiliate_edges = g.number_of_edges('affiliated_with')

        num_institutions = g.number_of_nodes('institution')
        num_authors = g.number_of_nodes('author')
        num_papers = g.number_of_nodes('paper')

        # check metadata.json
        output_dir = os.path.join(root_dir, 'chunked-data')
        json_file = os.path.join(output_dir, 'metadata.json')
        assert os.path.isfile(json_file)
        with open(json_file, 'rb') as f:
            meta_data = json.load(f)
        assert meta_data['graph_name'] == 'mag240m'
        assert len(meta_data['num_nodes_per_chunk'][0]) == num_chunks

        # check edge_index
        output_edge_index_dir = os.path.join(output_dir, 'edge_index')
        for utype, etype, vtype in g.canonical_etypes:
            fname = ':'.join([utype, etype, vtype])
            for i in range(num_chunks):
                chunk_f_name = os.path.join(
                    output_edge_index_dir, fname + str(i) + '.txt'
                )
                assert os.path.isfile(chunk_f_name)
                with open(chunk_f_name, 'r') as f:
                    header = f.readline()
                    num1, num2 = header.rstrip().split(' ')
                    assert isinstance(int(num1), int)
                    assert isinstance(int(num2), int)

        # check node_data
        output_node_data_dir = os.path.join(output_dir, 'node_data', 'paper')
        for feat in ['feat', 'label', 'year']:
            for i in range(num_chunks):
                chunk_f_name = '{}-{}.npy'.format(feat, i)
                chunk_f_name = os.path.join(output_node_data_dir, chunk_f_name)
                assert os.path.isfile(chunk_f_name)
                feat_array = np.load(chunk_f_name)
                assert feat_array.shape[0] == num_papers // num_chunks

        # check edge_data
        num_edges = {
            'paper:cites:paper': num_cite_edges,
            'author:writes:paper': num_write_edges,
            'paper:rev_writes:author': num_write_edges,
        }
        output_edge_data_dir = os.path.join(output_dir, 'edge_data')
        for etype, feat in [
            ['paper:cites:paper', 'count'],
            ['author:writes:paper', 'year'],
            ['paper:rev_writes:author', 'year'],
        ]:
            output_edge_sub_dir = os.path.join(output_edge_data_dir, etype)
            for i in range(num_chunks):
                chunk_f_name = '{}-{}.npy'.format(feat, i)
                chunk_f_name = os.path.join(output_edge_sub_dir, chunk_f_name)
                assert os.path.isfile(chunk_f_name)
                feat_array = np.load(chunk_f_name)
            assert feat_array.shape[0] == num_edges[etype] // num_chunks


@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_parts", [1, 2, 3, 4, 8])
def test_part_pipeline(num_chunks, num_parts):
    if num_chunks < num_parts:
        # num_parts should less/equal than num_chunks
        return

    include_edge_data = num_chunks == num_parts

    with tempfile.TemporaryDirectory() as root_dir:

        g = create_chunked_dataset(
            root_dir, num_chunks, include_edge_data=include_edge_data
        )

        all_ntypes = g.ntypes
        all_etypes = g.etypes

        num_cite_edges = g.number_of_edges('cites')
        num_write_edges = g.number_of_edges('writes')
        num_affiliate_edges = g.number_of_edges('affiliated_with')

        num_institutions = g.number_of_nodes('institution')
        num_authors = g.number_of_nodes('author')
        num_papers = g.number_of_nodes('paper')

        # Step1: graph partition
        in_dir = os.path.join(root_dir, 'chunked-data')
        output_dir = os.path.join(root_dir, 'parted_data')
        os.system(
            'python3 tools/partition_algo/random_partition.py '
            '--in_dir {} --out_dir {} --num_partitions {}'.format(
                in_dir, output_dir, num_parts
            )
        )
        for ntype in ['author', 'institution', 'paper']:
            fname = os.path.join(output_dir, '{}.txt'.format(ntype))
            with open(fname, 'r') as f:
                header = f.readline().rstrip()
                assert isinstance(int(header), int)

        # Step2: data dispatch
        partition_dir = os.path.join(root_dir, 'parted_data')
        out_dir = os.path.join(root_dir, 'partitioned')
        ip_config = os.path.join(root_dir, 'ip_config.txt')
        with open(ip_config, 'w') as f:
            for i in range(num_parts):
                f.write(f'127.0.0.{i + 1}\n')

        cmd = 'python3 tools/dispatch_data.py'
        cmd += f' --in-dir {in_dir}'
        cmd += f' --partitions-dir {partition_dir}'
        cmd += f' --out-dir {out_dir}'
        cmd += f' --ip-config {ip_config}'
        cmd += ' --process-group-timeout 60'
        cmd += ' --save-orig-nids'
        cmd += ' --save-orig-eids'
        os.system(cmd)

        # check metadata.json
        meta_fname = os.path.join(out_dir, 'metadata.json')
        with open(meta_fname, 'rb') as f:
            meta_data = json.load(f)

        for etype in all_etypes:
            assert len(meta_data['edge_map'][etype]) == num_parts
        assert meta_data['etypes'].keys() == set(all_etypes)
        assert meta_data['graph_name'] == 'mag240m'

        for ntype in all_ntypes:
            assert len(meta_data['node_map'][ntype]) == num_parts
        assert meta_data['ntypes'].keys() == set(all_ntypes)
        assert meta_data['num_edges'] == g.num_edges()
        assert meta_data['num_nodes'] == g.num_nodes()
        assert meta_data['num_parts'] == num_parts

        edge_dict = {}
        edge_data_gold = {}

        if include_edge_data:
            # Create Id Map here.
            num_edges = 0
            for utype, etype, vtype in g.canonical_etypes:
                fname = ':'.join([utype, etype, vtype])
                edge_dict[fname] = np.array(
                    [num_edges, num_edges + g.number_of_edges(etype)]
                ).reshape(1, 2)
                num_edges += g.number_of_edges(etype)

            assert num_edges == g.number_of_edges()
            id_map = dgl.distributed.id_map.IdMap(edge_dict)
            orig_etype_id, orig_type_eid = id_map(np.arange(num_edges))

            # check edge_data
            num_edges = {
                'paper:cites:paper': num_cite_edges,
                'author:writes:paper': num_write_edges,
                'paper:rev_writes:author': num_write_edges,
            }
            output_dir = os.path.join(root_dir, 'chunked-data')
            output_edge_data_dir = os.path.join(output_dir, 'edge_data')
            for etype, feat in [
                ['paper:cites:paper', 'count'],
                ['author:writes:paper', 'year'],
                ['paper:rev_writes:author', 'year'],
            ]:
                output_edge_sub_dir = os.path.join(output_edge_data_dir, etype)
                features = []
                for i in range(num_chunks):
                    chunk_f_name = '{}-{}.npy'.format(feat, i)
                    chunk_f_name = os.path.join(
                        output_edge_sub_dir, chunk_f_name
                    )
                    assert os.path.isfile(chunk_f_name)
                    feat_array = np.load(chunk_f_name)
                    assert feat_array.shape[0] == num_edges[etype] // num_chunks
                features.append(feat_array)
                edge_data_gold[etype + '/' + feat] = np.concatenate(features)

        for i in range(num_parts):
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

            # graph.dgl
            fname = os.path.join(sub_dir, 'graph.dgl')
            assert os.path.isfile(fname)
            g_list, data_dict = load_graphs(fname)
            part_g = g_list[0]
            assert isinstance(part_g, dgl.DGLGraph)

            # node_feat.dgl
            fname = os.path.join(sub_dir, 'node_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            all_tensors = [
                'paper/feat',
                'paper/label',
                'paper/year',
                'paper/orig_ids',
            ]
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)
            ndata_paper_orig_ids = tensor_dict['paper/orig_ids']

            # orig_nids.dgl
            fname = os.path.join(sub_dir, 'orig_nids.dgl')
            assert os.path.isfile(fname)
            orig_nids = load_tensors(fname)
            assert len(orig_nids.keys()) == 3
            assert torch.equal(ndata_paper_orig_ids, orig_nids['paper'])

            # orig_eids.dgl
            fname = os.path.join(sub_dir, 'orig_eids.dgl')
            assert os.path.isfile(fname)
            orig_eids = load_tensors(fname)
            assert len(orig_eids.keys()) == 4

            if include_edge_data:

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

                # Compare the data stored as edge features in this partition with the data
                # from the original graph.
                for idx, etype in enumerate(all_etypes):
                    if etype != key:
                        continue

                    # key in canonical form
                    tokens = key.split(":")
                    assert len(tokens) == 3

                    gold_type_ids = orig_type_eid[orig_etype_id == idx]
                    gold_data = edge_data_gold[key][gold_type_ids]
                    assert np.all(gold_data == part_data.numpy())
