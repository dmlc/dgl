import argparse
import dgl
import json
import numpy as np
import os
import sys
import torch

from dgl.data.utils import load_tensors, load_graphs

from chunk_graph import chunk_graph

def test_chunk_graph():
    # Step0: prepare chunked graph data format

    # A synthetic mini MAG240
    num_institutions = 20
    num_authors = 100
    num_papers = 600

    def rand_edges(num_src, num_dst, num_edges):
        eids = np.random.choice(num_src * num_dst, num_edges, replace=False)
        src = torch.from_numpy(eids // num_dst)
        dst = torch.from_numpy(eids % num_dst)

        return src, dst

    num_cite_edges = 2000
    num_write_edges = 1000
    num_affiliate_edges = 200

    # Structure
    data_dict = {
        ('paper', 'cites', 'paper'): rand_edges(num_papers, num_papers, num_cite_edges),
        ('author', 'writes', 'paper'): rand_edges(num_authors, num_papers, num_write_edges),
        ('author', 'affiliated_with', 'institution'): rand_edges(num_authors, num_institutions, num_affiliate_edges)
    }
    src, dst = data_dict[('author', 'writes', 'paper')]
    data_dict[('paper', 'rev_writes', 'author')] = (dst, src)
    g = dgl.heterograph(data_dict)

    # paper feat, label, year
    num_paper_feats = 3
    paper_feat = np.random.randn(num_papers, num_paper_feats)
    num_classes = 4
    paper_label = np.random.choice(num_classes, num_papers)
    paper_year = np.random.choice(2022, num_papers)

    # edge features
    cite_count = np.random.choice(10, num_cite_edges)
    write_year = np.random.choice(2022, num_write_edges)

    # Save features
    input_dir = 'data_test'
    os.makedirs(input_dir, exist_ok=True)
    for sub_d in ['paper', 'cites', 'writes']:
        os.makedirs(os.path.join(input_dir, sub_d), exist_ok=True)

    paper_feat_path = os.path.join(input_dir, 'paper/feat.npy')
    with open(paper_feat_path, 'wb') as f:
        np.save(f, paper_feat)

    paper_label_path = os.path.join(input_dir, 'paper/label.npy')
    with open(paper_label_path, 'wb') as f:
        np.save(f, paper_label)

    paper_year_path = os.path.join(input_dir, 'paper/year.npy')
    with open(paper_year_path, 'wb') as f:
        np.save(f, paper_year)

    cite_count_path = os.path.join(input_dir, 'cites/count.npy')
    with open(cite_count_path, 'wb') as f:
        np.save(f, cite_count)

    write_year_path = os.path.join(input_dir, 'writes/year.npy')
    with open(write_year_path, 'wb') as f:
        np.save(f, write_year)

    output_dir = 'chunked-data'
    num_chunks = 2
    chunk_graph(
        g,
        'mag240m',
        {'paper':
            {
            'feat': paper_feat_path,
            'label': paper_label_path,
            'year': paper_year_path
            }
        },
        {
            'cites': {'count': cite_count_path},
            'writes': {'year': write_year_path},
            # you can put the same data file if they indeed share the features.
            'rev_writes': {'year': write_year_path}
        },
        num_chunks=num_chunks,
        output_path=output_dir)

    # check metadata.json
    json_file = os.path.join(output_dir, 'metadata.json')
    assert os.path.isfile(json_file)
    with open(json_file, 'rb') as f:
        meta_data = json.load(f)
    assert meta_data['graph_name'] == 'mag240m'
    assert len(meta_data['num_nodes_per_chunk'][0]) == num_chunks

    # check edge_index
    output_edge_index_dir = os.path.join(output_dir, 'edge_index')
    for utype, etype, vtype in data_dict.keys():
        fname = ':'.join([utype, etype, vtype])
        for i in range(num_chunks):
            chunk_f_name = os.path.join(output_edge_index_dir, fname + str(i) + '.txt')
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
        'paper:rev_writes:author': num_write_edges
    }
    output_edge_data_dir = os.path.join(output_dir, 'edge_data')
    for etype, feat in [
        ['paper:cites:paper', 'count'],
        ['author:writes:paper', 'year'],
        ['paper:rev_writes:author', 'year']
    ]:
        output_edge_sub_dir = os.path.join(output_edge_data_dir, etype)
        for i in range(num_chunks):
            chunk_f_name = '{}-{}.npy'.format(feat, i)
            chunk_f_name = os.path.join(output_edge_sub_dir, chunk_f_name)
            assert os.path.isfile(chunk_f_name)
            feat_array = np.load(chunk_f_name)
            assert feat_array.shape[0] == num_edges[etype] // num_chunks

def test_partition():
    # Step1: graph partition
    output_dir = '2parts'
    os.system('python tools/partition_algorithms/random_partition.py '\
              '--metadata chunked-data/metadata.json --output_path {} --num_partitions 2'.format(output_dir))
    for ntype in ['author', 'institution', 'paper']:
        fname = os.path.join(output_dir, '{}.txt'.format(ntype))
        with open(fname, 'r') as f:
            header = f.readline().rstrip()
            assert isinstance(int(header), int)

def test_dispatch():
    # Step2: data dispatch
    out_dir = 'partitioned'
    num_parts = 2
    with open('ip_config.txt', 'w') as f:
        f.write('127.0.0.1\n')
        f.write('127.0.0.2\n')

    os.system('python tools/dispatch_data.py --in-dir chunked-data '\
              '--partitions-dir 2parts --out-dir {} --ip-config ip_config.txt'.format(out_dir))

    # check metadata.json
    meta_fname = os.path.join(out_dir, 'metadata.json')
    with open(meta_fname, 'rb') as f:
        meta_data = json.load(f)

    all_etypes = ['affiliated_with', 'writes', 'cites', 'rev_writes']
    for etype in all_etypes:
        assert len(meta_data['edge_map'][etype]) == num_parts
    assert meta_data['etypes'].keys() == set(all_etypes)
    assert meta_data['graph_name'] == 'mag240m'

    all_ntypes = ['author', 'institution', 'paper']
    for ntype in all_ntypes:
        assert len(meta_data['node_map'][ntype]) == num_parts
    assert meta_data['ntypes'].keys() == set(all_ntypes)
    assert meta_data['num_edges'] == 4200
    assert meta_data['num_nodes'] == 720
    assert meta_data['num_parts'] == num_parts
    assert meta_data['part_method'] == 'metis'

    for i in range(num_parts):
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
    test_chunk_graph()
    test_partition()
    test_dispatch()
