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

def test_parmetis_wrapper():
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
    with tempfile.TemporaryDirectory() as root_dir:
        print('root_dir', root_dir)
        input_dir = os.path.join(root_dir, 'data_test')
        os.makedirs(input_dir)
        for sub_d in ['paper', 'cites', 'writes']:
            os.makedirs(os.path.join(input_dir, sub_d))

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

        output_dir = os.path.join(root_dir, 'chunked-data')
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

        # Trigger ParMETIS wrapper here. 
        schema_path = os.path.join(root_dir, 'chunked-data/metadata.json')
        results_dir = os.path.join(root_dir, 'parmetis-data')
        os.system('mpirun -np 2 python3 tools/distpartitioning/parmetis_wrapper.py '\
                  '--schema {} --output {}'.format(schema_path, results_dir))

        # Now add all the tests and check whether the test has passed or failed
        # Test - 1
        # Read parmetis_nfiles and ensure all files are present
        parmetis_data_dir = os.path.join(root_dir, 'parmetis-data')
        assert os.path.isdir(parmetis_data_dir)
        parmetis_nodes_file = os.path.join(parmetis_data_dir, 'parmetis_nfiles.txt')
        assert os.path.isfile(parmetis_nodes_file)

        with open(parmetis_nodes_file, 'r') as f:
            lines = f.readlines()
            total_node_count = 0
            for line in lines:
                tokens = line.split(" ")
                print(tokens)
                assert len(tokens) == 3
                print(tokens[0])
                assert os.path.isfile(tokens[0])
                assert int(tokens[1]) == total_node_count

                #check contents of each of the nodes files here
                with open(tokens[0], 'r') as nf:
                    node_lines = nf.readlines()
                    node_count = len(node_lines)
                    total_node_count += node_count
                assert int(tokens[2]) == total_node_count
        
        # Count the total no. of nodes
        true_node_count = 0
        num_nodes_per_chunk = meta_data['num_nodes_per_chunk']
        for i in range(len(num_nodes_per_chunk)):
            node_per_part = num_nodes_per_chunk[i]
            for j in range(len(node_per_part)):
                true_node_count += node_per_part[j]
        assert total_node_count == true_node_count

        # Read parmetis_efiles and ensure all files are present
        parmetis_edges_file = os.path.join(parmetis_data_dir, 'parmetis_efiles.txt')
        assert os.path.isfile(parmetis_edges_file)

        with open(parmetis_edges_file, 'r') as f:
            lines = f.readlines()
            total_edge_count = 0
            for line in lines:
                ff = line.strip()
                assert os.path.isfile(ff)

                with open(ff, 'r') as ef:
                    edge_count = len(ef.readlines())
                    total_edge_count += edge_count

        # Count the total no. of edges
        true_edge_count = 0
        num_edges_per_chunk = meta_data['num_edges_per_chunk']
        for i in range(len(num_edges_per_chunk)):
            edges_per_part = num_edges_per_chunk[i]
            for j in range(len(edges_per_part)):
                true_edge_count += edges_per_part[j]
        assert true_edge_count == total_edge_count

if __name__ == '__main__':
    test_parmetis_wrapper()
