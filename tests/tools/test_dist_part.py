import json
import numpy as np
import os
import tempfile
import torch
import pytest, unittest
import dgl
from dgl.data.utils import load_tensors, load_graphs
from chunk_graph import chunk_graph

@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_parts", [1, 2, 3, 4, 8])
def test_part_pipeline(num_chunks, num_parts):
    if num_chunks < num_parts:
        # num_parts should less/equal than num_chunks
        return

    # Step0: prepare chunked graph data format

    # A synthetic mini MAG240
    num_institutions = 1200
    num_authors = 1200
    num_papers = 1200

    def rand_edges(num_src, num_dst, num_edges):
        eids = np.random.choice(num_src * num_dst, num_edges, replace=False)
        src = torch.from_numpy(eids // num_dst)
        dst = torch.from_numpy(eids % num_dst)

        return src, dst

    num_cite_edges = 24 * 1000
    num_write_edges = 12 * 1000
    num_affiliate_edges = 2400

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
    paper_orig_ids = np.arange(0, num_papers)

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

        paper_orig_ids_path = os.path.join(input_dir, 'paper/orig_ids.npy')
        with open(paper_orig_ids_path, 'wb') as f:
            np.save(f, paper_orig_ids)

        cite_count_path = os.path.join(input_dir, 'cites/count.npy')
        with open(cite_count_path, 'wb') as f:
            np.save(f, cite_count)

        write_year_path = os.path.join(input_dir, 'writes/year.npy')
        with open(write_year_path, 'wb') as f:
            np.save(f, write_year)

        output_dir = os.path.join(root_dir, 'chunked-data')
        chunk_graph(
            g,
            'mag240m',
            {'paper':
                {
                'feat': paper_feat_path,
                'label': paper_label_path,
                'year': paper_year_path,
                'orig_ids': paper_orig_ids_path
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
        for feat in ['feat', 'label', 'year', 'orig_ids']:
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

        # Step1: graph partition
        in_dir = os.path.join(root_dir, 'chunked-data')
        output_dir = os.path.join(root_dir, 'parted_data')
        os.system('python3 tools/partition_algo/random_partition.py '\
                  '--in_dir {} --out_dir {} --num_partitions {}'.format(
                    in_dir, output_dir, num_parts))
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
        cmd += ' --ssh-port 22'
        cmd += ' --process-group-timeout 60'
        cmd += ' --save-orig-nids'
        cmd += ' --save-orig-eids'
        os.system(cmd)

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
        assert meta_data['num_edges'] == g.num_edges()
        assert meta_data['num_nodes'] == g.num_nodes()
        assert meta_data['num_parts'] == num_parts

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
            part_g = g_list[0]
            assert isinstance(part_g, dgl.DGLGraph)

            # node_feat.dgl
            fname = os.path.join(sub_dir, 'node_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            all_tensors = ['paper/feat', 'paper/label', 'paper/year', 'paper/orig_ids']
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)
            ndata_paper_orig_ids = tensor_dict['paper/orig_ids']

            # edge_feat.dgl
            fname = os.path.join(sub_dir, 'edge_feat.dgl')
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)

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

