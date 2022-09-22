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

def test_edge_feature_support():
    # Step0: prepare chunked graph data format
    print('Starting edge feature tests...')

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
    #cite_count = np.arange(num_cite_edges)#np.random.choice(10, num_cite_edges)
    #write_year = np.arange(num_write_edges)#np.random.choice(2022, num_write_edges)
    cite_count = np.random.choice(10, num_cite_edges)
    write_year = np.random.choice(2022, num_write_edges)

    # Save features
    with tempfile.TemporaryDirectory() as root_dir:
        print('Testing root_dir', root_dir)
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
        print('Done with creating chunked graph')

        # check metadata.json
        json_file = os.path.join(output_dir, 'metadata.json')
        assert os.path.isfile(json_file)
        with open(json_file, 'rb') as f:
            meta_data = json.load(f)
        assert meta_data['graph_name'] == 'mag240m'
        assert len(meta_data['num_nodes_per_chunk'][0]) == num_chunks

        print('Metadata Source file: ', meta_data["edge_type"])

        # Create Id Map here. 
        edge_dict = {
                "author:affiliated_with:institution" : np.array([0, 200]).reshape(1,2), 
                "author:writes:paper" : np.array([200, 1200]).reshape(1, 2), 
                "paper:cites:paper" : np.array([1200, 3200]).reshape(1, 2), 
                "paper:rev_writes:author" : np.array([3200, 4200]).reshape(1, 2)
                }
        id_map = dgl.distributed.id_map.IdMap(edge_dict)


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
        print("Node data looks good !!!")

        # check edge_data
        edge_data_gold = {}
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
            features = []
            for i in range(num_chunks):
                chunk_f_name = '{}-{}.npy'.format(feat, i)
                chunk_f_name = os.path.join(output_edge_sub_dir, chunk_f_name)
                assert os.path.isfile(chunk_f_name)
                feat_array = np.load(chunk_f_name)
                assert feat_array.shape[0] == num_edges[etype] // num_chunks
                features.append(feat_array)
            if len(features) > 0: 
                if len(features[0].shape) == 1:
                    edge_data_gold[etype+'/'+feat] = np.concatenate(features)
                else:
                    edge_data_gold[etype+'/'+feat] = np.row_stack(features)

        # Step1: graph partition
        in_dir = os.path.join(root_dir, 'chunked-data')
        output_dir = os.path.join(root_dir, '2parts')
        os.system('python3 tools-parmetis-tests/partition_algo/random_partition.py '\
                  '--metadata {}/metadata.json --output_path {} --num_partitions {}'.format(
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

        os.system('python3 tools/dispatch_data.py '\
                  '--in-dir {} --partitions-dir {} --out-dir {} --ip-config {}'.format(
                    in_dir, partition_dir, out_dir, ip_config))
        print('Done with the pipeline...')

        # check metadata.json
        meta_fname = os.path.join(out_dir, 'metadata.json')
        with open(meta_fname, 'rb') as f:
            meta_data = json.load(f)

        all_etypes = ['affiliated_with', 'writes', 'cites', 'rev_writes']
        for etype in all_etypes:
            assert len(meta_data['edge_map'][etype]) == num_chunks
        assert meta_data['etypes'].keys() == set(all_etypes)
        assert meta_data['graph_name'] == 'mag240m'

        all_ntypes = ['author', 'institution', 'paper']
        for ntype in all_ntypes:
            assert len(meta_data['node_map'][ntype]) == num_chunks
        assert meta_data['ntypes'].keys() == set(all_ntypes)
        assert meta_data['num_edges'] == 4200
        assert meta_data['num_nodes'] == 720
        assert meta_data['num_parts'] == num_chunks

        etype_id, type_eid = id_map(np.arange(4200))
        print(f'MAP_TEST: {np.bincount(etype_id)}')

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
            print(fname)
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)

            # get orig_eids
            orig_type_eids = g.edata['orig_id'].numpy()
            orig_etype_ids = g.edata[dgl.ETYPE].numpy()
            #etype_id, type_eid = id_map(orig_eids)
            #print(f'[Rank: {i}] orig_eids: {orig_eids.shape}, min: {np.min(orig_eids)} max: {np.max(orig_eids)}')
            #print(f'[Rank: {i}] unique etype_ids: {np.bincount(etype_id)}')
            print(f'[Rank: {i}] orig_etype_ids: {np.bincount(orig_etype_ids)}')


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
            print(fname)
            assert os.path.isfile(fname)
            tensor_dict = load_tensors(fname)
            #print(tensor_dict)
            all_tensors = ['paper:cites:paper/count', 
                           'author:writes:paper/year', 
                           'paper:rev_writes:author/year']
            #print(tensor_dict.keys())
            for k, v in tensor_dict.items():
                print(f'[Rank: {i} k: {k} v: {v.numpy().shape}')
            assert tensor_dict.keys() == set(all_tensors)
            for key in all_tensors:
                assert isinstance(tensor_dict[key], torch.Tensor)

            # get orig_eids
            #orig_eids = g.edata['orig_id'].numpy()
            #etype_id, type_eid = id_map(orig_eids)
            orig_type_eids = g.edata['orig_id'].numpy()
            orig_etype_ids = g.edata[dgl.ETYPE].numpy()
            print(f'[Rank: {i}] orig_type_eids: {orig_type_eids.shape}, min: {np.min(orig_type_eids)} max: {np.max(orig_type_eids)}')
            print(f'[Rank: {i}] unique etype_ids: {np.bincount(orig_etype_ids)}')

            etype_names = list(edge_dict.keys())
            comparison = 0
            for idx, etype_name in enumerate(etype_names): 
                part_data = None
                key = None
                if etype_name+'/count' in tensor_dict: 
                    key = etype_name+'/count' 
                    part_data = tensor_dict[etype_name+'/count'].numpy()
                if etype_name+'/year' in tensor_dict: 
                    key = etype_name+'/year' 
                    part_data = tensor_dict[etype_name+'/year'].numpy()

                if part_data is None: 
                    continue

                comparison += 1
                gold_type_ids = orig_type_eids[ orig_etype_ids == idx ]
                print(f'[Rank: {i}] key: {k}')
                print(f'[Rank: {i}] gold_data: {edge_data_gold[key].shape}, type_ids: {gold_type_ids.shape}')

                gold_data = edge_data_gold[key][gold_type_ids]

                print(f'[Rank: {i} type_ids: {gold_type_ids.shape}')
                print(f'[Rank: {i} data: {gold_data.shape}')
                print(f'[Rank: {i} part_data: {part_data.shape}')
                #gold_data = np.sort(gold_data)
                #part_data = np.sort(part_data)
                print(gold_data[0:10])
                print(part_data[0:10])
                assert np.all(gold_data == part_data)


if __name__ == '__main__':
    #logging.basicConfig(level='INFO', format=f"[{platform.node()} %(levelname)s %(asctime)s PID:%(process)d] %(message)s")
    test_edge_feature_support()
