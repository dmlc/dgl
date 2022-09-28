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


def create_chunked_dataset(root_dir, num_chunks, include_masks=False):
    """
    This function creates a sample dataset, based on MAG240 dataset.

    Parameters:
    -----------
    root_dir : string
        directory in which all the files for the chunked dataset will be stored.
    """
    # Step0: prepare chunked graph data format.
    # A synthetic mini MAG240.
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

    # Structure.
    data_dict = {
        ('paper', 'cites', 'paper'): rand_edges(
            num_papers, num_papers, num_cite_edges
        ),
        ('author', 'writes', 'paper'): rand_edges(
            num_authors, num_papers, num_write_edges
        ),
        ('author', 'affiliated_with', 'institution'): rand_edges(
            num_authors, num_institutions, num_affiliate_edges
        ),
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

    # masks.
    if include_masks:
        paper_train_mask = np.random.randint(0, 2, num_papers)
        paper_test_mask = np.random.randint(0, 2, num_papers)
        paper_val_mask = np.random.randint(0, 2, num_papers)

        author_train_mask = np.random.randint(0, 2, num_authors)
        author_test_mask = np.random.randint(0, 2, num_authors)
        author_val_mask = np.random.randint(0, 2, num_authors)

        inst_train_mask = np.random.randint(0, 2, num_institutions)
        inst_test_mask = np.random.randint(0, 2, num_institutions)
        inst_val_mask = np.random.randint(0, 2, num_institutions)

    # Edge features.
    cite_count = np.random.choice(10, num_cite_edges)
    write_year = np.random.choice(2022, num_write_edges)

    # Save features.
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

    node_data = None
    if include_masks:
        for sub_d in ['author', 'institution']:
            os.makedirs(os.path.join(input_dir, sub_d))
        paper_train_mask_path = os.path.join(input_dir, 'paper/train_mask.npy')
        with open(paper_train_mask_path, 'wb') as f:
            np.save(f, paper_train_mask)

        paper_test_mask_path = os.path.join(input_dir, 'paper/test_mask.npy')
        with open(paper_test_mask_path, 'wb') as f:
            np.save(f, paper_test_mask)

        paper_val_mask_path = os.path.join(input_dir, 'paper/val_mask.npy')
        with open(paper_val_mask_path, 'wb') as f:
            np.save(f, paper_val_mask)

        author_train_mask_path = os.path.join(input_dir, 'author/train_mask.npy')
        with open(author_train_mask_path, 'wb') as f:
            np.save(f, author_train_mask)

        author_test_mask_path = os.path.join(input_dir, 'author/test_mask.npy')
        with open(author_test_mask_path, 'wb') as f:
            np.save(f, author_test_mask)

        author_val_mask_path = os.path.join(input_dir, 'author/val_mask.npy')
        with open(author_val_mask_path, 'wb') as f:
            np.save(f, author_val_mask)

        inst_train_mask_path = os.path.join(input_dir, 'institution/train_mask.npy')
        with open(inst_train_mask_path, 'wb') as f:
            np.save(f, inst_train_mask)

        inst_test_mask_path = os.path.join(input_dir, 'institution/test_mask.npy')
        with open(inst_test_mask_path, 'wb') as f:
            np.save(f, inst_test_mask)

        inst_val_mask_path = os.path.join(input_dir, 'institution/val_mask.npy')
        with open(inst_val_mask_path, 'wb') as f:
            np.save(f, inst_val_mask)

        node_data = {
                    'paper':
                    {
                        'feat': paper_feat_path,
                        'train_mask': paper_train_mask_path,
                        'test_mask': paper_test_mask_path,
                        'val_mask': paper_val_mask_path,
                        'label': paper_label_path,
                        'year': paper_year_path
                    },
                    'author':
                    {
                        'train_mask': author_train_mask_path,
                        'test_mask': author_test_mask_path,
                        'val_mask': author_val_mask_path

                    },
                    'institution':
                    {
                        'train_mask': inst_train_mask_path,
                        'test_mask': inst_test_mask_path,
                        'val_mask': inst_val_mask_path
                    }
                }
    else:
        node_data = {
                    'paper': {
                        'feat': paper_feat_path,
                        'label': paper_label_path,
                        'year': paper_year_path,
                    }
                }

    output_dir = os.path.join(root_dir, 'chunked-data')
    chunk_graph(
            g,
            'mag240m',
            node_data,
            {
                'cites': {'count': cite_count_path},
                'writes': {'year': write_year_path},
                'rev_writes': {'year': write_year_path},
            },
            num_chunks=num_chunks,
            output_path=output_dir,
    )
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
            "author:affiliated_with:institution": np.array([0, 200]).reshape(
                1, 2
            ),
            "author:writes:paper": np.array([200, 1200]).reshape(1, 2),
            "paper:cites:paper": np.array([1200, 3200]).reshape(1, 2),
            "paper:rev_writes:author": np.array([3200, 4200]).reshape(1, 2),
    }
    id_map = dgl.distributed.id_map.IdMap(edge_dict)

    # check edge_index
    output_edge_index_dir = os.path.join(output_dir, 'edge_index')
    for utype, etype, vtype in data_dict.keys():
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
    edge_data_gold = {}
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
                edge_data_gold[etype + '/' + feat] = np.concatenate(
                    features
                )
            else:
                edge_data_gold[etype + '/' + feat] = np.row_stack(features)

    return ['author', 'institution', 'paper'], \
           ['affiliated_with', 'writes', 'cites', 'rev_writes'], \
           edge_data_gold
