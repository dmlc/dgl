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



def create_chunked_dataset(
    root_dir, num_chunks, include_masks=False, data_fmt='numpy'
):
    """
    This function creates a sample dataset, based on MAG240 dataset.

    Parameters:
    -----------
    root_dir : string
        directory in which all the files for the chunked dataset will be stored.
    """
    # Step0: prepare chunked graph data format.
    # A synthetic mini MAG240.
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
        ('institution', 'writes', 'paper'): rand_edges(
            num_institutions, num_papers, num_write_edges
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
    paper_orig_ids = np.arange(0, num_papers)
    writes_orig_ids = np.arange(0, num_write_edges)

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
    write2_year = np.random.choice(2022, num_write_edges)

    # Save features.
    input_dir = os.path.join(root_dir, 'data_test')
    os.makedirs(input_dir)
    for sub_d in ['paper', 'cites', 'writes', 'writes2']:
        os.makedirs(os.path.join(input_dir, sub_d))

    paper_feat_path = os.path.join(input_dir, 'paper/feat.npy')
    with open(paper_feat_path, 'wb') as f:
        np.save(f, paper_feat)
    g.nodes['paper'].data['feat'] = torch.from_numpy(paper_feat)

    paper_label_path = os.path.join(input_dir, 'paper/label.npy')
    with open(paper_label_path, 'wb') as f:
        np.save(f, paper_label)
    g.nodes['paper'].data['label'] = torch.from_numpy(paper_label)

    paper_year_path = os.path.join(input_dir, 'paper/year.npy')
    with open(paper_year_path, 'wb') as f:
        np.save(f, paper_year)
    g.nodes['paper'].data['year'] = torch.from_numpy(paper_year)

    paper_orig_ids_path = os.path.join(input_dir, 'paper/orig_ids.npy')
    with open(paper_orig_ids_path, 'wb') as f:
        np.save(f, paper_orig_ids)
    g.nodes['paper'].data['orig_ids'] = torch.from_numpy(paper_orig_ids)

    cite_count_path = os.path.join(input_dir, 'cites/count.npy')
    with open(cite_count_path, 'wb') as f:
        np.save(f, cite_count)
    g.edges['cites'].data['count'] = torch.from_numpy(cite_count)

    write_year_path = os.path.join(input_dir, 'writes/year.npy')
    with open(write_year_path, 'wb') as f:
        np.save(f, write_year)
    g.edges[('author', 'writes', 'paper')].data['year'] = torch.from_numpy(write_year)
    g.edges['rev_writes'].data['year'] = torch.from_numpy(write_year)

    writes_orig_ids_path = os.path.join(input_dir, 'writes/orig_ids.npy')
    with open(writes_orig_ids_path, 'wb') as f:
        np.save(f, writes_orig_ids)
    g.edges[('author', 'writes', 'paper')].data['orig_ids'] = torch.from_numpy(writes_orig_ids)

    write2_year_path = os.path.join(input_dir, 'writes2/year.npy')
    with open(write2_year_path, 'wb') as f:
        np.save(f, write2_year)
    g.edges[('institution', 'writes', 'paper')].data['year'] = torch.from_numpy(write2_year)

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

        author_train_mask_path = os.path.join(
            input_dir, 'author/train_mask.npy'
        )
        with open(author_train_mask_path, 'wb') as f:
            np.save(f, author_train_mask)

        author_test_mask_path = os.path.join(input_dir, 'author/test_mask.npy')
        with open(author_test_mask_path, 'wb') as f:
            np.save(f, author_test_mask)

        author_val_mask_path = os.path.join(input_dir, 'author/val_mask.npy')
        with open(author_val_mask_path, 'wb') as f:
            np.save(f, author_val_mask)

        inst_train_mask_path = os.path.join(
            input_dir, 'institution/train_mask.npy'
        )
        with open(inst_train_mask_path, 'wb') as f:
            np.save(f, inst_train_mask)

        inst_test_mask_path = os.path.join(
            input_dir, 'institution/test_mask.npy'
        )
        with open(inst_test_mask_path, 'wb') as f:
            np.save(f, inst_test_mask)

        inst_val_mask_path = os.path.join(input_dir, 'institution/val_mask.npy')
        with open(inst_val_mask_path, 'wb') as f:
            np.save(f, inst_val_mask)

        node_data = {
            'paper': {
                'feat': paper_feat_path,
                'train_mask': paper_train_mask_path,
                'test_mask': paper_test_mask_path,
                'val_mask': paper_val_mask_path,
                'label': paper_label_path,
                'year': paper_year_path,
                'orig_ids': paper_orig_ids_path,
            },
            'author': {
                'train_mask': author_train_mask_path,
                'test_mask': author_test_mask_path,
                'val_mask': author_val_mask_path,
            },
            'institution': {
                'train_mask': inst_train_mask_path,
                'test_mask': inst_test_mask_path,
                'val_mask': inst_val_mask_path,
            },
        }
    else:
        node_data = {
            'paper': {
                'feat': paper_feat_path,
                'label': paper_label_path,
                'year': paper_year_path,
                'orig_ids': paper_orig_ids_path,
            }
        }

    edge_data = {
        'cites': {'count': cite_count_path},
        ('author', 'writes', 'paper'): {
            'year': write_year_path,
            'orig_ids': writes_orig_ids_path
        },
        'rev_writes': {'year': write_year_path},
        ('institution', 'writes', 'paper'): {
            'year': write2_year_path,
        },
    }

    output_dir = os.path.join(root_dir, 'chunked-data')
    chunk_graph(
        g,
        'mag240m',
        node_data,
        edge_data,
        num_chunks=num_chunks,
        output_path=output_dir,
        data_fmt=data_fmt,
    )
    print('Done with creating chunked graph')

    return g
