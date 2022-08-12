import dgl
import numpy as np
import os
import torch

from chunk_graph import chunk_graph

def test_chunk_graph():
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
    input_dir = '/data_test'
    os.makedirs(input_dir)
    for sub_d in ['paper', 'cites', 'writes']:
        os.makedirs(os.path.join(input_dir, sub_d))

    for fname, feat in {
        'paper/feat.npy': paper_feat,
        'paper/label.npy': paper_label,
        'paper/year.npy': paper_year,
        'cites/count.npy': cite_count,
        'writes/year.npy': write_year}.items():
        np.save(os.path.join(input_dir, fname), feat)

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
        4,
        output_dir)
