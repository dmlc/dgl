import os
import ssl
from six.moves import urllib

import pandas as pd
import numpy as np

import torch
import dgl

# === Below data preprocessing code are based on
# https://github.com/twitter-research/tgn

# Preprocess the raw data split each features

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])

            ts = float(e[2])
            label = float(e[3])  # int(e[3])

            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)

# Re index nodes for DGL convience
def reindex(df, bipartite=True):
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df

# Save edge list, features in different file for data easy process data
def run(data_name, bipartite=True):
    PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './data/ml_{}.csv'.format(data_name)
    OUT_FEAT = './data/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)

    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)

# === code from twitter-research-tgn end ===

# If you have new dataset follow by same format in Jodie,
# you can directly use name to retrieve dataset

def TemporalDataset(dataset):
    if not os.path.exists('./data/{}.bin'.format(dataset)):
        if not os.path.exists('./data/{}.csv'.format(dataset)):
            if not os.path.exists('./data'):
                os.mkdir('./data')

            url = 'https://snap.stanford.edu/jodie/{}.csv'.format(dataset)
            print("Start Downloading File....")
            context = ssl._create_unverified_context()
            data = urllib.request.urlopen(url, context=context)
            with open("./data/{}.csv".format(dataset), "wb") as handle:
                handle.write(data.read())

        print("Start Process Data ...")
        run(dataset)
        raw_connection = pd.read_csv('./data/ml_{}.csv'.format(dataset))
        raw_feature = np.load('./data/ml_{}.npy'.format(dataset))
        # -1 for re-index the node
        src = raw_connection['u'].to_numpy()-1
        dst = raw_connection['i'].to_numpy()-1
        # Create directed graph
        g = dgl.graph((src, dst))
        g.edata['timestamp'] = torch.from_numpy(
            raw_connection['ts'].to_numpy())
        g.edata['label'] = torch.from_numpy(raw_connection['label'].to_numpy())
        g.edata['feats'] = torch.from_numpy(raw_feature[1:, :]).float()
        dgl.save_graphs('./data/{}.bin'.format(dataset), [g])
    else:
        print("Data is exist directly loaded.")
        gs, _ = dgl.load_graphs('./data/{}.bin'.format(dataset))
        g = gs[0]
    return g

def TemporalWikipediaDataset():
    # Download the dataset
    return TemporalDataset('wikipedia')

def TemporalRedditDataset():
    return TemporalDataset('reddit')
