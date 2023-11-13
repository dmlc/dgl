import dask.dataframe as dd

import dgl
import numpy as np
import scipy.sparse as ssp
import torch
import tqdm


# This is the train-test split method most of the recommender system papers running on MovieLens
# takes.  It essentially follows the intuition of "training on the past and predict the future".
# One can also change the threshold to make validation and test set take larger proportions.
def train_test_split_by_time(df, timestamp, user):
    df["train_mask"] = np.ones((len(df),), dtype=np.bool_)
    df["val_mask"] = np.zeros((len(df),), dtype=np.bool_)
    df["test_mask"] = np.zeros((len(df),), dtype=np.bool_)
    df = dd.from_pandas(df, npartitions=10)

    def train_test_split(df):
        df = df.sort_values([timestamp])
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    meta_df = {
        "user_id": np.int64,
        "movie_id": np.int64,
        "rating": np.int64,
        "timestamp": np.int64,
        "user_id": np.int64,
        "train_mask": bool,
        "val_mask": bool,
        "test_mask": bool,
    }

    df = (
        df.groupby(user, group_keys=False)
        .apply(train_test_split, meta=meta_df)
        .compute(scheduler="processes")
        .sort_index()
    )
    print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
    return (
        df["train_mask"].to_numpy().nonzero()[0],
        df["val_mask"].to_numpy().nonzero()[0],
        df["test_mask"].to_numpy().nonzero()[0],
    )


def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices}, relabel_nodes=False
    )

    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[
                train_g.edges[etype].data[dgl.EID]
            ]

    return train_g


def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.num_nodes(utype)
    n_items = g.num_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix(
        (np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items)
    )
    test_matrix = ssp.coo_matrix(
        (np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items)
    )

    return val_matrix, test_matrix


def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / (
        values.max(0, keepdims=True) - values.min(0, keepdims=True)
    )
