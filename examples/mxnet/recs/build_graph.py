import os, sys
from dgl.data.utils import download, get_download_dir
import mxnet as mx
import numpy as np
from dgl import DGLGraph
from zipfile import ZipFile
from datetime import datetime


_urls = {
    'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m'   : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-10m'  : 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
}

_path2rating = {
    'ml-100k':'ml-100k/u.data',
    'ml-1m':'ml-1m/ratings.dat',
    'ml-10m':'ml-10M100K/ratings.dat',
}

_get_sep = lambda k:{
    'ml-1m'  : '::',
    'ml-10m' : '::',
}.get(k, None)


def find_idx(zf, path2rating, sep):
    users = {}
    items = {}
    ratings = {}
    ts_min, ts_max = np.inf, -np.inf
    with zf.open(path2rating) as f:
        for line in f:
            user, item, rating, timestamp = line.decode('latin1').strip('\n').split(sep)
            
            users.setdefault(user, 0)
            users[user] += 1
            
            items.setdefault(item, 0)
            items[item] += 1
            
            ratings.setdefault(rating, 0)
            ratings[rating] += 1

            ts_min = min(ts_min, float(timestamp))
            ts_max = max(ts_max, float(timestamp))

    user_idx = {k:i
                for i,k in enumerate(sorted(users))}

    item_idx = {k:i+len(users)
                for i,k in enumerate(sorted(items))}

    rating_idx = {k:i for i,k in
                  enumerate(sorted(ratings.keys()))}

    return user_idx, item_idx, rating_idx, (ts_min, ts_max)

def load_data(dataset, ctx, seed=42, train_ratio=0.8, val_ratio=0.1):
    download(_urls[dataset],
         '{}/{}.zip'.format(get_download_dir(), dataset))
    zf = ZipFile('{}/{}.zip'.format(get_download_dir(), dataset))
    print(zf.namelist())
    print('loading dataset {}'.format(dataset))

    user_idx, item_idx, rating_idx, ts_range = find_idx(
        zf, _path2rating[dataset], _get_sep(dataset))

    print(' num unique users {}'.format(len(user_idx)))
    print(' num unique items {}'.format(len(item_idx)))
    print(' unique ratings {}'.format(rating_idx))
    print(' time range {} - {}'.format(*[
        datetime.utcfromtimestamp(ts) for ts in ts_range]))

    G = DGLGraph()

    G.add_nodes(len(user_idx) + len(item_idx))

    G.ndata['is_user'] = mx.nd.concat(
    mx.nd.ones(len(user_idx), ctx=ctx),
    mx.nd.zeros(len(item_idx), ctx=ctx),
    dim=0)

    u = []
    v = []
    r = []
    t = []
    with zf.open(_path2rating[dataset]) as f:
        for line in f:
            user, item, rating, timestamp = line.decode('latin1').strip('\n').split(_get_sep(dataset))
            u.append(user_idx[user])
            v.append(item_idx[item])
            r.append(rating_idx[rating])
            t.append((float(timestamp) - ts_range[0]) /
                     (ts_range[1] - ts_range[0] + 1e-10))

    G.add_edges(u, v, {
        'r'         : mx.nd.array(r, dtype='float32', ctx=ctx),
        'time_pct'  : mx.nd.array(t, dtype='float32', ctx=ctx),
        'user2item' : mx.nd.ones(len(r), dtype='float32', ctx=ctx),
    })

    rng = np.random.RandomState(seed)
    G.edata['_rand'] = mx.nd.array(rng.rand(len(G.edges)),
        dtype='float32', ctx=ctx)

    G.edata['is_train'] = G.edata['_rand'] <= train_ratio
    G.edata['is_val']   = (G.edata['_rand'] > train_ratio) * \
                          (G.edata['_rand'] <= train_ratio + val_ratio)
    G.edata['is_test']  = G.edata['_rand'] > train_ratio + val_ratio

    G.add_edges(*reversed(G.all_edges()), data={
        'r'         : G.edata['r'],
        'time_pct'  : G.edata['time_pct'],
        'user2item' : mx.nd.zeros_like(G.edata['user2item']),
        'is_train'  : G.edata['is_train'],
        'is_val'    : G.edata['is_val'],
        'is_test'   : G.edata['is_test'],
    })

    print(' num user and item nodes {}'.format(len(G.nodes)))
    print(' num directed edges {}'.format(len(G.edges)))

    return G, len(rating_idx)
