import dgl
import numpy as np
import pickle
import pandas as pd
from .base import UserProductDataset
from functools import partial

class Reddit(UserProductDataset):
    def __init__(self, path):
        super(Reddit, self).__init__()

        self.path = path

        with open(path, 'rb') as f:
            data = pickle.load(f)

        subm = data['subm'].values
        users = data['users'].values
        n_subm = data['n_subm']
        n_users = data['n_users']

        self.users = pd.DataFrame({'id': np.unique(users)}).set_index('id')
        self.products = pd.DataFrame({'id': np.unique(subm)}).set_index('id')
        ratings = pd.DataFrame({'user_id': users, 'product_id': subm})
        product_count = ratings['product_id'].value_counts()
        product_count.name = 'product_count'
        ratings = ratings.join(product_count, on='product_id')
        self.ratings = ratings

        print('data split')
        self.ratings = self.data_split(self.ratings)
        print('build graph')
        self.build_graph()
        #print('find neighbors')
        #self.find_neighbors(0.2, 5000, 1000)

    def split_user(self, df, filter_counts=False):
        df_new = df.copy()
        df_new['prob'] = 0

        if filter_counts:
            df_new_sub = (df_new['product_count'] >= 10).nonzero()[0]
        else:
            df_new_sub = df_new['train'].nonzero()[0]
        prob = np.linspace(0, 1, df_new_sub.shape[0], endpoint=False)
        np.random.shuffle(prob)
        df_new['prob'].iloc[df_new_sub] = prob
        return df_new

    def data_split(self, ratings):
        import dask
        dask.config.set(scheduler='processes')
        import dask.dataframe as dd
        print(ratings.shape)
        ratings = dd.from_pandas(ratings, chunksize=524288)
        ratings = ratings.groupby('user_id').apply(
                partial(self.split_user, filter_counts=True))
        ratings['train'] = ratings['prob'] <= 0.8
        ratings['valid'] = (ratings['prob'] > 0.8) & (ratings['prob'] <= 0.9)
        ratings['test'] = ratings['prob'] > 0.9
        ratings = ratings.drop(['prob'], axis=1)
        return ratings.compute()

    def build_graph(self):
        import torch
        user_ids = list(self.users.index)
        product_ids = list(self.products.index)
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        product_ids_invmap = {id_: i for i, id_ in enumerate(product_ids)}
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.user_ids_invmap = user_ids_invmap
        self.product_ids_invmap = product_ids_invmap

        g = dgl.DGLGraph()
        g.add_nodes(len(user_ids) + len(product_ids))

        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_product_vertices = [product_ids_invmap[id_] + len(user_ids)
                                 for id_ in self.ratings['product_id'].values]
        self.rating_user_vertices = rating_user_vertices
        self.rating_product_vertices = rating_product_vertices

        g.add_edges(
                rating_user_vertices,
                rating_product_vertices,
                data={'inv': torch.zeros(self.ratings.shape[0], dtype=torch.uint8)})
        g.add_edges(
                rating_product_vertices,
                rating_user_vertices,
                data={'inv': torch.ones(self.ratings.shape[0], dtype=torch.uint8)})
        self.g = g
