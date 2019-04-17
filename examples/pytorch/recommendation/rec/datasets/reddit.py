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

    def data_split(self, ratings):
        import dask
        dask.config.set(scheduler='processes')
        import dask.dataframe as dd
        print(ratings.shape)
        ratings = dd.from_pandas(ratings, chunksize=524288)
        ratings = ratings.groupby('user_id').apply(
                partial(self.split_user, filter_counts=True))
        ratings['train'] = ratings['prob'] <= 0.95
        ratings['valid'] = (ratings['prob'] > 0.95) & (ratings['prob'] <= 0.98)
        ratings['test'] = ratings['prob'] > 0.98
        ratings = ratings.drop(['prob'], axis=1)
        return ratings.compute().reset_index(drop=True)

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

        features = np.load(self.path + '.features')
        x = torch.zeros(g.number_of_nodes(), features.shape[1])
        x[len(user_ids):] = torch.FloatTensor(features)
        g.ndata['features'] = x
        g.ndata['nid'] = torch.cat([
            torch.arange(1, 1 + len(user_ids)),
            torch.zeros(len(product_ids), dtype=torch.int64)])

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
