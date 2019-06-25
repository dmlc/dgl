import pandas as pd
import dgl
import os
import torch
import numpy as np
import scipy.sparse as sp
import time
from functools import partial
from .. import randomwalk
import stanfordnlp
import re
import tqdm
import string
from .base import UserProductDataset
from functools import reduce
import operator

class MovieLens(UserProductDataset):
    split_by_time = None
    inductive = False
    def __init__(self, directory):
        '''
        directory: path to movielens directory which should have the three
                   files:
                   users.dat
                   products.dat
                   ratings.dat
        '''
        self.directory = directory

        users = []
        products = []
        ratings = []

        # read ratings
        with open(os.path.join(directory, 'ratings.dat')) as f:
            for l in f:
                user_id, product_id, rating, timestamp = l.split('::')
                user_id = int(user_id)
                product_id = int(product_id)
                rating = float(rating)
                timestamp = int(timestamp)
                ratings.append({
                    'user_id': user_id,
                    'product_id': product_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        ratings = pd.DataFrame(ratings)
        product_count = ratings['product_id'].value_counts()
        product_count.name = 'product_count'
        ratings = ratings.join(product_count, on='product_id')
        self.ratings = ratings

        # read users
        user_file = os.path.join(directory, 'users.dat')
        if os.path.exists(user_file):
            with open(user_file) as f:
                for l in f:
                    id_, gender, age, occupation, zip_ = l.strip().split('::')
                    users.append({
                        'id': int(id_),
                        'gender': gender,
                        'age': age,
                        'occupation': occupation,
                        'zip': zip_,
                        })
            self.users = pd.DataFrame(users).set_index('id').astype('category')
        else:
            users = [{'id': id_} for id_ in ratings['user_id'].unique()]
            self.users = pd.DataFrame(users).set_index('id')

        # read products
        with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
            for l in f:
                id_, title, genres = l.strip().split('::')
                genres_set = set(genres.split('|'))

                # extract year
                assert re.match(r'.*\([0-9]{4}\)$', title)
                year = title[-5:-1]
                title = title[:-6].strip()

                data = {'id': int(id_), 'title': title, 'year': year}
                for g in genres_set:
                    data[g] = True
                products.append(data)
        self.products = (
                pd.DataFrame(products)
                .set_index('id')
                .fillna(False)
                .astype({'year': 'category'}))
        self.genres = self.products.columns[self.products.dtypes == bool]

        # drop users and products which do not exist in ratings
        self.ratings = self.data_split(self.ratings)
        self.users = self.users[self.users.index.isin(self.ratings['user_id'])]
        self.products = self.products[self.products.index.isin(self.ratings['product_id'])]

        if self.inductive:
            new_users = np.random.choice(self.users.index)
            self.ratings['new_user'] = self.ratings['user_id'].isin(new_users)
            tmp = self.ratings[self.ratings['new_user'], 'train']
            self.ratings[self.ratings['new_user'], 'train'] = \
                    self.ratings[self.ratings['new_user'], 'valid']
            self.ratings[self.ratings['new_user'], 'valid'] = tmp
            self.new_users = new_users
        else:
            self.ratings['new_user'] = False
            self.new_users = []

        self.build_graph()
        self.refresh_mask()
        self.generate_candidates()

    def build_graph(self):
        user_ids = list(self.users.index)
        product_ids = list(self.products.index)
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        product_ids_invmap = {id_: i for i, id_ in enumerate(product_ids)}
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.user_ids_invmap = user_ids_invmap
        self.product_ids_invmap = product_ids_invmap

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(len(user_ids) + len(product_ids))

        # user features
        for user_column in self.users.columns:
            udata = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
            # 0 for padding
            udata[:len(user_ids)] = \
                    torch.LongTensor(self.users[user_column].cat.codes.values.astype('int64') + 1)
            g.ndata[user_column] = udata

        # product genre
        product_genres = torch.from_numpy(self.products[self.genres].values.astype('float32'))
        g.ndata['genre'] = torch.zeros(g.number_of_nodes(), len(self.genres))
        g.ndata['genre'][len(user_ids):len(user_ids) + len(product_ids)] = product_genres

        # product year
        if 'year' in self.products.columns:
            g.ndata['year'] = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
            # 0 for padding
            g.ndata['year'][len(user_ids):len(user_ids) + len(product_ids)] = \
                    torch.LongTensor(self.products['year'].cat.codes.values.astype('int64') + 1)

        # product title
        nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma')
        vocab = set()
        title_words = []
        for t in tqdm.tqdm(self.products['title'].values):
            doc = nlp(t)
            words = set()
            for s in doc.sentences:
                words.update(w.lemma.lower() for w in s.words
                             if not re.fullmatch(r'['+string.punctuation+']+', w.lemma))
            vocab.update(words)
            title_words.append(words)
        vocab = list(vocab)
        vocab_invmap = {w: i for i, w in enumerate(vocab)}
        # bag-of-words
        g.ndata['title'] = torch.zeros(g.number_of_nodes(), len(vocab))
        for i, tw in enumerate(tqdm.tqdm(title_words)):
            g.ndata['title'][i, [vocab_invmap[w] for w in tw]] = 1
        self.vocab = vocab
        self.vocab_invmap = vocab_invmap

        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_product_vertices = [product_ids_invmap[id_] + len(user_ids)
                                 for id_ in self.ratings['product_id'].values]
        self.rating_user_vertices = rating_user_vertices
        self.rating_product_vertices = rating_product_vertices

        g.add_edges(
                rating_user_vertices,
                rating_product_vertices,
                data={'inv': torch.zeros(self.ratings.shape[0], dtype=torch.uint8),
                    'rating': torch.FloatTensor(self.ratings['rating'])})
        g.add_edges(
                rating_product_vertices,
                rating_user_vertices,
                data={'inv': torch.ones(self.ratings.shape[0], dtype=torch.uint8),
                    'rating': torch.FloatTensor(self.ratings['rating'])})
        self.g = g


    def find_neighbors(self, restart_prob, max_nodes, top_T):
        # TODO: replace with more efficient PPR estimation
        neighbor_probs, neighbors = randomwalk.random_walk_distribution_topt(
                self.g, self.g.nodes(), restart_prob, max_nodes, top_T)

        self.user_neighbors = []
        for i in range(len(self.user_ids)):
            user_neighbor = neighbors[i]
            self.user_neighbors.append(user_neighbor.tolist())

        self.product_neighbors = []
        for i in range(len(self.user_ids), len(self.user_ids) + len(self.product_ids)):
            product_neighbor = neighbors[i]
            self.product_neighbors.append(product_neighbor.tolist())

    def refresh_mask(self):
        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(self.ratings['train'].values.astype('uint8'))
        edge_data = {
                'valid': valid_tensor,
                'test': test_tensor,
                'train': train_tensor,
                }

        self.g.edges[self.rating_user_vertices, self.rating_product_vertices].data.update(edge_data)
        self.g.edges[self.rating_product_vertices, self.rating_user_vertices].data.update(edge_data)


class MovieLens20M(MovieLens):
    def __init__(self, directory):
        self.directory = directory

        ratings = pd.read_csv(os.path.join(directory, 'ratings.csv'))
        ratings = ratings.rename({'userId': 'user_id', 'movieId': 'product_id'}, axis=1)
        product_count = ratings['product_id'].value_counts()
        product_count.name = 'product_count'
        ratings = ratings.join(product_count, on='product_id')
        self.ratings = ratings

        self.users = pd.DataFrame({'id': ratings['user_id'].unique()})
        self.users = self.users.set_index('id')

        self.products = pd.read_csv(os.path.join(directory, 'movies.csv'))
        self.products['genres'] = self.products['genres'].map(lambda x: set(x.split('|')))
        self.genres = reduce(operator.or_, self.products['genres'], set())
        for genre in self.genres:
            self.products[genre] = self.products['genres'].map(lambda x: genre in x)
        self.products = self.products.drop('genres', axis=1).set_index('movieId')

        self.ratings = self.data_split(self.ratings)
        self.users = self.users[self.users.index.isin(self.ratings['user_id'])]
        self.products = self.products[self.products.index.isin(self.ratings['product_id'])]

        self.build_graph()
        self.refresh_mask()
        self.generate_candidates()
