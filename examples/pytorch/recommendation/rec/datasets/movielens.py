import pandas as pd
import dgl
import os
import torch
import scipy.sparse as sp
import time
from .. import randomwalk

class MovieLens(object):
    def __init__(self, directory):
        '''
        directory: path to movielens directory which should have the three
                   files:
                   users.dat
                   movies.dat
                   ratings.dat
        '''
        self.directory = directory

        users = []
        movies = []
        ratings = []

        # read users
        with open(os.path.join(directory, 'users.dat')) as f:
            for l in f:
                id_, gender, age, occupation, zip_ = l.split('::')
                users.append({
                    'id': int(id_),
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip': zip_,
                    })
        self.users = pd.DataFrame(users).set_index('id')

        # read movies
        with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
            for l in f:
                id_, title, genres = l.split('::')
                genres_set = set(genres.split('|'))
                data = {'id': int(id_), 'title': title}
                for g in genres_set:
                    data[g] = True
                movies.append(data)
        self.movies = pd.DataFrame(movies).set_index('id')

        # read ratings
        with open(os.path.join(directory, 'ratings.dat')) as f:
            for l in f:
                user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        self.ratings = pd.DataFrame(ratings)

        # drop users and movies which do not exist in ratings
        self.users = self.users[self.users.index.isin(self.ratings['user_id'])]
        self.movies = self.movies[self.movies.index.isin(self.ratings['movie_id'])]

        self.data_split()
        self.build_graph()
        self.find_neighbors(100, 20, 50)

    def data_split(self):
        test_set = self.ratings.sample(frac=0.02, random_state=1).index
        valid_set = self.ratings.sample(frac=0.02, random_state=2).index
        valid_set = valid_set.difference(test_set)
        prev_set = valid_set.union(test_set)
        train_set = self.ratings.sample(frac=0.1, random_state=3).index
        train_set = train_set.difference(prev_set)
        prev_set = prev_set.union(train_set)

        self.ratings['valid'] = self.ratings.index.isin(valid_set)
        self.ratings['test'] = self.ratings.index.isin(test_set)
        self.ratings['train'] = self.ratings.index.isin(train_set)
        self.ratings['prior'] = ~self.ratings.index.isin(prev_set)

    def build_graph(self):
        user_ids = list(self.users.index)
        movie_ids = list(self.movies.index)

        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        movie_ids_invmap = {id_: i for i, id_ in enumerate(movie_ids)}

        g = dgl.DGLGraph()
        g.add_nodes(len(user_ids) + len(movie_ids))
        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_movie_vertices = [movie_ids_invmap[id_] + len(user_ids)
                                 for id_ in self.ratings['movie_id'].values]

        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        train_tensor = torch.from_numpy(self.ratings['train'].values.astype('uint8'))
        prior_tensor = torch.from_numpy(self.ratings['prior'].values.astype('uint8'))
        edge_data = {
                'valid': valid_tensor,
                'test': test_tensor,
                'train': train_tensor,
                'prior': prior_tensor,
                }

        g.add_edges(rating_user_vertices, rating_movie_vertices)
        g.add_edges(rating_movie_vertices, rating_user_vertices)

        #g_mat = sp.coo_matrix(g.adjacency_matrix().to_dense().numpy())
        #self.g = dgl.DGLGraph(g_mat, readonly=True)
        self.g = g

        self.g.edges[rating_user_vertices, rating_movie_vertices].data.update(edge_data)
        self.g.edges[rating_movie_vertices, rating_user_vertices].data.update(edge_data)
        self.user_ids = user_ids
        self.movie_ids = movie_ids

    def find_neighbors(self, n_traces, n_hops, top_T):
        t0 = time.time()
        neighbor_probs, neighbors = randomwalk.random_walk_distribution_topt(
                self.g, self.g.nodes(), n_traces, n_hops, top_T)
        tt = time.time()

        print('Pre-generating neighbors took %.4f seconds' % (tt - t0))
        self.neighbor_probs = neighbor_probs
        self.neighbors = neighbors

        self.user_neighbors = []
        for i in range(len(self.user_ids)):
            user_neighbor = neighbors[i].numpy()
            user_neighbor = user_neighbor[user_neighbor < len(self.user_ids)]
            self.user_neighbors.append(user_neighbor)

        self.movie_neighbors = []
        for i in range(len(self.user_ids), len(self.user_ids) + len(self.movie_ids)):
            movie_neighbor = neighbors[i].numpy()
            movie_neighbor = movie_neighbor[movie_neighbor >= len(self.user_ids)]
            self.movie_neighbors.append(movie_neighbor)
