"""MovieLens dataset"""

import os
import scipy.sparse as sp

import numpy as np
import pandas as pd
import torch as th

import dgl 
from dgl.data.utils import download, extract_archive, get_download_dir

_urls = {
    'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m' : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
}

GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]

class MovieLens(object):
    """MovieLens dataset used by GCMC model
    """
    def __init__(self, data_name, testing=False,
                 test_ratio=0.1, valid_ratio=0.2):
        self._data_name = data_name

        # 1. download and extract
        download_dir = get_download_dir()
        zip_file_path = '{}/{}.zip'.format(download_dir, data_name)
        download(_urls[data_name], path=zip_file_path)
        extract_archive(zip_file_path, '{}/{}'.format(download_dir, data_name))
        self._dir = os.path.join(download_dir, data_name, data_name)
        
        print("Starting processing {} ...".format(self._data_name))

        # 2. load rating data
        if self._data_name == 'ml-100k':
            train_rating_data = self._load_raw_rates(os.path.join(self._dir, 'u1.base'), '\t')
            test_rating_data = self._load_raw_rates(os.path.join(self._dir, 'u1.test'), '\t')
            all_rating_data = pd.concat([train_rating_data, test_rating_data])
        elif self._data_name == 'ml-1m':
            all_rating_data = self._load_raw_rates(os.path.join(self._dir, 'ratings.dat'), '::')
            num_test = int(np.ceil(all_rating_data.shape[0] * test_ratio))
            shuffled_idx = np.random.permutation(all_rating_data.shape[0])
            test_rating_data = all_rating_data.iloc[shuffled_idx[: num_test]]
            train_rating_data = all_rating_data.iloc[shuffled_idx[num_test: ]]
        else:
            raise NotImplementedError
        num_valid = int(np.ceil(train_rating_data.shape[0] * valid_ratio))
        shuffled_idx = np.random.permutation(train_rating_data.shape[0])
        valid_rating_data = train_rating_data.iloc[shuffled_idx[: num_valid]]
        if not testing:
            train_rating_data = train_rating_data.iloc[shuffled_idx[num_valid:]]

        self._rating = np.sort(np.unique(all_rating_data["rating"].values))
        
        print("All rating pairs : {}".format(all_rating_data.shape[0]))
        # print("\tAll train rating pairs : {}".format(self.all_train_rating_data.shape[0]))
        print("\tTrain rating pairs : {}".format(train_rating_data.shape[0]))
        print("\tValid rating pairs : {}".format(valid_rating_data.shape[0]))
        print("\tTest rating pairs  : {}".format(test_rating_data.shape[0]))

        # 2. load user and movie data, and drop those unseen in rating_data
        user_data = self._load_raw_user_data()
        movie_data = self._load_raw_movie_data()
        user_data = self._drop_unseen_nodes(data_df=user_data,
                                            col_name="id",
                                            reserved_ids_set=set(all_rating_data["user_id"].values))
        movie_data = self._drop_unseen_nodes(data_df=movie_data,
                                            col_name="id",
                                            reserved_ids_set=set(all_rating_data["movie_id"].values))

        # 3. set user and movie feature to None
        user_feature = None
        movie_feature = None

        # 4. generate rating pairs
        # Map user/movie to the global id
        self._global_user_id_map = {ele: i for i, ele in enumerate(user_data['id'])}
        self._global_movie_id_map = {ele: i for i, ele in enumerate(movie_data['id'])}
        print('Total user number = {}, movie number = {}'.format(len(self._global_user_id_map),
                                                                 len(self._global_movie_id_map)))
        self._num_user = len(self._global_user_id_map)
        self._num_movie = len(self._global_movie_id_map)

        # TODO [zhoujf] expose train_gaph only
        # pair value is idx rather than id, and rating value starts from 1.0
        # self.all_train_rating_pairs, self.all_train_rating_values = self._generate_pair_value(self.all_train_rating_data)
        self.train_rating_pairs, self.train_rating_values = self._generate_pair_value(train_rating_data)
        self.valid_rating_pairs, self.valid_rating_values = self._generate_pair_value(valid_rating_data)
        self.test_rating_pairs, self.test_rating_values = self._generate_pair_value(test_rating_data)

        # 5. build traing graph 
        # Create adjacent matrix
        neutral_rating = 0
        self.rating_mx_train = np.full((self._num_user, self._num_movie), neutral_rating, dtype=np.int32)
        self.rating_mx_train[self.train_rating_pairs] = self.train_rating_values
        self.rating_mx_train = sp.csr_matrix(self.rating_mx_train)
        
        # train_user_ids = th.LongTensor(self.train_rating_pairs[0])
        # train_movie_ids = th.LongTensor(self.train_rating_pairs[1])
        # train_ratings = th.FloatTensor(self.train_rating_values)
        # self.train_graph = dgl.heterograph({
        #     # Heterogeneous graphs are organized as a dictionary of edges connecting two types of nodes.
        #     # We specify the edges of a type simply with a pair of user ID array and item ID array.
        #     ('user', 'watched', 'movie'): (train_user_ids, train_movie_ids), 
        #     # Since DGL graphs are directional, we need an inverse relation from items to users as well.
        #     ('movie', 'watched-by', 'user'): (train_movie_ids, train_user_ids)
        # })
        # self.train_graph.edges['watched'].data['rating'] = train_ratings
        # self.train_graph.edges['watched-by'].data['rating'] = train_ratings

    @property
    def num_rating(self):
        return self._rating.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    def _load_raw_user_data(self):
        """In MovieLens, the user attributes file have the following formats:

        ml-100k:
        user id | age | gender | occupation | zip code

        ml-1m:
        UserID::Gender::Age::Occupation::Zip-code

        Parameters
        ----------
        name : str

        Returns
        -------
        user_data : pd.DataFrame
        """
        if self._data_name == 'ml-100k':
            user_data = pd.read_csv(os.path.join(self._dir, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        elif self._data_name == 'ml-1m':
            user_data = pd.read_csv(os.path.join(self._dir, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        else:
            raise NotImplementedError
        return user_data

    def _load_raw_movie_data(self):
        """In MovieLens, the movie attributes may have the following formats:

        In ml_100k:

        movie id | movie title | release date | video release date | IMDb URL | [genres]

        In ml_1m, ml_10m:

        MovieID::Title (Release Year)::Genres

        Also, Genres are separated by |, e.g., Adventure|Animation|Children|Comedy|Fantasy

        Parameters
        ----------
        name : str

        Returns
        -------
        movie_data : pd.DataFrame
            For ml-100k, the column name is ['id', 'title', 'release_date', 'video_release_date', 'url'] + [GENRES (19)]]
            For ml-1m, the column name is ['id', 'title'] + [GENRES (18/20)]]
        """
        if self._data_name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._data_name == 'ml-1m':
            GENRES = GENRES_ML_1M
        else:
            raise NotImplementedError

        if self._data_name == 'ml-100k':
            file_path = os.path.join(self._dir, 'u.item')
            movie_data = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self._data_name == 'ml-1m':
            file_path = os.path.join(self._dir, 'movies.dat')
            movie_data = pd.read_csv(file_path, sep='::', header=None,
                                     names=['id', 'title', 'genres'], engine='python')
            genre_map = {ele: i for i, ele in enumerate(GENRES)}
            genre_map['Children\'s'] = genre_map['Children']
            genre_map['Childrens'] = genre_map['Children']
            movie_genres = np.zeros(shape=(movie_data.shape[0], len(GENRES)), dtype=np.float32)
            for i, genres in enumerate(movie_data['genres']):
                for ele in genres.split('|'):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        print('genres not found, filled with unknown: {}'.format(genres))
                        movie_genres[i, genre_map['unknown']] = 1.0
            for idx, genre_name in enumerate(GENRES):
                assert idx == genre_map[genre_name]
                movie_data[genre_name] = movie_genres[:, idx]
            movie_data = movie_data.drop(columns=["genres"])
        else:
            raise NotImplementedError
        return movie_data

    def _load_raw_rates(self, file_path, sep):
        """In MovieLens, the rates have the following format

        ml-100k
        user id \t movie id \t rating \t timestamp

        ml-1m/10m
        UserID::MovieID::Rating::Timestamp

        timestamp is unix timestamp and can be converted by pd.to_datetime(X, unit='s')

        Parameters
        ----------
        file_path : str

        Returns
        -------
        rating_data : pd.DataFrame
        """
        rating_data = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python')
        return rating_data

    def _drop_unseen_nodes(self, data_df, col_name, reserved_ids_set):
        data_df = data_df[data_df[col_name].isin(reserved_ids_set)]
        data_df.reset_index(drop=True, inplace=True)
        return data_df

    def _generate_pair_value(self, rating_data):
        rating_pairs = (np.array([self._global_user_id_map[ele] for ele in rating_data["user_id"]],
                                 dtype=np.int32),
                        np.array([self._global_movie_id_map[ele] for ele in rating_data["movie_id"]],
                                 dtype=np.int32))
        rating_values = rating_data["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

if __name__ == '__main__':
    from utils import links2subgraphs

    dataset = MovieLens("ml-100k", train_val=True)
    train_graphs, val_graphs, test_graphs = links2subgraphs(
            dataset.rating_mx_train, # dataset.rating_values, pool,
            dataset.train_rating_pairs, dataset.valid_rating_pairs, dataset.test_rating_pairs,
            dataset.train_rating_values, dataset.valid_rating_values, dataset.test_rating_values,
            hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1,
            train_val=True, parallel=False)
    