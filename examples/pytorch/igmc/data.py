"""MovieLens dataset"""

import os
import scipy.sparse as sp

import numpy as np
import pandas as pd
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
    def __init__(self, data_name, 
                 use_features=False, train_val=False,
                 test_ratio=0.1, valid_ratio=0.2):
        self._data_name = data_name
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        # self.pool = pool

        # download and extract
        download_dir = get_download_dir()
        zip_file_path = '{}/{}.zip'.format(download_dir, data_name)
        download(_urls[data_name], path=zip_file_path)
        extract_archive(zip_file_path, '{}/{}'.format(download_dir, data_name))

        self._dir = os.path.join(download_dir, data_name, data_name)
        
        print("Starting processing {} ...".format(self._data_name))
        self._load_raw_user_info()
        self._load_raw_movie_info()
        
        print('......')
        if self._data_name == 'ml-100k':
            self.all_train_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.base'), '\t')
            self.test_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.test'), '\t')
            self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info])
        elif self._data_name == 'ml-1m':
            self.all_rating_info = self._load_raw_rates(os.path.join(self._dir, 'ratings.dat'), '::')
            num_test = int(np.ceil(self.all_rating_info.shape[0] * self._test_ratio))
            shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
            self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
            self.all_train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]
        else:
            raise NotImplementedError
        print('......')
        num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
        self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
        if train_val:
            self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid:]]
        else:
            self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[:]]
        self.rating_values = np.sort(np.unique(self.train_rating_info["rating"].values))

        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating pairs : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        print("\t\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating pairs  : {}".format(self.test_rating_info.shape[0]))

        self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                                 cmp_col_name="id",
                                                 reserved_ids_set=set(self.all_rating_info["user_id"].values),
                                                 label="user")
        self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                                  cmp_col_name="id",
                                                  reserved_ids_set=set(self.all_rating_info["movie_id"].values),
                                                  label="movie")

        # Map user/movie to the global id
        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)
        
        ### Generate features
        self.user_feature = None
        self.movie_feature = None
        # self.user_feature_shape = (self.num_user, self.num_user)
        # self.movie_feature_shape = (self.num_movie, self.num_movie)
        # info_line = "Feature dim: "
        # info_line += "\nuser: {}".format(self.user_feature_shape)
        # info_line += "\nmovie: {}".format(self.movie_feature_shape)
        # print(info_line)

        self.all_train_rating_pairs, self.all_train_rating_values = self._generate_pair_value(self.all_train_rating_info)
        self.train_rating_pairs, self.train_rating_values = self._generate_pair_value(self.train_rating_info)
        self.valid_rating_pairs, self.valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        self.test_rating_pairs, self.test_rating_values = self._generate_pair_value(self.test_rating_info)

        # Create adjacent matrix
        neutral_rating = 0
        self.rating_mx_train = np.full((self._num_user, self._num_movie), neutral_rating, dtype=np.int32)
        self.rating_mx_train[self.train_rating_pairs] = self.train_rating_values
        # self.rating_mx_train = sp.csr_matrix(self.rating_mx_train)
        
        # Create subgraphs
        # adj_train is rating matrix
        # train_indices is our train_raing_pairs
        # train_labels is our train_rating_values
        # feature is currently None since we don't use side information

        # self.train_graphs, self.val_graphs, self.test_graphs = links2subgraphs(
        #         self.rating_mx_train, self.rating_values, # pool,
        #         self.train_rating_pairs, self.valid_rating_pairs, self.test_rating_pairs,
        #         self.train_rating_values, self.valid_rating_values, self.test_rating_values,
        #         args.hop, args.sample_ratio, args.max_nodes_per_hop, max_node_label=args.hop*2+1,
        #         user_feature=self.user_feature, movie_feature=self.movie_feature,
        #         train_val=train_val, parallel=True)

    @property
    def num_edge_types(self):
        return self.rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    def _load_raw_user_info(self):
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
        user_info : pd.DataFrame
        """
        if self._data_name == 'ml-100k':
            self.user_info = pd.read_csv(os.path.join(self._dir, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        elif self._data_name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(self._dir, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        else:
            raise NotImplementedError

    def _load_raw_movie_info(self):
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
        movie_info : pd.DataFrame
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
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self._data_name == 'ml-1m':
            file_path = os.path.join(self._dir, 'movies.dat')
            movie_info = pd.read_csv(file_path, sep='::', header=None,
                                     names=['id', 'title', 'genres'], engine='python')
            genre_map = {ele: i for i, ele in enumerate(GENRES)}
            genre_map['Children\'s'] = genre_map['Children']
            genre_map['Childrens'] = genre_map['Children']
            movie_genres = np.zeros(shape=(movie_info.shape[0], len(GENRES)), dtype=np.float32)
            for i, genres in enumerate(movie_info['genres']):
                for ele in genres.split('|'):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        print('genres not found, filled with unknown: {}'.format(genres))
                        movie_genres[i, genre_map['unknown']] = 1.0
            for idx, genre_name in enumerate(GENRES):
                assert idx == genre_map[genre_name]
                movie_info[genre_name] = movie_genres[:, idx]
            self.movie_info = movie_info.drop(columns=["genres"])
        else:
            raise NotImplementedError

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
        rating_info : pd.DataFrame
        """
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python')
        return rating_info

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        # print(" -----------------")
        # print("{}: {}(reserved) v.s. {}(from info)".format(label, len(reserved_ids_set),
        #                                                      len(set(orign_info[cmp_col_name].values))))
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            # print("\torign_info: ({}, {})".format(orign_info.shape[0], orign_info.shape[1]))
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            # print("\tAfter dropping, data shape: ({}, {})".format(data_info.shape[0], data_info.shape[1]))
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

if __name__ == '__main__':
    from utils import links2subgraphs

    dataset = MovieLens("ml-100k", use_features=False, train_val=True)
    train_graphs, val_graphs, test_graphs = links2subgraphs(
            dataset.rating_mx_train, dataset.rating_values, # pool,
            dataset.train_rating_pairs, dataset.valid_rating_pairs, dataset.test_rating_pairs,
            dataset.train_rating_values, dataset.valid_rating_values, dataset.test_rating_values,
            hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1,
            user_feature=dataset.user_feature, movie_feature=dataset.movie_feature,
            train_val=True, parallel=True)
    