"""MovieLens dataset"""

import os
import numpy as np
import pandas as pd
import torch as th

import dgl 
from dgl.data.utils import download, extract_archive, get_download_dir

GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']

class MovieLens(object):
    """MovieLens dataset used by GCMC model, ml-100k
    """
    def __init__(self, valid_ratio=0.2):
        self._data_name = 'ml-100k'

        # 1. download and extract
        download_dir = get_download_dir()
        self._dir = os.path.join(download_dir, 'ml-100k', 'ml-100k')
        if not os.path.exists(self._dir):
            zip_file_path = '{}/{}.zip'.format(download_dir, 'ml-100k')
            download('http://files.grouplens.org/datasets/movielens/ml-100k.zip', path=zip_file_path)
            extract_archive(zip_file_path, '{}/{}'.format(download_dir, 'ml-100k'))
        
        print("Starting processing {} ...".format(self._data_name))

        # 2. load rating data
        train_rating_data = self._load_raw_rates(os.path.join(self._dir, 'u1.base'), '\t')
        test_rating_data = self._load_raw_rates(os.path.join(self._dir, 'u1.test'), '\t')
        all_rating_data = pd.concat([train_rating_data, test_rating_data])
        
        num_valid = int(np.ceil(train_rating_data.shape[0] * valid_ratio))
        shuffled_idx = np.random.permutation(train_rating_data.shape[0])
        valid_rating_data = train_rating_data.iloc[shuffled_idx[: num_valid]]

        self._rating = np.sort(np.unique(all_rating_data["rating"].values))
        
        print("All rating pairs : {}".format(all_rating_data.shape[0]))
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

        # 3. generate rating pairs
        # Map user/movie to the global id
        self._global_user_id_map = {ele: i for i, ele in enumerate(user_data['id'])}
        self._global_movie_id_map = {ele: i for i, ele in enumerate(movie_data['id'])}
        print('Total user number = {}, movie number = {}'.format(len(self._global_user_id_map),
                                                                 len(self._global_movie_id_map)))
        self._num_user = len(self._global_user_id_map)
        self._num_movie = len(self._global_movie_id_map)

        # pair value is idx rather than id, and rating value starts from 1.0
        train_u_indices, train_v_indices, train_labels = self._generate_pair_value(train_rating_data)
        val_u_indices, val_v_indices, val_labels = self._generate_pair_value(valid_rating_data)
        test_u_indices, test_v_indices, test_labels = self._generate_pair_value(test_rating_data)

        # reindex u and v, v nodes start after u
        train_v_indices += self._num_user
        val_v_indices += self._num_user
        test_v_indices += self._num_user

        self.train_rating_pairs = (th.LongTensor(train_u_indices), th.LongTensor(train_v_indices))
        self.valid_rating_pairs = (th.LongTensor(val_u_indices), th.LongTensor(val_v_indices))
        self.test_rating_pairs = (th.LongTensor(test_u_indices), th.LongTensor(test_v_indices))
        self.train_rating_values = th.FloatTensor(train_labels)
        self.valid_rating_values = th.FloatTensor(val_labels)
        self.test_rating_values = th.FloatTensor(test_labels)

        # build dgl graph object, which is homogeneous and bidirectional and contains only training edges
        self.train_graph = dgl.graph((th.cat([self.train_rating_pairs[0], self.train_rating_pairs[1]]), 
                                      th.cat([self.train_rating_pairs[1], self.train_rating_pairs[0]])))
        self.train_graph.edata['etype'] = th.cat([self.train_rating_values, self.train_rating_values]).to(th.long)

    def _load_raw_user_data(self):
        """In MovieLens, the user attributes file have the following formats:

        ml-100k:
        user id | age | gender | occupation | zip code

        Parameters
        ----------
        name : str

        Returns
        -------
        user_data : pd.DataFrame
        """
        
        user_data = pd.read_csv(os.path.join(self._dir, 'u.user'), sep='|', header=None,
                                names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        return user_data

    def _load_raw_movie_data(self):
        """In MovieLens, the movie attributes may have the following formats:

        In ml-100k:

        movie id | movie title | release date | video release date | IMDb URL | [genres]

        Parameters
        ----------
        name : str

        Returns
        -------
        movie_data : pd.DataFrame
            For ml-100k, the column name is ['id', 'title', 'release_date', 'video_release_date', 'url'] + [GENRES (19)]]
            For ml-1m, the column name is ['id', 'title'] + [GENRES (18/20)]]
        """

        file_path = os.path.join(self._dir, 'u.item')
        movie_data = pd.read_csv(file_path, sep='|', header=None,
                                        names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES_ML_100K,
                                        engine='python', encoding="ISO-8859-1")
        return movie_data

    def _load_raw_rates(self, file_path, sep):
        """In MovieLens, the rates have the following format

        ml-100k
        user id \t movie id \t rating \t timestamp

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
        rating_data = rating_data.sample(frac=1).reset_index(drop=True)
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
        # label ranges from 0. to 4.
        rating_values = rating_data["rating"].values.astype(np.float32) - 1.
        return rating_pairs[0], rating_pairs[1], rating_values
