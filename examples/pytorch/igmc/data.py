"""MovieLens dataset"""

import os
import numpy as np
import pandas as pd
import torch as th

import dgl 
import zipfile

from dgl import save_graphs, load_graphs
from dgl.data.utils import download, save_info

from dgl.data import DGLDataset

'''
Jiahang Li
[x] TODO: follow dgl dataset flow for ml-100k
[x] TODO: appropriately deal with train, test split of ml-100k
[] TODO: add side features
[] TODO: add options to control the use of text embeddings
[] TODO: support ml-1M, ml-10M datasets
[] TODO: train and valid split of ml-100k

'''

GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']

class MovieLens(DGLDataset):
    """MovieLens dataset used by GCMC model, ml-100k
    """
    _url = {
        'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    }

    def __init__(self, name='ml-100k', raw_dir=None, force_reload=None, verbose=None, transform=None):
        super(MovieLens, self).__init__(name=name, url=self._url[name], raw_dir=raw_dir, force_reload=force_reload, verbose=verbose,
                                        transform=transform)

    def download(self):
        if self.url is not None:
            zip_file_path = os.path.join(self.raw_dir, self.name + ".zip")
            download(self.url, path=zip_file_path)

            with zipfile.ZipFile(zip_file_path, "r") as archive:
                archive.extractall(path=self.raw_dir)

    def process(self):
        print("Starting processing {} ...".format(self.name))
        train_rating_data = self._load_raw_rates(os.path.join(self.raw_path, 'u1.base'), '\t')
        test_rating_data = self._load_raw_rates(os.path.join(self.raw_path, 'u1.test'), '\t')
        all_rating_data = pd.concat([train_rating_data, test_rating_data])

        self._rating = np.sort(np.unique(all_rating_data["rating"].values))

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
    
        self._num_user = len(self._global_user_id_map)
        self._num_movie = len(self._global_movie_id_map)

        # pair value is idx rather than id, and rating value starts from 1.0
        train_u_indices, train_v_indices, train_labels = self._generate_pair_value(train_rating_data)
        test_u_indices, test_v_indices, test_labels = self._generate_pair_value(test_rating_data)

        # reindex u and v, v nodes start after u
        train_v_indices += self._num_user
        test_v_indices += self._num_user

        self.train_rating_pairs = (th.LongTensor(train_u_indices), th.LongTensor(train_v_indices))
        self.test_rating_pairs = (th.LongTensor(test_u_indices), th.LongTensor(test_v_indices))
        self.train_rating_values = th.FloatTensor(train_labels)
        self.test_rating_values = th.FloatTensor(test_labels)

        # build dgl graph object, which is homogeneous and bidirectional and contains only training edges
        self.train_graph = dgl.graph(self.train_rating_pairs[0], self.train_rating_pairs[1])
        self.test_graph = dgl.graph(self.test_rating_pairs[0], self.test_rating_pairs[1])
        self.train_graph.edata['etype'] = self.train_rating_values, self.train_rating_values.to(th.long)
        self.test_graph.edata['etype'] = self.test_rating_values.to(th.long)

        self.train_graph = dgl.to_bidirected(self.train_graph, copy_ndata=True)
        self.test_graph = dgl.to_bidirected(self.test_graph, copy_ndata=True)

    def has_cache(self):
        if os.path.exists(self.graph_path):
            return True
        return False

    def save(self):
        save_graphs(self.graph_path, [self.train_graph, self.test_graph])
        if self.verbose:
            print(f'Done saving data into {self.graph_path}.')

    def load(self):
        g_list, _ = load_graphs(self.graph_path)
        self.train_graph, self.test_graph = g_list[0], g_list[1]
        if self.verbose:
            print(f'Done loading data from {self.graph_path}.')

            print("All rating pairs : {}".format(
                int((self.train_graph.num_edges() + self.test_graph.num_edges()) / 2))
            )
            print("\tTrain rating pairs : {}".format(int(self.train_graph.num_edges() / 2)))
            print("\tTest rating pairs  : {}".format(int(self.test_graph.num_edges() / 2)))
    
    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self.train_graph, self.test_graph
        else:
            return self._transform(self.train_graph), self._transform(self.test_graph)
        
    def __len__(self):
        return 1
    
    @property
    def graph_path(self):
        return os.path.join(self.save_path, self.name + '.bin')
    
    @property
    def info_path(self):
        return os.path.join(self.save_path, self.name + '.pkl')

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
        
        user_data = pd.read_csv(os.path.join(self.raw_path, 'u.user'), sep='|', header=None,
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

        file_path = os.path.join(self.raw_path, 'u.item')
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

if __name__ == '__main__':
    train_graph, test_graph = MovieLens(verbose=True)[0]
    pass
