"""MovieLens dataset"""

import os
import numpy as np
import pandas as pd
import torch as th

import dgl 
import zipfile
import re

from dgl.data.utils import download, save_info, load_info, save_graphs, load_graphs

from dgl.data import DGLDataset

'''
Jiahang Li
[x] TODO: follow dgl dataset flow for ml-100k
[x] TODO: appropriately deal with train, test split of ml-100k
[x] TODO: add side features, text embeddings follow GCMC
[x] TODO: support ml-1M, ml-10M datasets
[] TODO: something wrong with etypes of ml-1M and ml-10M
[] TODO: something wrong with user feats of ml-1M and ml-10M
[] TODO: add other types of text embedding
[] TODO: train and valid split
[] TODO: add an option to specify where word embeddings are stored

'''

GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ["IMAX"]

class MovieLens(DGLDataset):
    """MovieLens dataset used by GCMC model, ml-100k
    """
    _url = {
        'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
        "ml-1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-10m": "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
    } 

    def __init__(self, name='ml-100k', valid_ratio=None, test_ratio=0.1, raw_dir=None, force_reload=None, verbose=None, transform=None):
        self.test_ratio = test_ratio
        if name == "ml-100k":
            self.genres = GENRES_ML_100K
        elif name == "ml-1m":
            self.genres = GENRES_ML_1M
        elif name == "ml-10m":
            self.genres = GENRES_ML_10M
        else:
            raise NotImplementedError
        
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
        # 1. dataset split: train + test
        if self.name == 'ml-100k':
            train_rating_data = self._load_raw_rates(os.path.join(self.raw_path, 'u1.base'), '\t')
            test_rating_data = self._load_raw_rates(os.path.join(self.raw_path, 'u1.test'), '\t')
            all_rating_data = pd.concat([train_rating_data, test_rating_data])
        elif self.name == 'ml-1m' or self.name == 'ml-10m':
            all_rating_data = self._load_raw_rates(
                os.path.join(self.raw_path, "ratings.dat"), "::"
            )
            num_test = int(
                np.ceil(all_rating_data.shape[0] * self.test_ratio)
            )
            shuffled_idx = np.random.permutation(all_rating_data.shape[0])
            test_rating_data = all_rating_data.iloc[
                shuffled_idx[:num_test]
            ]
            train_rating_data = all_rating_data.iloc[
                shuffled_idx[num_test:]
            ]

        # 2. load user and movie data, and drop those unseen in rating_data
        user_data = self._load_raw_user_data()
        movie_data = self._load_raw_movie_data()
        user_data = self._drop_unseen_nodes(data_df=user_data,
                                            col_name="id",
                                            reserved_ids_set=set(all_rating_data["user_id"].values))
        movie_data = self._drop_unseen_nodes(data_df=movie_data,
                                            col_name="id",
                                            reserved_ids_set=set(all_rating_data["movie_id"].values))

        user_feat, movie_feat = th.tensor(self._process_user_fea(user_data)), th.tensor(self._process_movie_fea(movie_data))
        self.feat = {
            "user_feat": user_feat,
            "movie_feat": movie_feat
        }

        # 3. generate rating pairs
        # Map user/movie to the global id
        self._global_user_id_map = {ele: i for i, ele in enumerate(user_data['id'])}
        self._global_movie_id_map = {ele: i for i, ele in enumerate(movie_data['id'])}

        # pair value is idx rather than id, and rating value starts from 1.0
        train_u_indices, train_v_indices, train_labels = self._generate_pair_value(train_rating_data)
        test_u_indices, test_v_indices, test_labels = self._generate_pair_value(test_rating_data)

        # reindex u and v, v nodes start after u
        num_user = len(self._global_user_id_map)
        train_v_indices += num_user
        test_v_indices += num_user

        self.train_rating_pairs = (th.LongTensor(train_u_indices), th.LongTensor(train_v_indices))
        self.test_rating_pairs = (th.LongTensor(test_u_indices), th.LongTensor(test_v_indices))
        self.train_rating_values = th.FloatTensor(train_labels)
        self.test_rating_values = th.FloatTensor(test_labels)
        self.info = {
            "train_rating_pairs": self.train_rating_pairs,
            "test_rating_pairs": self.test_rating_pairs
        }

        # build dgl graph object, which is homogeneous and bidirectional and contains only training edges
        self.train_graph = dgl.graph((self.train_rating_pairs[0], self.train_rating_pairs[1]))
        self.test_graph = dgl.graph((self.test_rating_pairs[0], self.test_rating_pairs[1]))
        self.train_graph.edata['etype'] = self.train_rating_values.to(th.long)
        self.test_graph.edata['etype'] = self.test_rating_values.to(th.long)

        self.train_graph = dgl.add_reverse_edges(self.train_graph, copy_edata=True)
        self.test_graph = dgl.add_reverse_edges(self.test_graph, copy_edata=True)

    def has_cache(self):
        if os.path.exists(self.graph_path):
            return True
        return False

    def save(self):
        save_graphs(self.graph_path, [self.train_graph, self.test_graph])
        save_info(self.info_path, self.info)
        save_info(self.feat_path, self.feat)
        if self.verbose:
            print(f'Done saving data into {self.graph_path}.')

    def load(self):
        g_list, _ = load_graphs(self.graph_path)
        self.info = load_info(self.info_path)
        self.feat = load_info(self.feat_path)
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
    def raw_path(self):
        if self.name == 'ml-10m':
            return os.path.join(self.raw_dir, 'ml-10M100K')
        return super().raw_path

    @property
    def graph_path(self):
        return os.path.join(self.raw_path, self.name + '.bin')
    
    @property
    def feat_path(self):
        return os.path.join(self.raw_path, self.name + '_feat.pkl')
    
    @property
    def info_path(self):
        return os.path.join(self.raw_path, self.name + '.pkl')
    
    def _process_user_fea(self, user_data):
        """
        adopted from GCMC
        Parameters
        ----------
        user_data : pd.DataFrame
        name : str
        For ml-100k and ml-1m, the column name is ['id', 'gender', 'age', 'occupation', 'zip_code'].
            We take the age, gender, and the one-hot encoding of the occupation as the user features.
        For ml-10m, there is no user feature and we set the feature to be a single zero.
        Returns
        -------
        user_features : np.ndarray
        """
        if self.name == "ml-100k" or self.name == "ml-1m":
            ages = user_data["age"].values.astype(np.float32)
            gender = (user_data["gender"] == "F").values.astype(np.float32)
            all_occupations = set(user_data["occupation"])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(
                shape=(user_data.shape[0], len(all_occupations)),
                dtype=np.float32,
            )
            occupation_one_hot[
                np.arange(user_data.shape[0]),
                np.array(
                    [
                        occupation_map[ele]
                        for ele in user_data["occupation"]
                    ]
                ),
            ] = 1
            user_features = np.concatenate(
                [
                    ages.reshape((user_data.shape[0], 1)) / 50.0,
                    gender.reshape((user_data.shape[0], 1)),
                    occupation_one_hot,
                ],
                axis=1,
            )
        elif self.name == "ml-10m":
            user_features = np.zeros(
                shape=(user_data.shape[0], 1), dtype=np.float32
            )
        else:
            raise NotImplementedError
        return user_features
    
    def _process_movie_fea(self, movie_data):
        """
        adopted from GCMC
        Parameters
        ----------
        movie_data : pd.DataFrame
        name :  str
        Returns
        -------
        movie_features : np.ndarray
            Generate movie features by concatenating embedding and the year
        """
        import torchtext
        from torchtext.data.utils import get_tokenizer

        # Old torchtext-legacy API commented below
        # TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
        tokenizer = get_tokenizer(
            "spacy", language="en_core_web_sm"
        )  # new API (torchtext 0.9+)
        embedding = torchtext.vocab.GloVe(name="840B", dim=300)

        title_embedding = np.zeros(
            shape=(movie_data.shape[0], 300), dtype=np.float32
        )
        release_years = np.zeros(
            shape=(movie_data.shape[0], 1), dtype=np.float32
        )
        p = re.compile(r"(.+)\s*\((\d+)\)")
        for i, title in enumerate(movie_data["title"]):
            match_res = p.match(title)
            if match_res is None:
                print(
                    "title {} cannot be matched, index={}, name={}".format(
                        title, i, self.name
                    )
                )
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            # Upgraded torchtext API:  TEXT.tokenize(title_context) --> tokenizer(title_context)
            title_embedding[i, :] = (
                embedding.get_vecs_by_tokens(tokenizer(title_context))
                .numpy()
                .mean(axis=0)
            )
            release_years[i] = float(year)
        movie_features = np.concatenate(
            (
                title_embedding,
                (release_years - 1950.0) / 100.0,
                movie_data[self.genres],
            ),
            axis=1,
        )
        return movie_features
    
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
        if self.name == "ml-100k":
            user_data = pd.read_csv(
                os.path.join(self.raw_path, "u.user"),
                sep="|",
                header=None,
                names=["id", "age", "gender", "occupation", "zip_code"],
                engine="python",
            )
        elif self.name == "ml-1m":
            user_data = pd.read_csv(
                os.path.join(self.raw_path, "users.dat"),
                sep="::",
                header=None,
                names=["id", "gender", "age", "occupation", "zip_code"],
                engine="python",
            )
        elif self.name == "ml-10m":
            rating_info = pd.read_csv(
                os.path.join(self.raw_path, "ratings.dat"),
                sep="::",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
                dtype={
                    "user_id": np.int32,
                    "movie_id": np.int32,
                    "ratings": np.float32,
                    "timestamp": np.int64,
                },
                engine="python",
            )
            user_data = pd.DataFrame(
                np.unique(rating_info["user_id"].values.astype(np.int32)),
                columns=["id"],
            )
        else:
            raise NotImplementedError
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
            For ml-100k, the column name is ['id', 'title', 'release_date', 'video_release_date', 'url'] + [self.genres (19)]]
            For ml-1m, the column name is ['id', 'title'] + [self.genres (18/20)]]
        """

        file_path = os.path.join(self.raw_path, 'u.item')
        if self.name == 'ml-100k':
            movie_data = pd.read_csv(file_path, sep='|', header=None,
                                            names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES_ML_100K,
                                            engine='python', encoding="ISO-8859-1")
        elif self.name == "ml-1m" or self.name == "ml-10m":
            file_path = os.path.join(self.raw_path, "movies.dat")
            movie_data = pd.read_csv(
                file_path,
                sep="::",
                header=None,
                names=["id", "title", "genres"],
                encoding="iso-8859-1",
                engine='python'
            )
            genre_map = {ele: i for i, ele in enumerate(self.genres)}
            genre_map["Children's"] = genre_map["Children"]
            genre_map["Childrens"] = genre_map["Children"]
            movie_genres = np.zeros(
                shape=(movie_data.shape[0], len(self.genres)), dtype=np.float32
            )
            for i, genres in enumerate(movie_data["genres"]):
                for ele in genres.split("|"):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        print(
                            "genres not found, filled with unknown: {}".format(
                                genres
                            )
                        )
                        movie_genres[i, genre_map["unknown"]] = 1.0
            for idx, genre_name in enumerate(self.genres):
                assert idx == genre_map[genre_name]
                movie_data[genre_name] = movie_genres[:, idx]
            movie_data= movie_data.drop(columns=["genres"])
        else:
            raise NotImplementedError

            
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
    movielens = MovieLens(name='ml-10m', verbose=True, force_reload=True)
    train_graph, test_graph = movielens[0]
    info, feat = movielens.info, movielens.feat
    pass
