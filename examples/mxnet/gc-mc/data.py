import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import gluonnlp as nlp
import networkx as nx
import dgl
import mxnet as mx

READ_DATASET_PATH = os.path.join("data_set")
GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

_word_embedding = nlp.embedding.GloVe('glove.840B.300d')
_tokenizer = nlp.data.transforms.SpacyTokenizer()

class MovieLens(object):
    def __init__(self, name, ctx, symm=True,
                 test_ratio=0.1, valid_ratio = 0.1):
        self._name = name
        self._ctx = ctx
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        print("Starting processing {} ...".format(self._name))
        self._load_raw_user_info()
        self._load_raw_movie_info()
        if self._name == 'ml-100k':
            train_rating_info = self._load_raw_rates(os.path.join(READ_DATASET_PATH, self._name, 'u1.base'), '\t')
            self.test_rating_info = self._load_raw_rates(os.path.join(READ_DATASET_PATH, self._name, 'u1.test'), '\t')
            self.all_rating_info = pd.concat([train_rating_info, self.test_rating_info])
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            self.all_rating_info = self._load_raw_rates(os.path.join(READ_DATASET_PATH, self._name, 'ratings.dat'), '::')
            num_test = int(np.ceil(self.all_rating_info.shape[0] * self._test_ratio))
            shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
            self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
            train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]
        else:
            raise NotImplementedError
        num_valid = int(np.ceil(train_rating_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(train_rating_info.shape[0])
        self.valid_rating_info = train_rating_info.iloc[shuffled_idx[: num_valid]]
        self.train_rating_info = train_rating_info.iloc[shuffled_idx[num_valid: ]]

        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        print("\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
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
        print("  -----------------")
        print("Generating user id map and movie id map ...")
        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)

        ### Generate features
        self._process_user_fea()
        self._process_movie_fea()
        #print("user_features: shape ({},{})".format(self.user_features.shape[0], self.user_features.shape[1]))
        #print("movie_features: shape ({},{})".format(self.movie_features.shape[0], self.movie_features.shape[1]))

        self.train_rating_pairs, self.train_rating_values = self._generata_pair_value(self.train_rating_info)
        self.valid_rating_pairs, self.valid_rating_values = self._generata_pair_value(self.valid_rating_info)
        self.test_rating_pairs, self.test_rating_values = self._generata_pair_value(self.test_rating_info)

        self.uv_train_graph, self.vu_train_graph = self._generate_graphs(self.train_rating_pairs,
                                                                         self.train_rating_values)

    def _generata_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele]
                                  for ele in rating_info["user_id"]], dtype=np.int64),
                        np.array([self.global_movie_id_map[ele]
                                  for ele in rating_info["movie_id"]], dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)

        return rating_pairs, rating_values

    def _generate_graphs(self, rating_pairs, rating_values):
        user_movie_ratings_coo = sp.coo_matrix(
            (rating_values, rating_pairs),
            shape=(self._num_user, self._num_movie),dtype=np.float32)
        movie_user_ratings_coo = user_movie_ratings_coo.transpose()

        user_movie_R = np.zeros((self._num_user, self._num_movie), dtype=np.float32)
        user_movie_R[rating_pairs] = rating_values
        movie_user_R = user_movie_R.transpose()

        uv_graph = dgl.DGLBipartiteGraph(
            metagraph=nx.MultiGraph([('user', 'movie', 'rating')]),
            number_of_nodes_by_type={'user': self._num_user,
                                     'movie': self._num_movie},
            edge_connections_by_type={('user', 'movie', 'rating'): user_movie_ratings_coo},
            # node_frame={"user": self.user_features, "movie": self.movie_features},
            readonly=True)

        vu_graph = dgl.DGLBipartiteGraph(
            metagraph=nx.MultiGraph([('movie', 'user', 'rating')]),
            number_of_nodes_by_type={'user': self._num_user,
                                     'movie': self._num_movie},
            edge_connections_by_type={('movie', 'user', 'rating'): movie_user_ratings_coo},
            # node_frame={"user": self.user_features, "movie": self.movie_features},
            readonly=True)

        uv_train_support_l = self.compute_support(user_movie_R, self.num_links, self._symm)
        for idx, support in enumerate(uv_train_support_l):
            sup_coo = support.tocoo()
            uv_graph.edges[np.array(sup_coo.row, dtype=np.int64),
                           np.array(sup_coo.col, dtype=np.int64)].data['support{}'.format(idx)] = \
                mx.nd.array(sup_coo.data, ctx=self._ctx, dtype=np.float32)

        vu_train_support_l = self.compute_support(movie_user_R, self.num_links, self._symm)
        for idx, support in enumerate(vu_train_support_l):
            sup_coo = support.tocoo()
            vu_graph.edges[np.array(sup_coo.row, dtype=np.int64),
                           np.array(sup_coo.col, dtype=np.int64)].data['support{}'.format(idx)] = \
                mx.nd.array(sup_coo.data, ctx=self._ctx, dtype=np.float32)

        return uv_graph, vu_graph

    @property
    def possible_rating_values(self):
        return np.unique(self.train_rating_info["rating"].values)

    @property
    def num_links(self):
        return self.possible_rating_values.size
    @property
    def name_user(self):
        return "user"

    @property
    def name_movie(self):
        return "movie"

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        print("  -----------------")
        print("{}: {}(reserved) v.s. {}(from info)".format(label, len(reserved_ids_set),
                                                             len(set(orign_info[cmp_col_name].values))))
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            print("\torign_info: ({}, {})".format(orign_info.shape[0], orign_info.shape[1]))
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            print("\tAfter dropping, data shape: ({}, {})".format(data_info.shape[0], data_info.shape[1]))
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info

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

    def _load_raw_user_info(self):
        """In MovieLens, the user attributes file have the following formats:

        ml-100k:
        user id | age | gender | occupation | zip code

        ml-1m:
        UserID::Gender::Age::Occupation::Zip-code

        For ml-10m, there is no user information. We read the user id from the rating file.

        Parameters
        ----------
        name : str

        Returns
        -------
        user_info : pd.DataFrame
        """
        if self._name == 'ml-100k':
            self.user_info = pd.read_csv(os.path.join(READ_DATASET_PATH, self._name, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        elif self._name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(READ_DATASET_PATH, self._name, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        elif self._name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(READ_DATASET_PATH, self._name, 'ratings.dat'), sep='::', header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                dtype={'user_id': np.int32, 'movie_id': np.int32, 'ratings': np.float32,
                       'timestamp': np.int64}, engine='python')
            self.user_info = pd.DataFrame(np.unique(rating_info['user_id'].values.astype(np.int32)),
                                     columns=['id'])
        else:
            raise NotImplementedError

    def _process_user_fea(self):
        """

        Parameters
        ----------
        user_info : pd.DataFrame
        name : str
        For ml-100k and ml-1m, the column name is ['id', 'gender', 'age', 'occupation', 'zip_code'].
            We take the age, gender, and the one-hot encoding of the occupation as the user features.
        For ml-10m, there is no user feature and we set the feature to be a single zero.

        Returns
        -------
        user_features : np.ndarray

        """
        if self._name == 'ml-100k' or self._name == 'ml-1m':
            ages = self.user_info['age'].values.astype(np.float32)
            gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
            all_occupations = set(self.user_info['occupation'])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)),
                                          dtype=np.float32)
            occupation_one_hot[np.arange(self.user_info.shape[0]),
                               np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
            self.user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                            gender.reshape((self.user_info.shape[0], 1)),
                                            occupation_one_hot], axis=1)
        elif self._name == 'ml-10m':
            self.user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)
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
            For ml-1m and ml-10m, the column name is ['id', 'title'] + [GENRES (18/20)]]
        """
        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        if self._name == 'ml-100k':
            file_path = os.path.join(READ_DATASET_PATH, self._name, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            file_path = os.path.join(READ_DATASET_PATH, self._name, 'movies.dat')
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

    def _process_movie_fea(self):
        """

        Parameters
        ----------
        movie_info : pd.DataFrame
        name :  str

        Returns
        -------
        movie_features : np.ndarray
            Generate movie features by concatenating embedding and the year

        """
        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)
        p = re.compile(r'(.+)\s*\((\d+)\)')
        for i, title in enumerate(self.movie_info['title']):
            match_res = p.match(title)
            if match_res is None:
                print('{} cannot be matched, index={}, name={}'.format(title, i, self._name))
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            title_embedding[i, :] =_word_embedding[_tokenizer(title_context)].asnumpy().mean(axis=0)
            release_years[i] = float(year)
            self.movie_features = np.concatenate((title_embedding, (release_years - 1950.0) / 100.0), axis=1)


    def compute_support(self, adj, num_links, symmetric):
        adj_unnormalized_l = []
        adj_train_int = sp.csr_matrix(adj, dtype=np.int32)
        for i in range(num_links):
            # build individual binary rating matrices (supports) for each rating
            adj_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)
            adj_unnormalized_l.append(adj_unnormalized)

        # degree_u and degree_v are row and column sums of adj+I
        adj_tot = np.sum(adj for adj in adj_unnormalized_l)  ## it is just the original training adj
        degree_u = np.asarray(adj_tot.sum(1)).flatten()
        degree_v = np.asarray(adj_tot.sum(0)).flatten()
        # set zeros to inf to avoid dividing by zero
        degree_u[degree_u == 0.] = np.inf
        degree_v[degree_v == 0.] = np.inf

        degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
        degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
        degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
        degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

        degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

        if symmetric:
            support_l = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adj_unnormalized_l]

        else:
            support_l = [degree_u_inv.dot(adj) for adj in adj_unnormalized_l]

        return support_l





if __name__ == '__main__':
    MovieLens("ml-100k")
    # MovieLens("ml-1m")
    # MovieLens("ml-10m")