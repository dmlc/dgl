"""MovieLens dataset"""
import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp

from .dgl_dataset import DGLBuiltinDataset
from .utils import download, extract_archive, get_download_dir
from .utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url
from .utils import generate_mask_tensor
from .utils import deprecate_property, deprecate_function
from ..utils import retry_method_with_fix
from ..convert import bipartite, hetero_from_relations

from .. import backend as F

GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

class MovieLensDataset(DGLBuiltinDataset):
    r"""MovieLens dataset

    The dataset stores MovieLens ratings in two types 
    of graphs. The encoder graph contains rating value 
    information in the form of edge types. The decoder 
    graph stores plain user-movie pairs in the form of 
    a bipartite graph with no rating information. All 
    graphs have two types of nodes: "user" and "movie".

    Parameters
    ----------
    name: str
        name can be 'ml-100k', 'ml-1m', 'ml-10m'.
    node_feat: bool, optional
        Whether to load node features for training. 
        If no node feature provied, use one-hot node 
        label instead. Default: False
    reverse: bool, optional
        Whether to add reverse edge for each edge.
        Default: True
    test_ratio: float
        percentage of edges used as test set.
    valid_ratio: float
        percentage of edges used as validation set.
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    _urls = {
        'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'ml-1m' : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-10m' : 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
    }

    def __init__(self, name, node_feat=False, reverse=True, test_ratio=0.1, valid_ratio=0.1, symm=True,
        raw_dir=None, force_reload=False, verbose=True):
        assert name.lower() in ['ml-100k', 'ml-1m', 'ml-10m']
        self._node_feat = node_feat
        self._reverse = reverse
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        self._symm = symm

        url = self._urls[name]
        super(MovieLensDataset, self).__init__(name,
                                               url=url,
                                               raw_dir=raw_dir,
                                               force_reload=force_reload,
                                               verbose=verbose)
    def process(self, path):
        if self.name == 'ml-10m':
            root_folder = 'ml-10M100K'
        else:
            root_folder = self.name
        root_path = os.path.join(self.raw_dir, self.name, root_folder)
        
        if self.verbose:
            print("Starting processing {} ...".format(self.name))

        self._load_raw_user_info(root_path)
        self._load_raw_movie_info(root_path)

        if self.verbose:
            print('......')

        if self.name == 'ml-100k':
            all_train_rating_info = self._load_raw_rates(os.path.join(root_path, 'u1.base'), '\t')
            test_rating_info = self._load_raw_rates(os.path.join(root_path, 'u1.test'), '\t')
            all_rating_info = pd.concat([all_train_rating_info, test_rating_info])
        elif self.name == 'ml-1m' or self._name == 'ml-10m':
            all_rating_info = self._load_raw_rates(os.path.join(root_path, 'ratings.dat'), '::')
            num_test = int(np.ceil(all_rating_info.shape[0] * self._test_ratio))
            shuffled_idx = np.random.permutation(all_rating_info.shape[0])
            test_rating_info = all_rating_info.iloc[shuffled_idx[: num_test]]
            all_train_rating_info = all_rating_info.iloc[shuffled_idx[num_test: ]]
        else:
            raise NotImplementedError

        num_valid = int(np.ceil(all_train_rating_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(all_train_rating_info.shape[0])
        valid_rating_info = all_train_rating_info.iloc[shuffled_idx[: num_valid]]
        train_rating_info = all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
        possible_rating_values = np.unique(train_rating_info["rating"].values)
        self._possible_rating_values = possible_rating_values

        if self.verbose:
            print("All rating pairs : {}".format(all_rating_info.shape[0]))
            print("\tAll train rating pairs : {}".format(all_train_rating_info.shape[0]))
            print("\t\tTrain rating pairs : {}".format(train_rating_info.shape[0]))
            print("\t\tValid rating pairs : {}".format(valid_rating_info.shape[0]))
            print("\tTest rating pairs  : {}".format(test_rating_info.shape[0]))

        self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                                 cmp_col_name="id",
                                                 reserved_ids_set=set(all_rating_info["user_id"].values),
                                                 label="user")
        self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                                  cmp_col_name="id",
                                                  reserved_ids_set=set(all_rating_info["movie_id"].values),
                                                  label="movie")

        # Map user/movie to the global id
        self.global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        if self.verbose:
            print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                    len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)

        ### Generate features
        if self._node_feat is False:
            self._user_feature = None
            self._movie_feature = None
        else:
            self._user_feature = F.tensor(self._process_user_fea(), dtype=F.data_type_dict['float32'])
            self._movie_feature = F.tensor(self._process_movie_fea(), dtype=F.data_type_dict['float32'])
        
        if self.user_feature is None:
            self._user_feature_shape = (self.num_user, self.num_user)
            self._movie_feature_shape = (self.num_movie, self.num_movie)
        else:
            self._user_feature_shape = self.user_feature.shape
            self._movie_feature_shape = self.movie_feature.shape

        if self.verbose:
            info_line = "Feature dim: "
            info_line += "\nuser: {}".format(self.user_feature_shape)
            info_line += "\nmovie: {}".format(self.movie_feature_shape)
            print(info_line)

        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(train_rating_info)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(test_rating_info)

        def _make_labels(ratings):
            labels = F.tensor(np.searchsorted(possible_rating_values, ratings), dtype=F.data_type_dict['int64'])
            return labels

        self._train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values)
        self._train_dec_graph = self._generate_dec_graph(train_rating_pairs)
        self._train_labels = _make_labels(train_rating_values)
        self._train_truths = F.tensor(train_rating_values)

        self._valid_enc_graph = self.train_enc_graph
        self._valid_dec_graph = self._generate_dec_graph(valid_rating_pairs)
        self._valid_truths = F.tensor(valid_rating_values)

        self._test_enc_graph = self._generate_enc_graph(all_train_rating_pairs, all_train_rating_values)
        self._test_dec_graph = self._generate_dec_graph(test_rating_pairs)
        self._test_truths = F.tensor(test_rating_values)

        if self._node_feat:
            self._test_enc_graph.ndata['feat'] = {'user' : self.user_feature,
                                                  'movie' : self.movie_feature}
            print(self.test_enc_graph.nodes['user'].data['feat'])
            print(self.test_enc_graph.nodes['movie'].data['feat'])
        self._train_dec_graph.edata['label'] = self._train_labels
        self._train_dec_graph.edata['truth'] = self._train_truths
        self._valid_dec_graph.edata['truth'] = self._valid_truths
        self._test_dec_graph.edata['truth'] = self._test_truths
        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = str(r).replace('.', '_')
                rst += graph.number_of_edges(str(r))
            return rst

        if self.verbose:
            print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
                _npairs(self.train_enc_graph)))
            print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
                self.train_dec_graph.number_of_edges()))
            print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
                _npairs(self.valid_enc_graph)))
            print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
                self.valid_dec_graph.number_of_edges()))
            print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
                _npairs(self.test_enc_graph)))
            print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
                self.test_dec_graph.number_of_edges()))

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
            os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs = [self.train_enc_graph, self.train_dec_graph, self.test_enc_graph, self.test_dec_graph]
        save_graphs(str(graph_path), graphs)
        save_info(str(info_path), {'user_feature_shape': self.user_feature_shape,
                                   'movie_feature_shape': self.movie_feature_shape,
                                   'possible_rating_values': self.possible_rating_values,
                                   'has_feat': self._node_feat})

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, _ = load_graphs(str(graph_path))
        self._train_enc_graph = graphs[0]
        self._train_dec_graph = graphs[1]
        self._valid_enc_graph = self._train_enc_graph
        self._valid_dec_graph = self._train_dec_graph
        self._test_enc_graph = graphs[2]
        self._test_dec_graph = graphs[3]
        info = load_info(str(info_path))

        self._user_feature_shape = info['user_feature_shape']
        self._movie_feature_shape = info['movie_feature_shape']
        self._possible_rating_values = info['possible_rating_values']
        self._node_feat = info['has_feat']

        # for backward compatability
        self._train_labels = self._train_dec_graph.edata['label']
        self._train_truths = self._train_dec_graph.edata['truth']
        self._valid_truths = self._valid_dec_graph.edata['truth']
        self._test_truths = self._test_dec_graph.edata['truth']
        if self._node_feat:
            self._user_feature = self.test_enc_graph.nodes['user'].data['feat']
            self._movie_feature = self.test_enc_graph.nodes['movie'].data['feat']
        else:
            self._user_feature = None
            self._movie_feature = None

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = str(r).replace('.', '_')
                rst += graph.number_of_edges(str(r))
            return rst

        if self.verbose:
            print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
                _npairs(self.train_enc_graph)))
            print("Train dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.train_dec_graph.number_of_nodes('user'), self.train_dec_graph.number_of_nodes('movie'),
                self.train_dec_graph.number_of_edges()))
            print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
                _npairs(self.valid_enc_graph)))
            print("Valid dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.valid_dec_graph.number_of_nodes('user'), self.valid_dec_graph.number_of_nodes('movie'),
                self.valid_dec_graph.number_of_edges()))
            print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
                _npairs(self.test_enc_graph)))
            print("Test dec graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
                self.test_dec_graph.number_of_nodes('user'), self.test_dec_graph.number_of_nodes('movie'),
                self.test_dec_graph.number_of_edges()))
        
    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values):
        user_movie_R = np.zeros((self.num_user, self.num_movie), dtype=np.float32)
        user_movie_R[rating_pairs] = rating_values
        movie_user_R = user_movie_R.transpose()

        rating_graphs = []
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = str(rating).replace('.', '_')
            bg = bipartite((rrow, rcol), 'user', rating, 'movie',
                           num_nodes=(self.num_user, self.num_movie))
            rev_bg = bipartite((rcol, rrow), 'movie', 'rev-%s' % rating, 'user',
                               num_nodes=(self.num_movie, self.num_user))
            rating_graphs.append(bg)
            rating_graphs.append(rev_bg)
        graph = hetero_from_relations(rating_graphs)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        # calculate edge norm
        def _calc_norm(x):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = F.tensor(1. / np.sqrt(x))
            return F.unsqueeze(x, 1)
        user_ci = []
        user_cj = []
        movie_ci = []
        movie_cj = []
        for r in self.possible_rating_values:
            r = str(r).replace('.', '_')
            user_ci.append(graph['rev-%s' % r].in_degrees())
            movie_ci.append(graph[r].in_degrees())
            if self._symm:
                user_cj.append(graph[r].out_degrees())
                movie_cj.append(graph['rev-%s' % r].out_degrees())
            else:
                user_cj.append(th.zeros((self.num_user,)))
                movie_cj.append(th.zeros((self.num_movie,)))
        user_ci = _calc_norm(sum(user_ci))
        movie_ci = _calc_norm(sum(movie_ci))
        if self._symm:
            user_cj = _calc_norm(sum(user_cj))
            movie_cj = _calc_norm(sum(movie_cj))
        else:
            user_cj = th.ones(self.num_user,).to(self._device)
            movie_cj = th.ones(self.num_movie,).to(self._device)
        graph.ndata['ci'] = {'user': user_ci,
                             'movie': movie_ci}
        graph.ndata['cj'] = {'user': user_cj,
                             'movie': movie_cj}

        return graph

    def _generate_dec_graph(self, rating_pairs):
        ones = np.ones_like(rating_pairs[0])
        user_movie_ratings_coo = sp.coo_matrix(
            (ones, rating_pairs),
            shape=(self.num_user, self.num_movie), dtype=np.float32)
        return bipartite(user_movie_ratings_coo, 'user', 'rate', 'movie')

    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        # print("  -----------------")
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

    def _load_raw_user_info(self, root_path):
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
        if self.name == 'ml-100k':
            self.user_info = pd.read_csv(os.path.join(root_path, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        elif self.name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(root_path, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        elif self.name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(root_path, 'ratings.dat'), sep='::', header=None,
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
        if self.name == 'ml-100k' or self.name == 'ml-1m':
            ages = self.user_info['age'].values.astype(np.float32)
            gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
            all_occupations = set(self.user_info['occupation'])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)),
                                          dtype=np.float32)
            occupation_one_hot[np.arange(self.user_info.shape[0]),
                               np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
            user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                            gender.reshape((self.user_info.shape[0], 1)),
                                            occupation_one_hot], axis=1)
        elif self._name == 'ml-10m':
            user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)
        else:
            raise NotImplementedError
        return user_features

    def _load_raw_movie_info(self, root_path):
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
        if self.name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self.name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self.name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        if self.name == 'ml-100k':
            file_path = os.path.join(root_path, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self.name == 'ml-1m' or self.name == 'ml-10m':
            file_path = os.path.join(root_path, 'movies.dat')
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
        import torchtext

        if self.name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self.name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self.name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        TEXT = torchtext.data.Field(tokenize='spacy')
        embedding = torchtext.vocab.GloVe(name='840B', dim=300)

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
            title_embedding[i, :] = embedding.get_vecs_by_tokens(TEXT.tokenize(title_context)).numpy().mean(axis=0)
            release_years[i] = float(year)
        movie_features = np.concatenate((title_embedding,
                                         (release_years - 1950.0) / 100.0,
                                         self.movie_info[GENRES]),
                                        axis=1)
        return movie_features

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def train_enc_graph(self):
        return self._train_enc_graph

    @property
    def train_dec_graph(self):
        return self._train_dec_graph

    @property
    def valid_enc_graph(self):
        return self._valid_enc_graph

    @property
    def valid_dec_graph(self):
        return self._valid_dec_graph

    @property
    def test_enc_graph(self):
        return self._test_enc_graph

    @property
    def test_dec_graph(self):
        return self._test_dec_graph

    @property
    def possible_rating_values(self):
        return self._possible_rating_values
        
    @property
    def user_feature_shape(self):
        return self._user_feature_shape

    @property
    def movie_feature_shape(self):
        return self._movie_feature_shape

    @property
    def train_labels(self):
        deprecate_property('dataset.train_labels', 'train_dnc_graph.edata[\'label\']')
        return self._train_labels

    @property
    def train_truths(self):
        deprecate_property('dataset.train_truths', 'train_dnc_graph.edata[\'truth\']')
        return self._train_truths

    @property
    def valid_truths(self):
        deprecate_property('dataset.valid_truths', 'valid_dnc_graph.edata[\'truth\']')
        return self._valid_truths

    @property
    def test_truths(self):
        deprecate_property('dataset.test_truths', 'test_dnc_graph.edata[\'truth\']')
        return self._test_truths

    @property
    def user_feature(self):
        deprecate_property('dataset.user_feature', 'test_enc_graph.nodes[\'user\'].data[\'feat\']')
        return self._user_feature

    @property
    def movie_feature(self):
        deprecate_property('dataset.movie_feature', 'test_enc_graph.nodes[\'movie\'].data[\'feat\']')
        return self._movie_feature

class MovieLen100kDataset(MovieLensDataset):
    r""" MovieLen100k dataset.
    
    The dataset stores MovieLens ratings in two types 
    of graphs. The encoder graph contains rating value 
    information in the form of edge types. The decoder 
    graph stores plain user-movie pairs in the form of 
    a bipartite graph with no rating information. All 
    graphs have two types of nodes: "user" and "movie".
    
    Statistics
    ----------
    Nodes: xxx
    Edges: xxx
    Number of relation types: xx
    Number of reversed relation types: xx
    Label Split: Train: xxx ,Valid: xxx, Test: xxx
    
    Parameters
    ----------
    node_feat: bool, optional
        Whether to load node features for training. 
        If no node feature provied, use one-hot node 
        label instead. Default: False
    reverse: bool, optional
        Whether to add reverse edge for each edge.
        Default: True
    test_ratio: float
        percentage of edges used as test set.
    valid_ratio: float
        percentage of edges used as validation set.
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ----------
    MovieLen100kDataset object with several properties:      
        train_enc_graph: A Heterogeneous graph for training. It is
            used for encoding node embeddings of users and movies.
            It only contains the edges for training.
            - nodes['user'].data['ci']: in-edges norm
            - nodes['user'].data['cj']: out-edges norm
            - nodes['movie'].data['ci']: in-edges norm
            - nodes['movie'].data['cj']: out-edges norm
        valid_enc_graph: A Heterogeneous graph for validation,
            identical to train_enc_graph.
        test_enc_graph: A Heterogeneous graph for testing. It is
            used for encoding node embeddings of users and movies.
            It contains all the edges for both training and testing.
            - nodes['user'].data['ci']: in-edges norm
            - nodes['user'].data['cj']: out-edges norm
            - nodes['user'].data['feat']: if node_feat is True (node 
                has input features), it store features of user nodes.
            - nodes['movie'].data['ci']: in-edges norm
            - nodes['movie'].data['cj']: out-edges norm
            - nodes['movie'].data['feat']: if node_feat is True (node 
                has input features), it store features of movie nodes.
        train_dec_graph: A bipartite graph for training. It is used 
            for generating the edge labels using the node embeddings 
            generated by train_enc_graph.
            - edata['label']: label of each edge.
            - edata['truth']: the rating value of each edge.
        valid_dec_graph: A bipartite graph for validation. It is used 
            when predicting the rating values using the node embeddings 
            generated by valid_enc_graph.
            - edata['truth']: the rating value of each edge.
        test_dec_graph: A bipartite graph for testing. It is used 
            when predicting the rating values using the node embeddings 
            generated by test_enc_graph.
            - edata['truth']: the rating value of each edge.
        possible_rating_values: The label to rating 
            value mapping.
        user_feature_shape: The shape of user feature
        movie_feature_shape: The shape of movie feature
    
    Examples
    ----------
    
    >>> dataset = MovieLen100kDataset()
    >>>
    >>> train_enc = dataset.train_enc_graph
    >>> train_dec = dataset.train_dec_graph
    >>> valid_enc = dataset.valid_enc_graph
    >>> valid_dec = dataset.valid_dec_graph
    >>> test_enc = dataset.test_enc_graph
    >>> test_dec = dataset.test_dec_graph
    >>> user_feat = test_enc.nodes['user'].data['feat']
    >>> movie_feat = test_enc.nodes['movie'].data['feat']
    >>>
    >>> # Train, Validation and Test
    >>>
    """
    def __init__(self, node_feat=False, reverse=True, test_ratio=0.1, valid_ratio=0.1, symm=True,
        raw_dir=None, force_reload=False, verbose=True):

        name = 'ml-100k'
        super(MovieLen100kDataset, self).__init__(name=name,
                                                  node_feat=node_feat,
                                                  reverse=reverse,
                                                  test_ratio=test_ratio,
                                                  valid_ratio=valid_ratio,
                                                  symm=symm,
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)

class MovieLen1MDataset(MovieLensDataset):
    r""" MovieLen100k dataset.
    
    The dataset stores MovieLens ratings in two types 
    of graphs. The encoder graph contains rating value 
    information in the form of edge types. The decoder 
    graph stores plain user-movie pairs in the form of 
    a bipartite graph with no rating information. All 
    graphs have two types of nodes: "user" and "movie".
    
    Statistics
    ----------
    Nodes: xxx
    Edges: xxx
    Number of relation types: xx
    Number of reversed relation types: xx
    Label Split: Train: xxx ,Valid: xxx, Test: xxx
    
    Parameters
    ----------
    node_feat: bool, optional
        Whether to load node features for training. 
        If no node feature provied, use one-hot node 
        label instead. Default: False
    reverse: bool, optional
        Whether to add reverse edge for each edge.
        Default: True
    test_ratio: float
        percentage of edges used as test set.
    valid_ratio: float
        percentage of edges used as validation set.
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ----------
    MovieLen1MDataset object with two properties:
        train_enc_graph: A Heterogeneous graph for training. It is
            used for encoding node embeddings of users and movies.
            It only contains the edges for training.
            - nodes['user'].data['ci']: in-edges norm
            - nodes['user'].data['cj']: out-edges norm
            - nodes['movie'].data['ci']: in-edges norm
            - nodes['movie'].data['cj']: out-edges norm
        valid_enc_graph: A Heterogeneous graph for validation,
            identical to train_enc_graph.
        test_enc_graph: A Heterogeneous graph for testing. It is
            used for encoding node embeddings of users and movies.
            It contains all the edges for both training and testing.
            - nodes['user'].data['ci']: in-edges norm
            - nodes['user'].data['cj']: out-edges norm
            - nodes['user'].data['feat']: if node_feat is True (node 
                has input features), it store features of user nodes.
            - nodes['movie'].data['ci']: in-edges norm
            - nodes['movie'].data['cj']: out-edges norm
            - nodes['movie'].data['feat']: if node_feat is True (node 
                has input features), it store features of movie nodes.
        train_dec_graph: A bipartite graph for training. It is used 
            for generating the edge labels using the node embeddings 
            generated by train_enc_graph.
            - edata['label']: label of each edge.
            - edata['truth']: the rating value of each edge.
        valid_dec_graph: A bipartite graph for validation. It is used 
            when predicting the rating values using the node embeddings 
            generated by valid_enc_graph.
            - edata['truth']: the rating value of each edge.
        test_dec_graph: A bipartite graph for testing. It is used 
            when predicting the rating values using the node embeddings 
            generated by test_enc_graph.
            - edata['truth']: the rating value of each edge.
        possible_rating_values: The label to rating 
            value mapping.
        user_feature_shape: The shape of user feature
        movie_feature_shape: The shape of movie feature
    
    Examples
    ----------
    
    >>> dataset = MovieLen1MDataset()
    >>>
    >>> train_enc = dataset.train_enc_graph
    >>> train_dec = dataset.train_dec_graph
    >>> valid_enc = dataset.valid_enc_graph
    >>> valid_dec = dataset.valid_dec_graph
    >>> test_enc = dataset.test_enc_graph
    >>> test_dec = dataset.test_dec_graph
    >>> user_feat = test_enc.nodes['user'].data['feat']
    >>> movie_feat = test_enc.nodes['movie'].data['feat']
    >>>
    >>> # Train, Validation and Test
    >>>

    """
    def __init__(self, node_feat=False, reverse=True, test_ratio=0.1, valid_ratio=0.1, symm=True,
        raw_dir=None, force_reload=False, verbose=True):

        name = 'ml-1m'
        super(MovieLen1MDataset, self).__init__(name=name,
                                                  node_feat=node_feat,
                                                  reverse=reverse,
                                                  test_ratio=test_ratio,
                                                  valid_ratio=valid_ratio,
                                                  symm=symm,
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)

class MovieLen10MDataset(MovieLensDataset):
    r""" MovieLen10MDataset dataset.
    
    The dataset stores MovieLens ratings in two types 
    of graphs. The encoder graph contains rating value 
    information in the form of edge types. The decoder 
    graph stores plain user-movie pairs in the form of 
    a bipartite graph with no rating information. All 
    graphs have two types of nodes: "user" and "movie".
    
    Statistics
    ----------
    Nodes: xxx
    Edges: xxx
    Number of relation types: xx
    Number of reversed relation types: xx
    Label Split: Train: xxx ,Valid: xxx, Test: xxx
    
    Parameters
    ----------
    node_feat: bool, optional
        Whether to load node features for training. 
        If no node feature provied, use one-hot node 
        label instead. Default: False
    reverse: bool, optional
        Whether to add reverse edge for each edge.
        Default: True
    test_ratio: float
        percentage of edges used as test set.
    valid_ratio: float
        percentage of edges used as validation set.
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ----------
    MovieLen10MDataset object with two properties:      
        train_enc_graph: A Heterogeneous graph for training. It is
            used for encoding node embeddings of users and movies.
            It only contains the edges for training.
            - nodes['user'].data['ci']: in-edges norm
            - nodes['user'].data['cj']: out-edges norm
            - nodes['movie'].data['ci']: in-edges norm
            - nodes['movie'].data['cj']: out-edges norm
        valid_enc_graph: A Heterogeneous graph for validation,
            identical to train_enc_graph.
        test_enc_graph: A Heterogeneous graph for testing. It is
            used for encoding node embeddings of users and movies.
            It contains all the edges for both training and testing.
            - nodes['user'].data['ci']: in-edges norm
            - nodes['user'].data['cj']: out-edges norm
            - nodes['user'].data['feat']: if node_feat is True (node 
                has input features), it store features of user nodes.
            - nodes['movie'].data['ci']: in-edges norm
            - nodes['movie'].data['cj']: out-edges norm
            - nodes['movie'].data['feat']: if node_feat is True (node 
                has input features), it store features of movie nodes.
        train_dec_graph: A bipartite graph for training. It is used 
            for generating the edge labels using the node embeddings 
            generated by train_enc_graph.
            - edata['label']: label of each edge.
            - edata['truth']: the rating value of each edge.
        valid_dec_graph: A bipartite graph for validation. It is used 
            when predicting the rating values using the node embeddings 
            generated by valid_enc_graph.
            - edata['truth']: the rating value of each edge.
        test_dec_graph: A bipartite graph for testing. It is used 
            when predicting the rating values using the node embeddings 
            generated by test_enc_graph.
            - edata['truth']: the rating value of each edge.
        possible_rating_values: The label to rating 
            value mapping.
        user_feature_shape: The shape of user feature
        movie_feature_shape: The shape of movie feature
    
    Examples
    ----------
    
    >>> dataset = MovieLen10MDataset()
    >>>
    >>> train_enc = dataset.train_enc_graph
    >>> train_dec = dataset.train_dec_graph
    >>> valid_enc = dataset.valid_enc_graph
    >>> valid_dec = dataset.valid_dec_graph
    >>> test_enc = dataset.test_enc_graph
    >>> test_dec = dataset.test_dec_graph
    >>> user_feat = test_enc.nodes['user'].data['feat']
    >>> movie_feat = test_enc.nodes['movie'].data['feat']
    >>>
    >>> # Train, Validation and Test
    >>>

    """
    def __init__(self, node_feat=False, reverse=True, test_ratio=0.1, valid_ratio=0.1, symm=True,
        raw_dir=None, force_reload=False, verbose=True):

        name = 'ml-10m'
        super(MovieLen10MDataset, self).__init__(name=name,
                                                  node_feat=node_feat,
                                                  reverse=reverse,
                                                  test_ratio=test_ratio,
                                                  valid_ratio=valid_ratio,
                                                  symm=symm,
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)
