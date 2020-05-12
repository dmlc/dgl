"""MovieLens dataset"""
import os
import re
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import multiprocessing as mp
import torch as th
from tqdm import tqdm
import networkx as nx
import scipy.sparse as ssp

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir

_urls = {
    'ml-100k' : 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m' : 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-10m' : 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
}

READ_DATASET_PATH = get_download_dir()
GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

class MovieLens(object):
    """MovieLens dataset used by GCMC model

    TODO(minjie): make this dataset more general

    The dataset stores MovieLens ratings in two types of graphs. The encoder graph
    contains rating value information in the form of edge types. The decoder graph
    stores plain user-movie pairs in the form of a bipartite graph with no rating
    information. All graphs have two types of nodes: "user" and "movie".

    The training, validation and test set can be summarized as follows:

    training_enc_graph : training user-movie pairs + rating info
    training_dec_graph : training user-movie pairs
    valid_enc_graph : training user-movie pairs + rating info
    valid_dec_graph : validation user-movie pairs
    test_enc_graph : training user-movie pairs + validation user-movie pairs + rating info
    test_dec_graph : test user-movie pairs

    Attributes
    ----------
    train_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for training.
    train_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for training.
    train_labels : torch.Tensor
        The categorical label of each user-movie pair
    train_truths : torch.Tensor
        The actual rating values of each user-movie pair
    valid_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for validation.
    valid_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for validation.
    valid_labels : torch.Tensor
        The categorical label of each user-movie pair
    valid_truths : torch.Tensor
        The actual rating values of each user-movie pair
    test_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for test.
    test_dec_graph : dgl.DGLHeteroGraph
        Decoder graph for test.
    test_labels : torch.Tensor
        The categorical label of each user-movie pair
    test_truths : torch.Tensor
        The actual rating values of each user-movie pair
    user_feature : torch.Tensor
        User feature tensor. If None, representing an identity matrix.
    movie_feature : torch.Tensor
        Movie feature tensor. If None, representing an identity matrix.
    possible_rating_values : np.ndarray
        Available rating values in the dataset

    Parameters
    ----------
    name : str
        Dataset name. Could be "ml-100k", "ml-1m", "ml-10m"
    device : torch.device
        Device context
    mix_cpu_gpu : bool, optional
        If true, the ``user_feature`` attribute is stored in CPU
    use_one_hot_fea : bool, optional
        If true, the ``user_feature`` attribute is None, representing an one-hot identity
        matrix. (Default: False)
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    test_ratio : float, optional
        Ratio of test data
    valid_ratio : float, optional
        Ratio of validation data

    """
    def __init__(self, name, device, args, mix_cpu_gpu=False,
                 use_one_hot_fea=False, symm=True,
                 test_ratio=0.1, valid_ratio=0.1):
        self._name = name
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        # download and extract
        download_dir = get_download_dir()
        zip_file_path = '{}/{}.zip'.format(download_dir, name)
        download(_urls[name], path=zip_file_path)
        extract_archive(zip_file_path, '{}/{}'.format(download_dir, name))
        if name == 'ml-10m':
            root_folder = 'ml-10M100K'
        else:
            root_folder = name
        self._dir = os.path.join(download_dir, name, root_folder)
        print("Starting processing {} ...".format(self._name))
        self._load_raw_user_info()
        self._load_raw_movie_info()
        print('......')
        if self._name == 'ml-100k':
            self.all_train_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.base'), '\t')
            self.test_rating_info = self._load_raw_rates(os.path.join(self._dir, 'u1.test'), '\t')
            self.all_rating_info = pd.concat([self.all_train_rating_info, self.test_rating_info])
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
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
        if args.train_val:
            self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[:]]
        else:
            self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
        self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)

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
        if use_one_hot_fea:
            self.user_feature = None
            self.movie_feature = None
        else:
            # if mix_cpu_gpu, we put features in CPU
            if mix_cpu_gpu:
                self.user_feature = th.FloatTensor(self._process_user_fea())
                self.movie_feature = th.FloatTensor(self._process_movie_fea())
            else:
                self.user_feature = th.FloatTensor(self._process_user_fea()).to(self._device)
                self.movie_feature = th.FloatTensor(self._process_movie_fea()).to(self._device)
        if self.user_feature is None:
            self.user_feature_shape = (self.num_user, self.num_user)
            self.movie_feature_shape = (self.num_movie, self.num_movie)
        else:
            self.user_feature_shape = self.user_feature.shape
            self.movie_feature_shape = self.movie_feature.shape
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(self.train_rating_info)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(self.test_rating_info)

        # Create adjacent matrix
        # TODO: we do not +1 and do not create rating map since no need for ml dataset
        # but might needed for others
        neutral_rating = 0
        rating_mx_train = np.full((self._num_user, self._num_movie), neutral_rating, dtype=np.int32)
        assert 0 not in train_rating_values, '0 should not be a valid rating type'
        rating_mx_train[train_rating_pairs] = train_rating_values 
        self.rating_mx_train = sp.csr_matrix(rating_mx_train)
        # Create subgraphs
        # adj_train is rating matrix
        # train_indices is our train_raing_pairs
        # train_labels is our train_rating_values
        # feature is currently None since we don't use side information
        self.class_values = np.sort(np.unique(train_rating_values))

        self.train_graphs, self.val_graphs, self.test_graphs = links2subgraphs(
                rating_mx_train,
                train_rating_pairs,
                valid_rating_pairs,
                test_rating_pairs,
                train_rating_values,
                valid_rating_values,
                test_rating_values,
                self.class_values, 
                args.hop,
                args.sample_ratio,
                args.max_nodes_per_hop,
                None,
                None,
                args.hop*2+1,
                parallel=True)

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie

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
            self.user_info = pd.read_csv(os.path.join(self._dir, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        elif self._name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(self._dir, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        elif self._name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(self._dir, 'ratings.dat'), sep='::', header=None,
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
            user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                            gender.reshape((self.user_info.shape[0], 1)),
                                            occupation_one_hot], axis=1)
        elif self._name == 'ml-10m':
            user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)
        else:
            raise NotImplementedError
        return user_features

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
            file_path = os.path.join(self._dir, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
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

        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
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


def links2subgraphs(
        A,
        train_indices, 
        val_indices, 
        test_indices, 
        train_labels, 
        val_labels, 
        test_labels, 
        class_values, 
        h=1, 
        sample_ratio=1.0, 
        max_nodes_per_hop=None, 
        u_features=None, 
        v_features=None, 
        max_node_label=None, 
        parallel=True):
    # extract enclosing subgraphs
    if max_node_label is None:  # if not provided, infer from graphs
        max_n_label = {'max_node_label': 0}

    def helper(A, links, g_labels):
        g_list = []
        if not parallel or max_node_label is None:
            with tqdm(total=len(links[0])) as pbar:
                for i, j, g_label in zip(links[0], links[1], g_labels):
                    g = subgraph_extraction_labeling(g_label, (i, j), A, h, max_node_label, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values)
                    g_list.append(g) 
                    pbar.update(1)
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(parallel_worker, [(g_label, (i, j), A, h, max_node_label, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values) for i, j, g_label in zip(links[0], links[1], g_labels)])
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            g_list += results.get()
            pool.close()
            pbar.close()
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_indices, train_labels)
    val_graphs = helper(A, val_indices, val_labels)
    test_graphs = helper(A, test_indices, test_labels)
    return train_graphs, val_graphs, test_graphs


def subgraph_extraction_labeling(g_label, ind, A, h=1, max_node_label=3, sample_ratio=1.0, max_nodes_per_hop=None, u_features=None, v_features=None, class_values=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    u_nodes, v_nodes = [ind[0]], [ind[1]]
    u_dist, v_dist = [0], [0]
    u_visited, v_visited = set([ind[0]]), set([ind[1]])
    u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    for dist in range(1, h+1):
        v_fringe, u_fringe = neighbors(u_fringe, A, True), neighbors(v_fringe, A, False)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)

    subgraph = A[u_nodes, :][:, v_nodes]

    # remove link between target nodes
    subgraph[0, 0] = 0
    u, v, r = ssp.find(subgraph)
    v += len(u_nodes)
    r = r.astype(int)
    
    # Constructing DGL graph
    # v nodes start after u
    rating_graphs = []
    num_user, num_movie = A.shape
    
    # Add bidirection link
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    # NOTE: RelGraphConv count relation from 0??
    bi_r = np.concatenate([r, r]) - 1 
    subgraph_info = {'g_label': g_label, 'src': src, 'dst': dst, 'etype': bi_r}

    # get structural node labels
    # NOTE: only use subgraph here
    u_node_labels = [x*2 for x in u_dist]
    v_node_labels = [x*2+1 for x in v_dist]
    u_x = one_hot(u_node_labels, max_node_label+1)
    v_x = one_hot(v_node_labels, max_node_label+1)
 
    subgraph_info['x'] = np.concatenate([u_x, v_x], axis=0)
    return subgraph_info
 
    '''
    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]
    node_features = None
    # Let's first try without node features
    if False: 
        # directly use padded node features
        if u_features is not None and v_features is not None:
            u_extended = np.concatenate([u_features, np.zeros([u_features.shape[0], v_features.shape[1]])], 1)
            v_extended = np.concatenate([np.zeros([v_features.shape[0], u_features.shape[1]]), v_features], 1)
            node_features = np.concatenate([u_extended, v_extended], 0)
    if False:
        # use identity features (one-hot encodings of node idxes)
        u_ids = one_hot(u_nodes, A.shape[0]+A.shape[1])
        v_ids = one_hot([x+A.shape[0] for x in v_nodes], A.shape[0]+A.shape[1])
        node_ids = np.concatenate([u_ids, v_ids], 0)
        #node_features = np.concatenate([node_features, node_ids], 1)
        node_features = node_ids
    if False:
        # only output node features for the target user and item
        if u_features is not None and v_features is not None:
            node_features = [u_features[0], v_features[0]]
    '''
    #graph.nodes['user'].data['x'] = u_x
    #graph.nodes['movie'].data['x'] = v_x



def parallel_worker(g_label, ind, A, h=1, max_node_label=3, sample_ratio=1.0, max_nodes_per_hop=None, u_features=None, v_features=None, class_values=None):
    g = subgraph_extraction_labeling(g_label, ind, A, h, max_node_label, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values)
    return g


def neighbors(fringe, A, row=True):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        if row:
            _, nei, _ = ssp.find(A[node, :])
        else:
            _, nei, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


if __name__ == '__main__':
    MovieLens("ml-100k", device=th.device('cpu'), symm=True, use_one_hot_fea=True)
