"""MovieLens dataset"""
import os
import zipfile
import shutil

import numpy as np
import pandas as pd


from torch import LongTensor, Tensor
from ..convert import graph
from ..transforms.functional import add_reverse_edges
from .dgl_dataset import DGLDataset
from ..base import dgl_warning

from .utils import (
    download,
    load_graphs,
    load_info,
    save_graphs,
    save_info,
    split_dataset,
)

GENRES_ML_100K = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ["IMAX"]

try:
    import torch
except ImportError:
    HAS_TORCH = False
else:
    HAS_TORCH = True

def check_pytorch():
    """Check if PyTorch is the backend."""
    if not HAS_TORCH:
        raise ModuleNotFoundError(
            "MovieLensDataset requires PyTorch to be the backend."
        )

class MovieLensDataset(DGLDataset):
    r"""MovieLens dataset for edge prediction tasks. The raw datasets are extracted from 
    `MovieLens <https://grouplens.org/datasets/movielens/>`, introduced by
    `Movielens unplugged: experiences with an occasionally connected recommender system <https://dl.acm.org/doi/10.1145/604045.604094>`.

    The datasets consist of user ratings for movies and incorporate additional user/movie information in the form of features. 
    The nodes represent users and movies, and the edges store ratings that users assign to movies.

    Statistics:

    MovieLens-100K (ml-100k)

    - Users: 943
    - Movies: 1,682
    - Ratings: 100,000 (1, 2, 3, 4, 5)

    MovieLens-1M (ml-1m)

    - Users: 6,040
    - Movies: 3,706
    - Ratings: 1,000,209 (1, 2, 3, 4, 5)

    MovieLens-10M (ml-10m)

    - Users: 69,878 
    - Movies: 10,677
    - Ratings: 10,000,054 (0.5, 1, 1.5, ..., 4.5, 5.0)

    Parameters
    ----------
    name: str
        Dataset name. (:obj:`"ml-100k"`, :obj:`"ml-1m"`, :obj:`"ml-10m"`). 
    valid_ratio: int
        Ratio of validation samples out of the whole dataset. Should be in (0.0, 1.0). 
    test_ratio: int, optional
        Ratio of testing samples out of the whole dataset. Should be in (0.0, 1.0) for :obj:`"ml-1m"` and :obj:`"ml-10m"`. This parameter is invalid
        when :obj:`name` is :obj:`"ml-100k"`, since its testing samples are pre-specified. 
        Default: None
    raw_dir : str, optional
        Raw file directory to download/store the data.
        Default: ~/.dgl/
    force_reload : bool, optional
        Whether to re-download(if the dataset has not been downloaded) and re-process the dataset. 
        Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    random_state : int, optional
        Random seed used for random dataset split. Default: 0

    Notes
    -----
    - Each time when the dataset is loaded with a different setting of training, validation and test ratio, the parameter
    :obj:`force_reload` should be set to :obj:`True`. Otherwise the previous split of the dataset will be loaded and the new split will not come into effect.
    updated in the storage.
    - When test_ratio is valid, that is, the dataset is :obj:`"ml-1m"` or :obj:`"ml-10m"`, 
    the sum of valid_ratio and test_ratio must be in (0.0, 1.0).
    - The number of edges is doubled to form an undirected(bidirected) graph structure.

    Examples
    --------
    >>> from dgl.data import MovieLensDataset
    >>> dataset = MovieLensDataset(name='ml-100k', valid_ratio=0.2)
    >>> train_g, valid_g, test_g, info = dataset[0]
    >>> train_g
    Graph(num_nodes=2625, num_edges=128000,
        ndata_schemes={}
        edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), 'etype': Scheme(shape=(), dtype=torch.float32)})
    >>> # get ratings of edges in the training graph
    >>> ratings = train_g.edata['etype']
    >>> ratings
    tensor([3., 3., 2.,  ..., 4., 4., 2.])
    >>> # get training, validation and testing rating pairs
    >>> train_rating, valid_rating, test_rating = \
    ...     info['train_rating_pairs'], info['valid_rating_pairs'], info['test_rating_pairs']
    >>> train_rating[0] # node index of users in training rating pairs
    tensor([614, 772, 531, ..., 674, 639, 740])
    >>> train_rating[1] # node index of movies in training rating pairs
    tensor([1236, 954, 1487, ..., 1842, 1631, 1168])
    >>> # get the rating of a certain user-movie rating pair
    >>> u, m = train_rating[0][0], train_rating[1][0]
    >>> eid = train_g.edge_ids(u, m)
    >>> rating = train_g.edata['etype'][eid]
    >>> rating
    tensor([3.])
    >>> # get input features of users and movies respectively
    >>> user_feat, movie_feat = info['user_feat'], info['movie_feat']
    >>> user_feat
    tensor([[0.4800, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [1.0600, 1.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.4600, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            ...,
            [0.4000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.9600, 1.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.4400, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]])

    """

    _url = {
        "ml-100k": "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ml-1m": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-10m": "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
    }

    def __init__(
        self,
        name,
        valid_ratio,
        test_ratio=None,
        raw_dir=None,
        force_reload=None,
        verbose=None,
        transform=None,
        random_state=0
    ):
        check_pytorch()
        assert name in [
            "ml-100k",
            "ml-1m",
            "ml-10m",
        ], f"currently movielens does not support {name}"

        # test regarding valid and test split ratio
        assert valid_ratio > 0.0 and valid_ratio < 1.0, f"valid_ratio {valid_ratio} must be in (0.0, 1.0)"

        if name in ['ml-1m', 'ml-10m']:
            assert test_ratio is not None and test_ratio > 0.0 and test_ratio < 1.0, \
                f"test_ratio {test_ratio} must be set to (0.0, 1.0) when using ml-1m and ml-10m"
            assert test_ratio + valid_ratio > 0.0 and test_ratio + valid_ratio < 1.0, \
                f"test_ratio {test_ratio} + valid_ratio {valid_ratio} must be set to (0.0, 1.0) when using ml-1m and ml-10m"

        if name == 'ml-100k' and test_ratio is not None:
            dgl_warning(f"test_ratio {test_ratio} is not set to None for ml-100k, "
                           "while testing samples would not be affected by test_ratio since "
                           "testing samples of ml-100k have been pre-specified.")

        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        if name == "ml-100k":
            self.genres = GENRES_ML_100K
        elif name == "ml-1m":
            self.genres = GENRES_ML_1M
        elif name == "ml-10m":
            self.genres = GENRES_ML_10M
        else:
            raise NotImplementedError

        super(MovieLensDataset, self).__init__(
            name=name,
            url=self._url[name],
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        zip_file_path = os.path.join(self.raw_dir, self.name + ".zip")
        download(self.url, path=zip_file_path)

        with zipfile.ZipFile(zip_file_path, "r") as archive:
            archive.extractall(path=self.raw_dir)

        '''
        TODO: Here are codes downloading movie features from dgl s3. 
        Right now codes are used for testing only.
        '''
        dir = os.path.join(self.raw_dir, "movie_feat")
        shutil.copyfile(os.path.join(dir, self.name + '_movie_feat.pkl'), 
                        os.path.join(self.raw_path, self.name + '_movie_feat.pkl'))

    def process(self):
        print(f"Starting processing {self.name} ...")

        # 0. loading movie features
        movie_feat = load_info(os.path.join(self.raw_path, self.name + '_movie_feat.pkl'))

        # 1. dataset split: train + (valid + ) test
        if self.name == "ml-100k":
            train_rating_data = self._load_raw_rates(
                os.path.join(self.raw_path, "u1.base"), "\t"
            )
            test_rating_data = self._load_raw_rates(
                os.path.join(self.raw_path, "u1.test"), "\t"
            )
            indices = np.arange(len(train_rating_data))
            train, valid, _ = split_dataset(
                indices,
                [1 - self.valid_ratio, self.valid_ratio, 0.0],
                shuffle=True, random_state=self.random_state
            )
            train_rating_data, valid_rating_data = (
                train_rating_data.iloc[train.indices],
                train_rating_data.iloc[valid.indices],
            )
            all_rating_data = pd.concat(
                [train_rating_data, valid_rating_data, test_rating_data]
            )

        elif self.name == "ml-1m" or self.name == "ml-10m":
            all_rating_data = self._load_raw_rates(
                os.path.join(self.raw_path, "ratings.dat"), "::"
            )
            indices = np.arange(len(all_rating_data))
            train, valid, test = split_dataset(
                indices,
                [
                    1 - self.valid_ratio - self.test_ratio,
                    self.valid_ratio,
                    self.test_ratio,
                ],
                shuffle=True, random_state=self.random_state
            )
            train_rating_data, valid_rating_data, test_rating_data = (
                all_rating_data.iloc[train.indices],
                all_rating_data.iloc[valid.indices],
                all_rating_data.iloc[test.indices],
            )

        # 2. load user and movie data, and drop those unseen in rating_data
        user_data = self._load_raw_user_data()
        movie_data = self._load_raw_movie_data()
        user_data = self._drop_unseen_nodes(
            data_df=user_data,
            col_name="id",
            reserved_ids_set=set(all_rating_data["user_id"].values),
        )
        movie_data = self._drop_unseen_nodes(
            data_df=movie_data,
            col_name="id",
            reserved_ids_set=set(all_rating_data["movie_id"].values),
        )

        user_feat = Tensor(self._process_user_feat(user_data)) 

        # 3. generate rating pairs
        # Map user/movie to the global id
        self._global_user_id_map = {
            ele: i for i, ele in enumerate(user_data["id"])
        }
        self._global_movie_id_map = {
            ele: i for i, ele in enumerate(movie_data["id"])
        }

        # pair value is idx rather than id
        (
            train_u_indices,
            train_v_indices,
            train_labels,
        ) = self._generate_pair_value(train_rating_data)
        (
            valid_u_indices,
            valid_v_indices,
            valid_labels,
        ) = self._generate_pair_value(valid_rating_data)
        test_u_indices, test_v_indices, test_labels = self._generate_pair_value(
            test_rating_data
        )

        # reindex u and v, v nodes start after u
        num_user = len(self._global_user_id_map)
        train_v_indices += num_user
        valid_v_indices += num_user
        test_v_indices += num_user

        self.train_rating_pairs = (
            LongTensor(train_u_indices),
            LongTensor(train_v_indices),
        )
        self.valid_rating_pairs = (
            LongTensor(valid_u_indices),
            LongTensor(valid_v_indices),
        )
        self.test_rating_pairs = (
            LongTensor(test_u_indices),
            LongTensor(test_v_indices),
        )
        self.train_rating_values = Tensor(train_labels)
        self.valid_rating_values = Tensor(valid_labels)
        self.test_rating_values = Tensor(test_labels)
        self.info = {
            "train_rating_pairs": self.train_rating_pairs,
            "valid_rating_pairs": self.valid_rating_pairs,
            "test_rating_pairs": self.test_rating_pairs,
            "user_feat": user_feat,
            "movie_feat": movie_feat
        }

        # build dgl graph object, which is homogeneous and bidirectional and contains only training edges
        self.train_graph = graph(
            (self.train_rating_pairs[0], self.train_rating_pairs[1])
        )
        self.valid_graph = graph(
            (self.valid_rating_pairs[0], self.valid_rating_pairs[1])
        )
        self.test_graph = graph(
            (self.test_rating_pairs[0], self.test_rating_pairs[1])
        )
        self.train_graph.edata["etype"] = self.train_rating_values
        self.valid_graph.edata["etype"] = self.valid_rating_values
        self.test_graph.edata["etype"] = self.test_rating_values

        self.train_graph = add_reverse_edges(self.train_graph, copy_edata=True)
        self.valid_graph = add_reverse_edges(self.valid_graph, copy_edata=True)
        self.test_graph = add_reverse_edges(self.test_graph, copy_edata=True)

        self.train_graph.edata.pop('_ID')
        self.valid_graph.edata.pop('_ID')
        self.test_graph.edata.pop('_ID')

    def has_cache(self):
        if os.path.exists(self.graph_path) and os.path.exists(self.info_path):
            return True
        return False

    def save(self):
        save_graphs(
            self.graph_path,
            [self.train_graph, self.valid_graph, self.test_graph]
        )
        save_info(self.info_path, self.info)
        if self.verbose:
            print(f"Done saving data into {self.raw_path}.")

    def load(self):
        g_list, self.info = load_graphs(self.graph_path)
        self.train_graph, self.valid_graph, self.test_graph = (
            g_list[0], g_list[1], g_list[2]
        )
        self.info = load_info(self.info_path)
        if self.verbose:
            print(f"Done loading data from {self.raw_path}.")

            print(
                "All rating pairs : {}".format(
                    int(
                        (
                            self.train_graph.num_edges()
                            + self.valid_graph.num_edges()
                            + self.test_graph.num_edges()
                        )
                        / 2
                    )
                )
            )
            print(
                "\tTrain rating pairs : {}".format(
                    int(self.train_graph.num_edges() / 2)
                )
            )
            print(
                "\tValid rating pairs : {}".format(
                    int(self.valid_graph.num_edges() / 2)
                )
            )
            print(
                "\tTest rating pairs  : {}".format(
                    int(self.test_graph.num_edges() / 2)
                )
            )

    def __getitem__(self, idx):
        assert (
            idx == 0
        ), "This dataset has only one set of training, validation and testing graph"
        if self._transform is None:
            return self.train_graph, self.valid_graph, self.test_graph, self.info
        else:
            return (
                self._transform(self.train_graph),
                self._transform(self.valid_graph),
                self._transform(self.test_graph),
            ), self.info

    def __len__(self):
        return 1

    @property
    def raw_path(self):
        if self.name == "ml-10m":
            return os.path.join(self.raw_dir, "ml-10M100K")
        return os.path.join(self.raw_dir, self.name)

    @property
    def graph_path(self):
        return os.path.join(self.raw_path, self.name + ".bin")
    
    @property
    def info_path(self):
        return os.path.join(self.raw_path, self.name + ".pkl")

    def _process_user_feat(self, user_data):
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
                    [occupation_map[ele] for ele in user_data["occupation"]]
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

    def _load_raw_user_data(self):
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
        file_path = os.path.join(self.raw_path, "u.item")
        if self.name == "ml-100k":
            movie_data = pd.read_csv(
                file_path,
                sep="|",
                header=None,
                names=[
                    "id",
                    "title",
                    "release_date",
                    "video_release_date",
                    "url",
                ]
                + GENRES_ML_100K,
                engine="python",
                encoding="ISO-8859-1",
            )
        elif self.name == "ml-1m" or self.name == "ml-10m":
            file_path = os.path.join(self.raw_path, "movies.dat")
            movie_data = pd.read_csv(
                file_path,
                sep="::",
                header=None,
                names=["id", "title", "genres"],
                encoding="iso-8859-1",
                engine="python",
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
                movie_data[genre_name] = movie_genres[:, idx]
            movie_data = movie_data.drop(columns=["genres"])
        else:
            raise NotImplementedError

        return movie_data

    def _load_raw_rates(self, file_path, sep):
        rating_data = pd.read_csv(
            file_path,
            sep=sep,
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
        rating_data = rating_data.reset_index(drop=True)
        return rating_data

    def _drop_unseen_nodes(self, data_df, col_name, reserved_ids_set):
        data_df = data_df[data_df[col_name].isin(reserved_ids_set)]
        data_df.reset_index(drop=True, inplace=True)
        return data_df

    def _generate_pair_value(self, rating_data):
        rating_pairs = (
            np.array(
                [
                    self._global_user_id_map[ele]
                    for ele in rating_data["user_id"]
                ],
                dtype=np.int32,
            ),
            np.array(
                [
                    self._global_movie_id_map[ele]
                    for ele in rating_data["movie_id"]
                ],
                dtype=np.int32,
            ),
        )
        rating_values = rating_data["rating"].values.astype(np.float32)
        return rating_pairs[0], rating_pairs[1], rating_values