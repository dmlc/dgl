import os
import pickle

import numpy as np
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from .. import backend as F
from ..convert import graph as dgl_graph

from .dgl_dataset import DGLDataset
from .utils import download, extract_archive, load_graphs, save_graphs, Subset


def sigma(dists, kth=8):
    num_nodes = dists.shape[0]

    # Compute sigma and reshape.
    if kth > num_nodes:
        # Handling for graphs with num_nodes less than kth.
        sigma = np.array([1] * num_nodes).reshape(num_nodes, 1)
    else:
        # Get k-nearest neighbors for each node.
        knns = np.partition(dists, kth, axis=-1)[:, : kth + 1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / kth

    return sigma + 1e-8


def compute_adjacency_matrix_images(coord, feat, use_feat=True):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance.
    c_dist = cdist(coord, coord)

    if use_feat:
        # Compute feature distance.
        f_dist = cdist(feat, feat)
        # Compute adjacency.
        A = np.exp(
            -((c_dist / sigma(c_dist)) ** 2) - (f_dist / sigma(f_dist)) ** 2
        )
    else:
        A = np.exp(-((c_dist / sigma(c_dist)) ** 2))

    # Convert to symmetric matrix.
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A


def compute_edges_list(A, kth=9):
    # Get k-similar neighbor indices for each node.
    num_nodes = A.shape[0]
    new_kth = num_nodes - kth

    if num_nodes > kth:
        knns = np.argpartition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
    else:
        # Handling for graphs with less than kth nodes.
        # In such cases, the resulting graph will be fully connected.
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(
            num_nodes, num_nodes
        )
        knn_values = A

        # Removing self loop.
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:, None]].reshape(
                num_nodes, -1
            )
            knns = knns[knns != np.arange(num_nodes)[:, None]].reshape(
                num_nodes, -1
            )
    return knns, knn_values


class SuperPixelDataset(DGLDataset):
    def __init__(
        self,
        raw_dir=None,
        name="MNIST",
        split="train",
        use_feature=False,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        assert split in ["train", "test"], "split not valid."
        assert name in ["MNIST", "CIFAR10"], "name not valid."

        self.use_feature = use_feature
        self.split = split
        self._dataset_name = name
        self.graphs = []
        self.labels = []

        super().__init__(
            name="Superpixel",
            raw_dir=raw_dir,
            url="""
            https://www.dropbox.com/s/y2qwa77a0fxem47/superpixels.zip?dl=1
            """,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    @property
    def img_size(self):
        r"""Size of dataset image."""
        if self._dataset_name == "MNIST":
            return 28
        return 32

    @property
    def save_path(self):
        r"""Directory to save the processed dataset."""
        return os.path.join(self.raw_path, "processed")

    @property
    def raw_data_path(self):
        r"""Path to save the raw dataset file."""
        return os.path.join(self.raw_path, "superpixels.zip")

    @property
    def graph_path(self):
        r"""Path to save the processed dataset file."""
        if self.use_feature:
            return os.path.join(
                self.save_path,
                f"use_feat_{self._dataset_name}_{self.split}.pkl",
            )
        return os.path.join(
            self.save_path, f"{self._dataset_name}_{self.split}.pkl"
        )

    def download(self):
        path = download(self.url, path=self.raw_data_path)
        extract_archive(path, target_dir=self.raw_path, overwrite=True)

    def process(self):
        if self._dataset_name == "MNIST":
            plk_file = "mnist_75sp"
        elif self._dataset_name == "CIFAR10":
            plk_file = "cifar10_150sp"

        with open(
            os.path.join(
                self.raw_path, "superpixels", f"{plk_file}_{self.split}.pkl"
            ),
            "rb",
        ) as f:
            self.labels, self.sp_data = pickle.load(f)
            self.labels = F.tensor(self.labels)

        self.Adj_matrices = []
        self.node_features = []
        self.edges_lists = []
        self.edge_features = []

        for index, sample in enumerate(
            tqdm(self.sp_data, desc=f"Processing {self.split} dataset")
        ):
            mean_px, coord = sample[:2]
            coord = coord / self.img_size

            if self.use_feature:
                A = compute_adjacency_matrix_images(
                    coord, mean_px
                )  # using super-pixel locations + features
            else:
                A = compute_adjacency_matrix_images(
                    coord, mean_px, False
                )  # using only super-pixel locations
            edges_list, edge_values_list = compute_edges_list(A)

            N_nodes = A.shape[0]

            mean_px = mean_px.reshape(N_nodes, -1)
            coord = coord.reshape(N_nodes, 2)
            x = np.concatenate((mean_px, coord), axis=1)

            edge_values_list = edge_values_list.reshape(-1)

            self.node_features.append(x)
            self.edge_features.append(edge_values_list)
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)

        for index in tqdm(
            range(len(self.sp_data)), desc=f"Dump {self.split} dataset"
        ):
            N = self.node_features[index].shape[0]

            src_nodes = []
            dst_nodes = []
            for src, dsts in enumerate(self.edges_lists[index]):
                # handling for 1 node where the self loop would be the only edge
                if N == 1:
                    src_nodes.append(src)
                    dst_nodes.append(dsts)
                else:
                    dsts = dsts[dsts != src]
                    srcs = [src] * len(dsts)
                    src_nodes.extend(srcs)
                    dst_nodes.extend(dsts)

            src_nodes = F.tensor(src_nodes)
            dst_nodes = F.tensor(dst_nodes)

            g = dgl_graph((src_nodes, dst_nodes), num_nodes=N)
            g.ndata["feat"] = F.zerocopy_from_numpy(
                self.node_features[index]
            ).to(F.float32)
            g.edata["feat"] = (
                F.zerocopy_from_numpy(self.edge_features[index])
                .to(F.float32)
                .unsqueeze(1)
            )

            self.graphs.append(g)

    def load(self):
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.labels = label_dict["labels"]

    def save(self):
        save_graphs(
            self.graph_path, self.graphs, labels={"labels": self.labels}
        )

    def has_cache(self):
        return os.path.exists(self.graph_path)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the idx-th sample.

        Parameters
        ---------
        idx : int or tensor
            The sample index.
            1-D tensor as `idx` is allowed when transform is None.

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
            Graph with node feature stored in ``feat`` field and its label.
        or
        :class:`dgl.data.utils.Subset`
            Subset of the dataset at specified indices
        """
        if F.is_tensor(idx) and idx.dim() == 1:
            if self._transform is None:
                return Subset(self, idx.cpu())

            raise ValueError(
                "Tensor idx not supported when transform is not None."
            )

        if self._transform is None:
            return self.graphs[idx], self.labels[idx]

        return self._transform(self.graphs[idx]), self.labels[idx]


class MNISTSuperPixelDataset(SuperPixelDataset):
    r"""MNIST superpixel dataset for the graph classification task.

    DGL dataset of MNIST and CIFAR10 in the benchmark-gnn which contains graphs
    converted fromt the original MINST and CIFAR10 images.

    Reference `<http://arxiv.org/abs/2003.00982>`_

    Statistics:

        - Train examples: 60,000
        - Test examples: 10,000
        - Size of dataset images: 28

    Parameters
    ----------
    raw_dir : str
        Directory to store all the downloaded raw datasets.
        Default: "~/.dgl/".
    split : str
        Should be chosen from ["train", "test"]
        Default: "train".
    use_feature: bool

        - True: Adj matrix defined from super-pixel locations + features
        - False: Adj matrix defined from super-pixel locations (only)

        Default: False.
    force_reload : bool
        Whether to reload the dataset.
        Default: False.
    verbose : bool
        Whether to print out progress information.
        Default: False.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Examples
    ---------
    >>> from dgl.data import MNISTSuperPixelDataset

    >>> # MNIST dataset
    >>> train_dataset = MNISTSuperPixelDataset(split="train")
    >>> len(train_dataset)
    60000
    >>> graph, label = train_dataset[0]
    >>> graph
    Graph(num_nodes=71, num_edges=568,
        ndata_schemes={'feat': Scheme(shape=(3,), dtype=torch.float32)}
        edata_schemes={'feat': Scheme(shape=(1,), dtype=torch.float32)})

    >>> # support tensor to be index when transform is None
    >>> # see details in __getitem__ function
    >>> import torch
    >>> idx = torch.tensor([0, 1, 2])
    >>> train_dataset_subset = train_dataset[idx]
    >>> train_dataset_subset[0]
    Graph(num_nodes=71, num_edges=568,
        ndata_schemes={'feat': Scheme(shape=(3,), dtype=torch.float32)}
        edata_schemes={'feat': Scheme(shape=(1,), dtype=torch.float32)})
    """

    def __init__(
        self,
        raw_dir=None,
        split="train",
        use_feature=False,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        super().__init__(
            raw_dir=raw_dir,
            name="MNIST",
            split=split,
            use_feature=use_feature,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class CIFAR10SuperPixelDataset(SuperPixelDataset):
    r"""CIFAR10 superpixel dataset for the graph classification task.

    DGL dataset of CIFAR10 in the benchmark-gnn which contains graphs
    converted fromt the original CIFAR10 images.

    Reference `<http://arxiv.org/abs/2003.00982>`_

    Statistics:

        - Train examples: 50,000
        - Test examples: 10,000
        - Size of dataset images: 32

    Parameters
    ----------
    raw_dir : str
        Directory to store all the downloaded raw datasets.
        Default: "~/.dgl/".
    split : str
        Should be chosen from ["train", "test"]
        Default: "train".
    use_feature: bool

        - True: Adj matrix defined from super-pixel locations + features
        - False: Adj matrix defined from super-pixel locations (only)

        Default: False.
    force_reload : bool
        Whether to reload the dataset.
        Default: False.
    verbose : bool
        Whether to print out progress information.
        Default: False.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Examples
    ---------
    >>> from dgl.data import CIFAR10SuperPixelDataset

    >>> # CIFAR10 dataset
    >>> train_dataset = CIFAR10SuperPixelDataset(split="train")
    >>> len(train_dataset)
    50000
    >>> graph, label = train_dataset[0]
    >>> graph
    Graph(num_nodes=123, num_edges=984,
        ndata_schemes={'feat': Scheme(shape=(5,), dtype=torch.float32)}
        edata_schemes={'feat': Scheme(shape=(1,), dtype=torch.float32)}),

    >>> # support tensor to be index when transform is None
    >>> # see details in __getitem__ function
    >>> import torch
    >>> idx = torch.tensor([0, 1, 2])
    >>> train_dataset_subset = train_dataset[idx]
    >>> train_dataset_subset[0]
    Graph(num_nodes=123, num_edges=984,
        ndata_schemes={'feat': Scheme(shape=(5,), dtype=torch.float32)}
        edata_schemes={'feat': Scheme(shape=(1,), dtype=torch.float32)}),
    """

    def __init__(
        self,
        raw_dir=None,
        split="train",
        use_feature=False,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        super().__init__(
            raw_dir=raw_dir,
            name="CIFAR10",
            split=split,
            use_feature=use_feature,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )
