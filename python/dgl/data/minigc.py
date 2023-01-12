"""A mini synthetic dataset for graph classification benchmark."""
import math
import os

import networkx as nx
import numpy as np

from .. import backend as F
from ..convert import from_networkx
from ..transforms import add_self_loop
from .dgl_dataset import DGLDataset
from .utils import load_graphs, makedirs, save_graphs

__all__ = ["MiniGCDataset"]


class MiniGCDataset(DGLDataset):
    """The synthetic graph classification dataset class.

    The datset contains 8 different types of graphs.

    - class 0 : cycle graph
    - class 1 : star graph
    - class 2 : wheel graph
    - class 3 : lollipop graph
    - class 4 : hypercube graph
    - class 5 : grid graph
    - class 6 : clique graph
    - class 7 : circular ladder graph

    Parameters
    ----------
    num_graphs: int
        Number of graphs in this dataset.
    min_num_v: int
        Minimum number of nodes for graphs
    max_num_v: int
        Maximum number of nodes for graphs
    seed: int, default is 0
        Random seed for data generation
    transform: callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_graphs : int
        Number of graphs
    min_num_v : int
        The minimum number of nodes
    max_num_v : int
        The maximum number of nodes
    num_classes : int
        The number of classes

    Examples
    --------
    >>> data = MiniGCDataset(100, 16, 32, seed=0)

    The dataset instance is an iterable

    >>> len(data)
    100
    >>> g, label = data[64]
    >>> g
    Graph(num_nodes=20, num_edges=82,
          ndata_schemes={}
          edata_schemes={})
    >>> label
    tensor(5)

    Batch the graphs and labels for mini-batch training

    >>> graphs, labels = zip(*[data[i] for i in range(16)])
    >>> batched_graphs = dgl.batch(graphs)
    >>> batched_labels = torch.tensor(labels)
    >>> batched_graphs
    Graph(num_nodes=356, num_edges=1060,
          ndata_schemes={}
          edata_schemes={})
    """

    def __init__(
        self,
        num_graphs,
        min_num_v,
        max_num_v,
        seed=0,
        save_graph=True,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        self.num_graphs = num_graphs
        self.min_num_v = min_num_v
        self.max_num_v = max_num_v
        self.seed = seed
        self.save_graph = save_graph

        super(MiniGCDataset, self).__init__(
            name="minigc",
            hash_key=(num_graphs, min_num_v, max_num_v, seed),
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        self.graphs = []
        self.labels = []
        self._generate(self.seed)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the idx-th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (:class:`dgl.Graph`, Tensor)
            The graph and its label.
        """
        if self._transform is None:
            g = self.graphs[idx]
        else:
            g = self._transform(self.graphs[idx])
        return g, self.labels[idx]

    def has_cache(self):
        graph_path = os.path.join(
            self.save_path, "dgl_graph_{}.bin".format(self.hash)
        )
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        if self.save_graph:
            graph_path = os.path.join(
                self.save_path, "dgl_graph_{}.bin".format(self.hash)
            )
            save_graphs(str(graph_path), self.graphs, {"labels": self.labels})

    def load(self):
        graphs, label_dict = load_graphs(
            os.path.join(self.save_path, "dgl_graph_{}.bin".format(self.hash))
        )
        self.graphs = graphs
        self.labels = label_dict["labels"]

    @property
    def num_classes(self):
        """Number of classes."""
        return 8

    def _generate(self, seed):
        if seed is not None:
            np.random.seed(seed)
        self._gen_cycle(self.num_graphs // 8)
        self._gen_star(self.num_graphs // 8)
        self._gen_wheel(self.num_graphs // 8)
        self._gen_lollipop(self.num_graphs // 8)
        self._gen_hypercube(self.num_graphs // 8)
        self._gen_grid(self.num_graphs // 8)
        self._gen_clique(self.num_graphs // 8)
        self._gen_circular_ladder(self.num_graphs - len(self.graphs))
        # preprocess
        for i in range(self.num_graphs):
            # convert to DGLGraph, and add self loops
            self.graphs[i] = add_self_loop(from_networkx(self.graphs[i]))
        self.labels = F.tensor(np.array(self.labels).astype(np.int64))

    def _gen_cycle(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.cycle_graph(num_v)
            self.graphs.append(g)
            self.labels.append(0)

    def _gen_star(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            # nx.star_graph(N) gives a star graph with N+1 nodes
            g = nx.star_graph(num_v - 1)
            self.graphs.append(g)
            self.labels.append(1)

    def _gen_wheel(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.wheel_graph(num_v)
            self.graphs.append(g)
            self.labels.append(2)

    def _gen_lollipop(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            path_len = np.random.randint(2, num_v // 2)
            g = nx.lollipop_graph(m=num_v - path_len, n=path_len)
            self.graphs.append(g)
            self.labels.append(3)

    def _gen_hypercube(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.hypercube_graph(int(math.log(num_v, 2)))
            g = nx.convert_node_labels_to_integers(g)
            self.graphs.append(g)
            self.labels.append(4)

    def _gen_grid(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            assert num_v >= 4, (
                "We require a grid graph to contain at least two "
                "rows and two columns, thus 4 nodes, got {:d} "
                "nodes".format(num_v)
            )
            n_rows = np.random.randint(2, num_v // 2)
            n_cols = num_v // n_rows
            g = nx.grid_graph([n_rows, n_cols])
            g = nx.convert_node_labels_to_integers(g)
            self.graphs.append(g)
            self.labels.append(5)

    def _gen_clique(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.complete_graph(num_v)
            self.graphs.append(g)
            self.labels.append(6)

    def _gen_circular_ladder(self, n):
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.circular_ladder_graph(num_v // 2)
            self.graphs.append(g)
            self.labels.append(7)
