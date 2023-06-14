"""
Datasets introduced in the 'A Critical Look at the Evaluation of GNNs under Heterophily: Are We
Really Making Progress? <https://arxiv.org/abs/2302.11640>'__ paper.
"""
import os

import numpy as np

from ..convert import graph
from ..transforms.functional import to_bidirected
from .dgl_dataset import DGLBuiltinDataset
from .utils import download


class HeterophilousGraphDataset(DGLBuiltinDataset):
    r"""Datasets introduced in the 'A Critical Look at the Evaluation of GNNs under Heterophily:
    Are We Really Making Progress? <https://arxiv.org/abs/2302.11640>'__ paper.

    Parameters
    ----------
    name : str
        Name of the dataset. One of 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers',
        'questions'.
    raw_dir : str
        Raw file directory to store the processed data.
    force_reload : bool
        Whether to re-download the data source.
    verbose : bool
        Whether to print progress information.
    transform : callable
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self,
        name,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        transform=None,
    ):
        name = name.lower().replace("-", "_")
        url = f"https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{name}.npz"
        super(HeterophilousGraphDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def download(self):
        download(
            url=self.url, path=os.path.join(self.raw_path, f"{self.name}.npz")
        )

    def process(self):
        """Load and process the data."""
        try:
            import torch
        except ImportError:
            raise ModuleNotFoundError(
                "This dataset requires PyTorch to be the backend."
            )

        data = np.load(os.path.join(self.raw_path, f"{self.name}.npz"))
        src = torch.from_numpy(data["edges"][:, 0])
        dst = torch.from_numpy(data["edges"][:, 1])
        features = torch.from_numpy(data["node_features"])
        labels = torch.from_numpy(data["node_labels"])
        train_masks = torch.from_numpy(data["train_masks"].T)
        val_masks = torch.from_numpy(data["val_masks"].T)
        test_masks = torch.from_numpy(data["test_masks"].T)
        num_nodes = len(labels)
        num_classes = len(labels.unique())

        self._num_classes = num_classes

        self._g = to_bidirected(graph((src, dst), num_nodes=num_nodes))
        self._g.ndata["feat"] = features
        self._g.ndata["label"] = labels
        self._g.ndata["train_mask"] = train_masks
        self._g.ndata["val_mask"] = val_masks
        self._g.ndata["test_mask"] = test_masks

    def has_cache(self):
        return os.path.exists(self.raw_path)

    def load(self):
        self.process()

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph."
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes


class RomanEmpireDataset(HeterophilousGraphDataset):
    r"""Roman-empire dataset from the 'A Critical Look at the Evaluation of GNNs under Heterophily:
    Are We Really Making Progress? <https://arxiv.org/abs/2302.11640>'__ paper.

    This dataset is based on the Roman Empire article from English Wikipedia, which was selected
    since it is one of the longest articles on Wikipedia. Each node in the graph corresponds to one
    (non-unique) word in the text. Thus, the number of nodes in the graph is equal to the article’s
    length. Two words are connected with an edge if at least one of the following two conditions
    holds: either these words follow each other in the text, or these words are connected in the
    dependency tree of the sentence (one word depends on the other). Thus, the graph is a chain
    graph with additional shortcut edges corresponding to syntactic dependencies between words. The
    class of a node is its syntactic role (17 most frequent roles were selected as unique classes
    and all the other roles were grouped into the 18th class). Node features are word embeddings.

    Statistics:

    - Nodes: 22662
    - Edges: 65854
    - Classes: 18
    - Node features: 300
    - 10 train/val/test splits

    Parameters
    ----------
    raw_dir : str, optional
        Raw file directory to store the processed data. Default: ~/.dgl/
    force_reload : bool, optional
        Whether to re-download the data source. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access. Default: None

    Attributes
    ----------
    num_classes : int
        Number of node classes


    Examples
    --------

    >>> from dgl.data import RomanEmpireDataset
    >>> dataset = RomanEmpireDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get the first data split
    >>> train_mask = g.ndata["train_mask"][:, 0]
    >>> val_mask = g.ndata["val_mask"][:, 0]
    >>> test_mask = g.ndata["test_mask"][:, 0]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(RomanEmpireDataset, self).__init__(
            name="roman-empire",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class AmazonRatingsDataset(HeterophilousGraphDataset):
    r"""Amazon-ratings dataset from the 'A Critical Look at the Evaluation of GNNs under
    Heterophily: Are We Really Making Progress? <https://arxiv.org/abs/2302.11640>'__ paper.

    This dataset is based on the Amazon product co-purchasing data. Nodes are products (books, music
    CDs, DVDs, VHS video tapes), and edges connect products that are frequently bought together. The
    task is to predict the average rating given to a product by reviewers. All possible rating
    values were grouped into five classes. Node features are the mean of word embeddings for words
    in the product description.

    Statistics:

    - Nodes: 24492
    - Edges: 186100
    - Classes: 5
    - Node features: 300
    - 10 train/val/test splits

    Parameters
    ----------
    raw_dir : str, optional
        Raw file directory to store the processed data. Default: ~/.dgl/
    force_reload : bool, optional
        Whether to re-download the data source. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access. Default: None

    Attributes
    ----------
    num_classes : int
        Number of node classes


    Examples
    --------

    >>> from dgl.data import AmazonRatingsDataset
    >>> dataset = AmazonRatingsDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get the first data split
    >>> train_mask = g.ndata["train_mask"][:, 0]
    >>> val_mask = g.ndata["val_mask"][:, 0]
    >>> test_mask = g.ndata["test_mask"][:, 0]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(AmazonRatingsDataset, self).__init__(
            name="amazon-ratings",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class MinesweeperDataset(HeterophilousGraphDataset):
    r"""Minesweeper dataset from the 'A Critical Look at the Evaluation of GNNs under Heterophily:
    Are We Really Making Progress? <https://arxiv.org/abs/2302.11640>'__ paper.

    This dataset is inspired by the Minesweeper game. The graph is a regular 100x100 grid where each
    node (cell) is connected to eight neighboring nodes (with the exception of nodes at the edge of
    the grid, which have fewer neighbors). 20% of the nodes are randomly selected as mines. The task
    is to predict which nodes are mines. The node features are one-hot-encoded numbers of
    neighboring mines. However, for randomly selected 50% of the nodes, the features are unknown,
    which is indicated by a separate binary feature.

    Statistics:

    - Nodes: 10000
    - Edges: 78804
    - Classes: 2
    - Node features: 7
    - 10 train/val/test splits

    Parameters
    ----------
    raw_dir : str, optional
        Raw file directory to store the processed data. Default: ~/.dgl/
    force_reload : bool, optional
        Whether to re-download the data source. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access. Default: None

    Attributes
    ----------
    num_classes : int
        Number of node classes


    Examples
    --------

    >>> from dgl.data import MinesweeperDataset
    >>> dataset = MinesweeperDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get the first data split
    >>> train_mask = g.ndata["train_mask"][:, 0]
    >>> val_mask = g.ndata["val_mask"][:, 0]
    >>> test_mask = g.ndata["test_mask"][:, 0]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(MinesweeperDataset, self).__init__(
            name="minesweeper",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class TolokersDataset(HeterophilousGraphDataset):
    r"""Tolokers dataset from the 'A Critical Look at the Evaluation of GNNs under Heterophily:
    Are We Really Making Progress? <https://arxiv.org/abs/2302.11640>'__ paper.

    This dataset is based on data from the Toloka crowdsourcing platform. The nodes represent
    tolokers (workers). An edge connects two tolokers if they have worked on the same task. The goal
    is to predict which tolokers have been banned in one of the projects. Node features are based on
    the worker’s profile information and task performance statistics.

    Statistics:

    - Nodes: 11758
    - Edges: 1038000
    - Classes: 2
    - Node features: 10
    - 10 train/val/test splits

    Parameters
    ----------
    raw_dir : str, optional
        Raw file directory to store the processed data. Default: ~/.dgl/
    force_reload : bool, optional
        Whether to re-download the data source. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access. Default: None

    Attributes
    ----------
    num_classes : int
        Number of node classes


    Examples
    --------

    >>> from dgl.data import TolokersDataset
    >>> dataset = TolokersDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get the first data split
    >>> train_mask = g.ndata["train_mask"][:, 0]
    >>> val_mask = g.ndata["val_mask"][:, 0]
    >>> test_mask = g.ndata["test_mask"][:, 0]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(TolokersDataset, self).__init__(
            name="tolokers",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class QuestionsDataset(HeterophilousGraphDataset):
    r"""Questions dataset from the 'A Critical Look at the Evaluation of GNNs under Heterophily:
    Are We Really Making Progress? <https://arxiv.org/abs/2302.11640>'__ paper.

    This dataset is based on data from the question-answering website Yandex Q. Nodes are users, and
    an edge connects two nodes if one user answered the other user’s question. The task is to
    predict which users remained active on the website (were not deleted or blocked). Node features
    are the mean of word embeddings for words in the user description. Users that do not have
    description are indicated by a separate binary feature.

    Statistics:

    - Nodes: 48921
    - Edges: 307080
    - Classes: 2
    - Node features: 301
    - 10 train/val/test splits

    Parameters
    ----------
    raw_dir : str, optional
        Raw file directory to store the processed data. Default: ~/.dgl/
    force_reload : bool, optional
        Whether to re-download the data source. Default: False
    verbose : bool, optional
        Whether to print progress information. Default: True
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access. Default: None

    Attributes
    ----------
    num_classes : int
        Number of node classes


    Examples
    --------

    >>> from dgl.data import QuestionsDataset
    >>> dataset = QuestionsDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get the first data split
    >>> train_mask = g.ndata["train_mask"][:, 0]
    >>> val_mask = g.ndata["val_mask"][:, 0]
    >>> test_mask = g.ndata["test_mask"][:, 0]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(QuestionsDataset, self).__init__(
            name="questions",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )
