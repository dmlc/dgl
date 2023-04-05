"""Datasets introduced in the Geom-GCN paper."""
import os

import numpy as np

from ..convert import graph
from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url


class GeomGCNDataset(DGLBuiltinDataset):
    r"""Datasets introduced in
    `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    Parameters
    ----------
    name : str
        Name of the dataset.
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

    def __init__(self, name, raw_dir, force_reload, verbose, transform):
        url = _get_dgl_url(f"dataset/{name}.zip")
        super(GeomGCNDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        """Load and process the data."""
        try:
            import torch
        except ImportError:
            raise ModuleNotFoundError(
                "This dataset requires PyTorch to be the backend."
            )

        # Process node features and labels.
        with open(f"{self.raw_path}/out1_node_feature_label.txt", "r") as f:
            data = f.read().split("\n")[1:-1]
        features = [
            [float(v) for v in r.split("\t")[1].split(",")] for r in data
        ]
        features = torch.tensor(features, dtype=torch.float)
        labels = [int(r.split("\t")[2]) for r in data]
        self._num_classes = max(labels) + 1
        labels = torch.tensor(labels, dtype=torch.long)

        # Process graph structure.
        with open(f"{self.raw_path}/out1_graph_edges.txt", "r") as f:
            data = f.read().split("\n")[1:-1]
            data = [[int(v) for v in r.split("\t")] for r in data]
        dst, src = torch.tensor(data, dtype=torch.long).t().contiguous()

        self._g = graph((src, dst), num_nodes=features.size(0))
        self._g.ndata["feat"] = features
        self._g.ndata["label"] = labels

        # Process 10 train/val/test node splits.
        train_masks, val_masks, test_masks = [], [], []
        for i in range(10):
            filepath = f"{self.raw_path}/{self.name}_split_0.6_0.2_{i}.npz"
            f = np.load(filepath)
            train_masks += [torch.from_numpy(f["train_mask"])]
            val_masks += [torch.from_numpy(f["val_mask"])]
            test_masks += [torch.from_numpy(f["test_mask"])]
        self._g.ndata["train_mask"] = torch.stack(train_masks, dim=1).bool()
        self._g.ndata["val_mask"] = torch.stack(val_masks, dim=1).bool()
        self._g.ndata["test_mask"] = torch.stack(test_masks, dim=1).bool()

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


class ChameleonDataset(GeomGCNDataset):
    r"""Wikipedia page-page network on chameleons from `Multi-scale Attributed
    Node Embedding <https://arxiv.org/abs/1909.13021>`__ and later modified by
    `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    Nodes represent articles from the English Wikipedia, edges reflect mutual
    links between them. Node features indicate the presence of particular nouns
    in the articles. The nodes were classified into 5 classes in terms of their
    average monthly traffic.

    Statistics:

    - Nodes: 2277
    - Edges: 36101
    - Number of Classes: 5
    - 10 train/val/test splits

        - Train: 1092
        - Val: 729
        - Test: 456

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

    Notes
    -----
    The graph does not come with edges for both directions.

    Examples
    --------

    >>> from dgl.data import ChameleonDataset
    >>> dataset = ChameleonDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get data split
    >>> train_mask = g.ndata["train_mask"]
    >>> val_mask = g.ndata["val_mask"]
    >>> test_mask = g.ndata["test_mask"]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(ChameleonDataset, self).__init__(
            name="chameleon",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class SquirrelDataset(GeomGCNDataset):
    r"""Wikipedia page-page network on squirrels from `Multi-scale Attributed
    Node Embedding <https://arxiv.org/abs/1909.13021>`__ and later modified by
    `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    Nodes represent articles from the English Wikipedia, edges reflect mutual
    links between them. Node features indicate the presence of particular nouns
    in the articles. The nodes were classified into 5 classes in terms of their
    average monthly traffic.

    Statistics:

    - Nodes: 5201
    - Edges: 217073
    - Number of Classes: 5
    - 10 train/val/test splits

        - Train: 2496
        - Val: 1664
        - Test: 1041

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

    Notes
    -----
    The graph does not come with edges for both directions.

    Examples
    --------

    >>> from dgl.data import SquirrelDataset
    >>> dataset = SquirrelDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get data split
    >>> train_mask = g.ndata["train_mask"]
    >>> val_mask = g.ndata["val_mask"]
    >>> test_mask = g.ndata["test_mask"]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(SquirrelDataset, self).__init__(
            name="squirrel",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class CornellDataset(GeomGCNDataset):
    r"""Cornell subset of
    `WebKB <http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-11/www/wwkb/>`__,
    later modified by `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    Nodes represent web pages. Edges represent hyperlinks between them. Node
    features are the bag-of-words representation of web pages. The web pages
    are manually classified into the five categories, student, project, course,
    staff, and faculty.

    Statistics:

    - Nodes: 183
    - Edges: 298
    - Number of Classes: 5
    - 10 train/val/test splits

        - Train: 87
        - Val: 59
        - Test: 37

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

    Notes
    -----
    The graph does not come with edges for both directions.

    Examples
    --------

    >>> from dgl.data import CornellDataset
    >>> dataset = CornellDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get data split
    >>> train_mask = g.ndata["train_mask"]
    >>> val_mask = g.ndata["val_mask"]
    >>> test_mask = g.ndata["test_mask"]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(CornellDataset, self).__init__(
            name="cornell",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class TexasDataset(GeomGCNDataset):
    r"""Texas subset of
    `WebKB <http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-11/www/wwkb/>`__,
    later modified by `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    Nodes represent web pages. Edges represent hyperlinks between them. Node
    features are the bag-of-words representation of web pages. The web pages
    are manually classified into the five categories, student, project, course,
    staff, and faculty.

    Statistics:

    - Nodes: 183
    - Edges: 325
    - Number of Classes: 5
    - 10 train/val/test splits

        - Train: 87
        - Val: 59
        - Test: 37

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

    Notes
    -----
    The graph does not come with edges for both directions.

    Examples
    --------

    >>> from dgl.data import TexasDataset
    >>> dataset = TexasDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get data split
    >>> train_mask = g.ndata["train_mask"]
    >>> val_mask = g.ndata["val_mask"]
    >>> test_mask = g.ndata["test_mask"]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(TexasDataset, self).__init__(
            name="texas",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )


class WisconsinDataset(GeomGCNDataset):
    r"""Wisconsin subset of
    `WebKB <http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-11/www/wwkb/>`__,
    later modified by `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`__

    Nodes represent web pages. Edges represent hyperlinks between them. Node
    features are the bag-of-words representation of web pages. The web pages
    are manually classified into the five categories, student, project, course,
    staff, and faculty.

    Statistics:

    - Nodes: 251
    - Edges: 515
    - Number of Classes: 5
    - 10 train/val/test splits

        - Train: 120
        - Val: 80
        - Test: 51

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

    Notes
    -----
    The graph does not come with edges for both directions.

    Examples
    --------

    >>> from dgl.data import WisconsinDataset
    >>> dataset = WisconsinDataset()
    >>> g = dataset[0]
    >>> num_classes = dataset.num_classes

    >>> # get node features
    >>> feat = g.ndata["feat"]

    >>> # get data split
    >>> train_mask = g.ndata["train_mask"]
    >>> val_mask = g.ndata["val_mask"]
    >>> test_mask = g.ndata["test_mask"]

    >>> # get labels
    >>> label = g.ndata['label']
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(WisconsinDataset, self).__init__(
            name="wisconsin",
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )
