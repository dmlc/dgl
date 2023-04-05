"""
Actor-only induced subgraph of the film-directoractor-writer network.
"""
import os

import numpy as np

from ..convert import graph
from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url


class ActorDataset(DGLBuiltinDataset):
    r"""Actor-only induced subgraph of the film-directoractor-writer network
    from `Social Influence Analysis in Large-scale Networks
    <https://dl.acm.org/doi/10.1145/1557019.1557108>`, introduced by
    `Geom-GCN: Geometric Graph Convolutional Networks
    <https://arxiv.org/abs/2002.05287>`

    Nodes represent actors, and edges represent co-occurrence on the same
    Wikipedia page. Node features correspond to some keywords in the Wikipedia
    pages.

    Statistics:

    - Nodes: 7600
    - Edges: 33391
    - Number of Classes: 5
    - 10 train/val/test splits

        - Train: 3648
        - Val: 2432
        - Test: 1520

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
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        super(ActorDataset, self).__init__(
            name="actor",
            url=_get_dgl_url("dataset/actor.zip"),
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
            data = [x.split("\t") for x in f.read().split("\n")[1:-1]]

            rows, cols = [], []
            labels = torch.empty(len(data), dtype=torch.long)
            for n_id, col, label in data:
                col = [int(x) for x in col.split(",")]
                rows += [int(n_id)] * len(col)
                cols += col

                labels[int(n_id)] = int(label)

            row, col = torch.tensor(rows), torch.tensor(cols)
            features = torch.zeros(len(data), int(col.max()) + 1)
            features[row, col] = 1.0

            self._num_classes = int(labels.max().item()) + 1

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
