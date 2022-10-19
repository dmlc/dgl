"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import json
import os

import numpy as np

from .. import backend as F
from ..base import DGLError
from ..convert import graph as create_dgl_graph
from ..sampling.negative import _calc_redundancy
from . import utils
from .dgl_dataset import DGLDataset

__all__ = ["AsNodePredDataset", "AsLinkPredDataset", "AsGraphPredDataset"]


class AsNodePredDataset(DGLDataset):
    """Repurpose a dataset for a standard semi-supervised transductive
    node prediction task.

    The class converts a given dataset into a new dataset object such that:

      - Contains only one graph, accessible from ``dataset[0]``.
      - The graph stores:

        - Node labels in ``g.ndata['label']``.
        - Train/val/test masks in ``g.ndata['train_mask']``, ``g.ndata['val_mask']``,
          and ``g.ndata['test_mask']`` respectively.
      - In addition, the dataset contains the following attributes:

        - ``num_classes``, the number of classes to predict.
        - ``train_idx``, ``val_idx``, ``test_idx``, train/val/test indexes.

    If the input dataset contains heterogeneous graphs, users need to specify the
    ``target_ntype`` argument to indicate which node type to make predictions for.
    In this case:

      - Node labels are stored in ``g.nodes[target_ntype].data['label']``.
      - Training masks are stored in ``g.nodes[target_ntype].data['train_mask']``.
        So do validation and test masks.

    The class will keep only the first graph in the provided dataset and
    generate train/val/test masks according to the given split ratio. The generated
    masks will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. They must sum to one.
    target_ntype : str, optional
        The node type to add split mask for.

    Attributes
    ----------
    num_classes : int
        Number of classes to predict.
    train_idx : Tensor
        An 1-D integer tensor of training node IDs.
    val_idx : Tensor
        An 1-D integer tensor of validation node IDs.
    test_idx : Tensor
        An 1-D integer tensor of test node IDs.

    Examples
    --------
    >>> ds = dgl.data.AmazonCoBuyComputerDataset()
    >>> print(ds)
    Dataset("amazon_co_buy_computer", num_graphs=1, save_path=...)
    >>> new_ds = dgl.data.AsNodePredDataset(ds, [0.8, 0.1, 0.1])
    >>> print(new_ds)
    Dataset("amazon_co_buy_computer-as-nodepred", num_graphs=1, save_path=...)
    >>> print('train_mask' in new_ds[0].ndata)
    True
    """

    def __init__(self, dataset, split_ratio=None, target_ntype=None, **kwargs):
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.target_ntype = target_ntype
        super().__init__(
            self.dataset.name + "-as-nodepred",
            hash_key=(split_ratio, target_ntype, dataset.name, "nodepred"),
            **kwargs
        )

    def process(self):
        is_ogb = hasattr(self.dataset, "get_idx_split")
        if is_ogb:
            g, label = self.dataset[0]
            self.g = g.clone()
            self.g.ndata["label"] = F.reshape(label, (g.num_nodes(),))
        else:
            self.g = self.dataset[0].clone()

        if "label" not in self.g.nodes[self.target_ntype].data:
            raise ValueError(
                "Missing node labels. Make sure labels are stored "
                "under name 'label'."
            )

        if self.split_ratio is None:
            if is_ogb:
                split = self.dataset.get_idx_split()
                train_idx, val_idx, test_idx = (
                    split["train"],
                    split["valid"],
                    split["test"],
                )
                n = self.g.num_nodes()
                train_mask = utils.generate_mask_tensor(
                    utils.idx2mask(train_idx, n)
                )
                val_mask = utils.generate_mask_tensor(
                    utils.idx2mask(val_idx, n)
                )
                test_mask = utils.generate_mask_tensor(
                    utils.idx2mask(test_idx, n)
                )
                self.g.ndata["train_mask"] = train_mask
                self.g.ndata["val_mask"] = val_mask
                self.g.ndata["test_mask"] = test_mask
            else:
                assert (
                    "train_mask" in self.g.nodes[self.target_ntype].data
                ), "train_mask is not provided, please specify split_ratio to generate the masks"
                assert (
                    "val_mask" in self.g.nodes[self.target_ntype].data
                ), "val_mask is not provided, please specify split_ratio to generate the masks"
                assert (
                    "test_mask" in self.g.nodes[self.target_ntype].data
                ), "test_mask is not provided, please specify split_ratio to generate the masks"
        else:
            if self.verbose:
                print("Generating train/val/test masks...")
            utils.add_nodepred_split(self, self.split_ratio, self.target_ntype)

        self._set_split_index()

        self.num_classes = getattr(self.dataset, "num_classes", None)
        if self.num_classes is None:
            self.num_classes = len(
                F.unique(self.g.nodes[self.target_ntype].data["label"])
            )

    def has_cache(self):
        return os.path.isfile(
            os.path.join(self.save_path, "graph_{}.bin".format(self.hash))
        )

    def load(self):
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "r"
        ) as f:
            info = json.load(f)
            if (
                info["split_ratio"] != self.split_ratio
                or info["target_ntype"] != self.target_ntype
            ):
                raise ValueError(
                    "Provided split ratio is different from the cached file. "
                    "Re-process the dataset."
                )
            self.split_ratio = info["split_ratio"]
            self.target_ntype = info["target_ntype"]
            self.num_classes = info["num_classes"]
        gs, _ = utils.load_graphs(
            os.path.join(self.save_path, "graph_{}.bin".format(self.hash))
        )
        self.g = gs[0]
        self._set_split_index()

    def save(self):
        utils.save_graphs(
            os.path.join(self.save_path, "graph_{}.bin".format(self.hash)),
            [self.g],
        )
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "w"
        ) as f:
            json.dump(
                {
                    "split_ratio": self.split_ratio,
                    "target_ntype": self.target_ntype,
                    "num_classes": self.num_classes,
                },
                f,
            )

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1

    def _set_split_index(self):
        """Add train_idx/val_idx/test_idx as dataset attributes according to corresponding mask."""
        ndata = self.g.nodes[self.target_ntype].data
        self.train_idx = F.nonzero_1d(ndata["train_mask"])
        self.val_idx = F.nonzero_1d(ndata["val_mask"])
        self.test_idx = F.nonzero_1d(ndata["test_mask"])


def negative_sample(g, num_samples):
    """Random sample negative edges from graph, excluding self-loops,
    the result samples might be less than num_samples
    """
    num_nodes = g.num_nodes()
    redundancy = _calc_redundancy(num_samples, g.num_edges(), num_nodes**2)
    sample_size = int(num_samples * (1 + redundancy))
    edges = np.random.randint(0, num_nodes, size=(2, sample_size))
    edges = np.unique(edges, axis=1)
    # remove self loop
    mask_self_loop = edges[0] == edges[1]
    # remove existing edges
    has_edges = F.asnumpy(g.has_edges_between(edges[0], edges[1]))
    mask = ~(np.logical_or(mask_self_loop, has_edges))
    edges = edges[:, mask]
    if edges.shape[1] >= num_samples:
        edges = edges[:, :num_samples]
    return edges


class AsLinkPredDataset(DGLDataset):
    """Repurpose a dataset for link prediction task.

    The created dataset will include data needed for link prediction.
    Currently it only supports homogeneous graphs.
    It will keep only the first graph in the provided dataset and
    generate train/val/test edges according to the given split ratio,
    and the correspondent negative edges based on the neg_ratio. The generated
    edges will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    neg_ratio : int, optional
        Indicate how much negative samples to be sampled
        The number of the negative samples will be equal or less than neg_ratio * num_positive_edges.

    Attributes
    -------
    feat_size: int
        The size of the feature dimension in the graph
    train_graph: DGLGraph
        The DGLGraph for training
    val_edges: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        The validation set edges, encoded as
        ((positive_edge_src, positive_edge_dst), (negative_edge_src, negative_edge_dst))
    test_edges: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
        The test set edges, encoded as
        ((positive_edge_src, positive_edge_dst), (negative_edge_src, negative_edge_dst))

    Examples
    --------
    >>> ds = dgl.data.CoraGraphDataset()
    >>> print(ds)
    Dataset("cora_v2", num_graphs=1, save_path=...)
    >>> new_ds = dgl.data.AsLinkPredDataset(ds, [0.8, 0.1, 0.1])
    >>> print(new_ds)
    Dataset("cora_v2-as-linkpred", num_graphs=1, save_path=/home/ubuntu/.dgl/cora_v2-as-linkpred)
    >>> print(hasattr(new_ds, "test_edges"))
    True
    """

    def __init__(self, dataset, split_ratio=None, neg_ratio=3, **kwargs):
        self.g = dataset[0]
        self.num_nodes = self.g.num_nodes()
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.neg_ratio = neg_ratio
        super().__init__(
            dataset.name + "-as-linkpred",
            hash_key=(neg_ratio, split_ratio, dataset.name, "linkpred"),
            **kwargs
        )

    def process(self):
        if self.split_ratio is None:
            # Handle logics for OGB link prediction dataset
            assert hasattr(
                self.dataset, "get_edge_split"
            ), "dataset doesn't have get_edge_split method, please specify split_ratio and neg_ratio to generate the split"
            # This is likely to be an ogb dataset
            self.edge_split = self.dataset.get_edge_split()
            self._train_graph = self.g
            if "source_node" in self.edge_split["test"]:
                # Probably ogbl-citation2
                pos_e = (
                    self.edge_split["valid"]["source_node"],
                    self.edge_split["valid"]["target_node"],
                )
                neg_e_size = self.edge_split["valid"]["target_node_neg"].shape[
                    -1
                ]
                neg_e_src = np.repeat(
                    self.edge_split["valid"]["source_node"], neg_e_size
                )
                neg_e_dst = np.reshape(
                    self.edge_split["valid"]["target_node_neg"], -1
                )
                self._val_edges = pos_e, (neg_e_src, neg_e_dst)
                pos_e = (
                    self.edge_split["test"]["source_node"],
                    self.edge_split["test"]["target_node"],
                )
                neg_e_size = self.edge_split["test"]["target_node_neg"].shape[
                    -1
                ]
                neg_e_src = np.repeat(
                    self.edge_split["test"]["source_node"], neg_e_size
                )
                neg_e_dst = np.reshape(
                    self.edge_split["test"]["target_node_neg"], -1
                )
                self._test_edges = pos_e, (neg_e_src, neg_e_dst)
            elif "edge" in self.edge_split["test"]:
                # Probably ogbl-collab
                pos_e_tensor, neg_e_tensor = (
                    self.edge_split["valid"]["edge"],
                    self.edge_split["valid"]["edge_neg"],
                )
                pos_e = (pos_e_tensor[:, 0], pos_e_tensor[:, 1])
                neg_e = (neg_e_tensor[:, 0], neg_e_tensor[:, 1])
                self._val_edges = pos_e, neg_e

                pos_e_tensor, neg_e_tensor = (
                    self.edge_split["test"]["edge"],
                    self.edge_split["test"]["edge_neg"],
                )
                pos_e = (pos_e_tensor[:, 0], pos_e_tensor[:, 1])
                neg_e = (neg_e_tensor[:, 0], neg_e_tensor[:, 1])
                self._test_edges = pos_e, neg_e
            # delete edge split to save memory
            self.edge_split = None
        else:
            assert self.split_ratio is not None, "Need to specify split_ratio"
            assert self.neg_ratio is not None, "Need to specify neg_ratio"
            ratio = self.split_ratio
            graph = self.dataset[0]
            n = graph.num_edges()
            src, dst = graph.edges()
            src, dst = F.asnumpy(src), F.asnumpy(dst)
            n_train, n_val, n_test = (
                int(n * ratio[0]),
                int(n * ratio[1]),
                int(n * ratio[2]),
            )

            idx = np.random.permutation(n)
            train_pos_idx = idx[:n_train]
            val_pos_idx = idx[n_train : n_train + n_val]
            test_pos_idx = idx[n_train + n_val :]
            neg_src, neg_dst = negative_sample(
                graph, self.neg_ratio * (n_val + n_test)
            )
            neg_n_val, neg_n_test = (
                self.neg_ratio * n_val,
                self.neg_ratio * n_test,
            )
            neg_val_src, neg_val_dst = neg_src[:neg_n_val], neg_dst[:neg_n_val]
            neg_test_src, neg_test_dst = (
                neg_src[neg_n_val:],
                neg_dst[neg_n_val:],
            )
            self._val_edges = (
                F.tensor(src[val_pos_idx]),
                F.tensor(dst[val_pos_idx]),
            ), (F.tensor(neg_val_src), F.tensor(neg_val_dst))
            self._test_edges = (
                F.tensor(src[test_pos_idx]),
                F.tensor(dst[test_pos_idx]),
            ), (F.tensor(neg_test_src), F.tensor(neg_test_dst))
            self._train_graph = create_dgl_graph(
                (src[train_pos_idx], dst[train_pos_idx]),
                num_nodes=self.num_nodes,
            )
            self._train_graph.ndata["feat"] = graph.ndata["feat"]

    def has_cache(self):
        return os.path.isfile(
            os.path.join(self.save_path, "graph_{}.bin".format(self.hash))
        )

    def load(self):
        gs, tensor_dict = utils.load_graphs(
            os.path.join(self.save_path, "graph_{}.bin".format(self.hash))
        )
        self.g = gs[0]
        self._train_graph = self.g
        self._val_edges = (
            tensor_dict["val_pos_src"],
            tensor_dict["val_pos_dst"],
        ), (tensor_dict["val_neg_src"], tensor_dict["val_neg_dst"])
        self._test_edges = (
            tensor_dict["test_pos_src"],
            tensor_dict["test_pos_dst"],
        ), (tensor_dict["test_neg_src"], tensor_dict["test_neg_dst"])

        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "r"
        ) as f:
            info = json.load(f)
            self.split_ratio = info["split_ratio"]
            self.neg_ratio = info["neg_ratio"]

    def save(self):
        tensor_dict = {
            "val_pos_src": self._val_edges[0][0],
            "val_pos_dst": self._val_edges[0][1],
            "val_neg_src": self._val_edges[1][0],
            "val_neg_dst": self._val_edges[1][1],
            "test_pos_src": self._test_edges[0][0],
            "test_pos_dst": self._test_edges[0][1],
            "test_neg_src": self._test_edges[1][0],
            "test_neg_dst": self._test_edges[1][1],
        }
        utils.save_graphs(
            os.path.join(self.save_path, "graph_{}.bin".format(self.hash)),
            [self._train_graph],
            tensor_dict,
        )
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "w"
        ) as f:
            json.dump(
                {"split_ratio": self.split_ratio, "neg_ratio": self.neg_ratio},
                f,
            )

    @property
    def feat_size(self):
        return self._train_graph.ndata["feat"].shape[-1]

    @property
    def train_graph(self):
        return self._train_graph

    @property
    def val_edges(self):
        return self._val_edges

    @property
    def test_edges(self):
        return self._test_edges

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1


class AsGraphPredDataset(DGLDataset):
    """Repurpose a dataset for standard graph property prediction task.

    The created dataset will include data needed for graph property prediction.
    Currently it only supports homogeneous graphs.

    The class converts a given dataset into a new dataset object such that:

      - It stores ``len(dataset)`` graphs.
      - The i-th graph and its label is accessible from ``dataset[i]``.

    The class will generate a train/val/test split if :attr:`split_ratio` is provided.
    The generated split will be cached to disk for fast re-loading. If the provided split
    ratio differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. They must sum to one.

    Attributes
    ----------
    num_tasks : int
        Number of tasks to predict.
    num_classes : int
        Number of classes to predict per task, None for regression datasets.
    train_idx : Tensor
        An 1-D integer tensor of training node IDs.
    val_idx : Tensor
        An 1-D integer tensor of validation node IDs.
    test_idx : Tensor
        An 1-D integer tensor of test node IDs.
    node_feat_size : int
        Input node feature size, None if not applicable.
    edge_feat_size : int
        Input edge feature size, None if not applicable.

    Examples
    --------

    >>> from dgl.data import AsGraphPredDataset
    >>> from ogb.graphproppred import DglGraphPropPredDataset
    >>> dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
    >>> new_dataset = AsGraphPredDataset(dataset)
    >>> print(new_dataset)
    Dataset("ogbg-molhiv-as-graphpred", num_graphs=41127, save_path=...)
    >>> print(len(new_dataset))
    41127
    >>> print(new_dataset[0])
    (Graph(num_nodes=19, num_edges=40,
           ndata_schemes={'feat': Scheme(shape=(9,), dtype=torch.int64)}
           edata_schemes={'feat': Scheme(shape=(3,), dtype=torch.int64)}), tensor([0]))
    """

    def __init__(self, dataset, split_ratio=None, **kwargs):
        self.dataset = dataset
        self.split_ratio = split_ratio
        super().__init__(
            dataset.name + "-as-graphpred",
            hash_key=(split_ratio, dataset.name, "graphpred"),
            **kwargs
        )

    def process(self):
        is_ogb = hasattr(self.dataset, "get_idx_split")
        if self.split_ratio is None:
            if is_ogb:
                split = self.dataset.get_idx_split()
                self.train_idx = split["train"]
                self.val_idx = split["valid"]
                self.test_idx = split["test"]
            else:
                # Handle FakeNewsDataset
                try:
                    self.train_idx = F.nonzero_1d(self.dataset.train_mask)
                    self.val_idx = F.nonzero_1d(self.dataset.val_mask)
                    self.test_idx = F.nonzero_1d(self.dataset.test_mask)
                except:
                    raise DGLError(
                        "The input dataset does not have default train/val/test\
                        split. Please specify split_ratio to generate the split."
                    )
        else:
            if self.verbose:
                print("Generating train/val/test split...")
            train_ratio, val_ratio, _ = self.split_ratio
            num_graphs = len(self.dataset)
            num_train = int(num_graphs * train_ratio)
            num_val = int(num_graphs * val_ratio)

            idx = np.random.permutation(num_graphs)
            self.train_idx = F.tensor(idx[:num_train])
            self.val_idx = F.tensor(idx[num_train : num_train + num_val])
            self.test_idx = F.tensor(idx[num_train + num_val :])

        if hasattr(self.dataset, "num_classes"):
            # GINDataset, MiniGCDataset, FakeNewsDataset, TUDataset,
            # LegacyTUDataset, BA2MotifDataset
            self.num_classes = self.dataset.num_classes
        else:
            # None for multi-label classification and regression
            self.num_classes = None

        if hasattr(self.dataset, "num_tasks"):
            # OGB datasets
            self.num_tasks = self.dataset.num_tasks
        else:
            self.num_tasks = 1

    def has_cache(self):
        return os.path.isfile(
            os.path.join(self.save_path, "info_{}.json".format(self.hash))
        )

    def load(self):
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "r"
        ) as f:
            info = json.load(f)
            if info["split_ratio"] != self.split_ratio:
                raise ValueError(
                    "Provided split ratio is different from the cached file. "
                    "Re-process the dataset."
                )
            self.split_ratio = info["split_ratio"]
            self.num_tasks = info["num_tasks"]
            self.num_classes = info["num_classes"]

        split = np.load(
            os.path.join(self.save_path, "split_{}.npz".format(self.hash))
        )
        self.train_idx = F.zerocopy_from_numpy(split["train_idx"])
        self.val_idx = F.zerocopy_from_numpy(split["val_idx"])
        self.test_idx = F.zerocopy_from_numpy(split["test_idx"])

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "w"
        ) as f:
            json.dump(
                {
                    "split_ratio": self.split_ratio,
                    "num_tasks": self.num_tasks,
                    "num_classes": self.num_classes,
                },
                f,
            )
        np.savez(
            os.path.join(self.save_path, "split_{}.npz".format(self.hash)),
            train_idx=F.zerocopy_to_numpy(self.train_idx),
            val_idx=F.zerocopy_to_numpy(self.val_idx),
            test_idx=F.zerocopy_to_numpy(self.test_idx),
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @property
    def node_feat_size(self):
        g = self[0][0]
        return g.ndata["feat"].shape[-1] if "feat" in g.ndata else None

    @property
    def edge_feat_size(self):
        g = self[0][0]
        return g.edata["feat"].shape[-1] if "feat" in g.edata else None
