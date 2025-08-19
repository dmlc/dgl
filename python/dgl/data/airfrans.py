""" PPIDataset for inductive learning. """
import json
import os

import numpy as np

from dgl import graph
from .. import backend as F
from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, load_graphs, load_info, save_graphs, save_info

class AirfRANSDataset(DGLBuiltinDataset):
    r"""The AirfRANS dataset from the "AirfRANS: High Fidelity Computational
    Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes
    Solutions" paper, consisting of 1,000
    simulations of steady-state aerodynamics over 2D airfoils in a subsonic
    flight regime.

    The different tasks (:obj:`"full"`, :obj:`"scarce"`, :obj:`"reynolds"`,
    :obj:`"aoa"`) define the utilized training and test splits.

    Each simulation is given as a point cloud defined as the nodes of the
    simulation mesh. Each point of a point cloud is described via 5
    features: the inlet velocity (two components in meter per second), the
    distance to the airfoil (one component in meter), and the normals (two
    components in meter, set to :obj:`0` if the point is not on the airfoil).
    Each point is given a target of 4 components for the underyling regression
    task: the velocity (two components in meter per second), the pressure
    divided by the specific mass (one component in meter squared per second
    squared), the turbulent kinematic viscosity (one component in meter squared
    per second).
    Finaly, a boolean is attached to each point to inform if this point lies on
    the airfoil or not.

    Reference: 
    `NeurIPS Paper<https://arxiv.org/abs/2212.07564>`_

    A library for manipulating simulations of the dataset is available `here
    <https://airfrans.readthedocs.io/en/latest/index.html>`_.

    The dataset is released under the `ODbL v1.0 License
    <https://opendatacommons.org/licenses/odbl/1-0/>`_.

    Statistics:

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #labels
        * - 1,000
          - ~180,000
          - 0
          - 5
          - 4

    Notes
    -----
        Data objects contain no edge indices to be agnostic to the simulation
        mesh. You are free to build a graph upon it.

    Parameters
    ----------
    mode : str
        Must be one of ('train', 'test').
        Default: 'train'
    task : str
        The task to study that defines the train and test splits ('full', 'scarce', 'reynolds', 'aoa').
        Default: 'full'
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset.
        Default: False
    verbose : bool
        Whether to print out progress information.
        Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_features : int
        Number of features for each node
    num_labels : int
        Number of labels for each node
    positions : Tensor
        Node positions
    labels : Tensor
        Node labels
    features : Tensor
        Node features
    surfaces : Tensor
        Boolean attached to each node to specify if it lies on the surface of the airfoil

    Examples
    --------
    >>> dataset = AirfRANSDataset(mode='test', task='scarce')
    >>> graph_names = dataset.graph_names
    >>> for g in dataset:
    ....    name = g.name
    ....    pos = g.ndata['pos']
    ....    feat = g.ndata['feat']
    ....    label = g.ndata['label']
    ....    surf = g.ndata['surf']
    ....    # your code here
    >>>
    """

    def __init__(
        self,
        mode="train",
        task="full",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        assert mode in ["train", "test"]
        assert task in ["full", "scarce", "reynolds", "aoa"]
        self.mode = mode
        self.task = task
        _url = _get_dgl_url("dataset/airfrans.zip")
        super(AirfRANSDataset, self).__init__(
            name="airfrans",
            url=_url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        position_file = os.path.join(
            self.save_path, "airfrans_positions.npy"
        )
        label_file = os.path.join(
            self.save_path, "airfrans_labels.npy"
        )
        feat_file = os.path.join(
            self.save_path, "airfrans_feats.npy"
        )
        surf_file = os.path.join(
            self.save_path, "airfrans_surfaces.npy"
        )
        graph_name_file = os.path.join(
            self.save_path, "manifest.json"
        )

        self._positions = np.load(position_file, allow_pickle = True)
        self._labels = np.load(label_file, allow_pickle = True)
        self._feats = np.load(feat_file, allow_pickle = True)
        self._surfaces = np.load(surf_file, allow_pickle = True)

        with open(graph_name_file, 'r') as f:
            self._graph_names = json.load(f)

        total = self._graph_names['full_train'] + self._graph_names['full_test']
        partial = set(self._graph_names[f'{self.task}_{self.mode}'])
        self.graphs = []
        for k, s in enumerate(total):
            if s in partial:
                g = graph(([], []), num_nodes=self._positions[k].shape[0])
                g.ndata["pos"] = F.tensor(
                    self._positions[k], dtype=F.data_type_dict["float32"]
                )
                g.ndata["feat"] = F.tensor(
                    self._feats[k], dtype=F.data_type_dict["float32"]
                )
                g.ndata["label"] = F.tensor(
                    self._labels[k], dtype=F.data_type_dict["float32"]
                )
                g.ndata["surf"] = F.tensor(
                    self._surfaces[k], dtype=F.data_type_dict["float32"]
                )

                self.graphs.append(g)

    @property
    def graph_list_path(self):
        return os.path.join(
            self.save_path, "{}_dgl_graph_list.bin".format(self.mode)
        )

    @property
    def info_path(self):
        return os.path.join(self.save_path, "{}_info.pkl".format(self.mode))

    def has_cache(self):
        return (
            os.path.exists(self.graph_list_path)
            and os.path.exists(self.g_path)
            and os.path.exists(self.info_path)
        )

    def save(self):
        save_graphs(self.graph_list_path, self.graphs)
        save_info(
            self.info_path, {"positions": self._positions, "labels": self._labels, "feats": self._feats, "surfaces": self._surfaces}
        )

    def load(self):
        self.graphs = load_graphs(self.graph_list_path)
        info = load_info(self.info_path)
        self._positions = info["positions"]
        self._labels = info["labels"]
        self._feats = info["feats"]
        self._surfaces = info["surfaces"]

    @property
    def num_features(self):
        return 5
    
    @property
    def num_labels(self):
        return 4
    
    @property
    def graph_names(self):
        return self._graph_names

    def __len__(self):
        """Return number of samples in this dataset."""
        return len(self.graphs)

    def __getitem__(self, item):
        """Get the item^th sample.

        Parameters
        ---------
        item : int
            The sample index.

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features and node labels.

            - ``ndata['pos']``: node positions
            - ``ndata['feat']``: node features
            - ``ndata['label']``: node labels
            - ``ndata['surf']``: node surfaces boolean (``True`` if the node lies on the surface of the airfoil)
        """
        if self._transform is None:
            return self.graphs[item]
        else:
            return self._transform(self.graphs[item])
