"""Datasets used in How Powerful Are Graph Neural Networks?
(chen jun)
Datasets include:
MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, PTC, REDDITBINARY, REDDITMULTI5K
https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip
"""

import os

import numpy as np

from .. import backend as F
from ..convert import graph as dgl_graph
from ..utils import retry_method_with_fix
from .dgl_dataset import DGLBuiltinDataset
from .utils import (
    download,
    extract_archive,
    load_graphs,
    load_info,
    loadtxt,
    save_graphs,
    save_info,
)


class GINDataset(DGLBuiltinDataset):
    """Dataset Class for `How Powerful Are Graph Neural Networks? <https://arxiv.org/abs/1810.00826>`_.

    This is adapted from `<https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip>`_.

    The class provides an interface for nine datasets used in the paper along with the paper-specific
    settings. The datasets are ``'MUTAG'``, ``'COLLAB'``, ``'IMDBBINARY'``, ``'IMDBMULTI'``,
    ``'NCI1'``, ``'PROTEINS'``, ``'PTC'``, ``'REDDITBINARY'``, ``'REDDITMULTI5K'``.

    If ``degree_as_nlabel`` is set to ``False``, then ``ndata['label']`` stores the provided node label,
    otherwise ``ndata['label']`` stores the node in-degrees.

    For graphs that have node attributes, ``ndata['attr']`` stores the node attributes.
    For graphs that have no attribute, ``ndata['attr']`` stores the corresponding one-hot encoding
    of ``ndata['label']``.

    Parameters
    ---------
    name: str
        dataset name, one of
        (``'MUTAG'``, ``'COLLAB'``, \
        ``'IMDBBINARY'``, ``'IMDBMULTI'``, \
        ``'NCI1'``, ``'PROTEINS'``, ``'PTC'``, \
        ``'REDDITBINARY'``, ``'REDDITMULTI5K'``)
    self_loop: bool
        add self to self edge if true
    degree_as_nlabel: bool
        take node degree as label and feature if true
    transform: callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_classes : int
        Number of classes for multiclass classification

    Examples
    --------
    >>> data = GINDataset(name='MUTAG', self_loop=False)

    The dataset instance is an iterable

    >>> len(data)
    188
    >>> g, label = data[128]
    >>> g
    Graph(num_nodes=13, num_edges=26,
          ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'attr': Scheme(shape=(7,), dtype=torch.float32)}
          edata_schemes={})
    >>> label
    tensor(1)

    Batch the graphs and labels for mini-batch training

    >>> graphs, labels = zip(*[data[i] for i in range(16)])
    >>> batched_graphs = dgl.batch(graphs)
    >>> batched_labels = torch.tensor(labels)
    >>> batched_graphs
    Graph(num_nodes=330, num_edges=748,
          ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'attr': Scheme(shape=(7,), dtype=torch.float32)}
          edata_schemes={})
    """

    def __init__(
        self,
        name,
        self_loop,
        degree_as_nlabel=False,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        self._name = name  # MUTAG
        gin_url = "https://raw.githubusercontent.com/weihua916/powerful-gnns/master/dataset.zip"
        self.ds_name = "nig"

        self.self_loop = self_loop
        self.graphs = []
        self.labels = []

        # relabel
        self.glabel_dict = {}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0

        # flags
        self.degree_as_nlabel = degree_as_nlabel
        self.nattrs_flag = False
        self.nlabels_flag = False

        super(GINDataset, self).__init__(
            name=name,
            url=gin_url,
            hash_key=(name, self_loop, degree_as_nlabel),
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    @property
    def raw_path(self):
        return os.path.join(self.raw_dir, "GINDataset")

    def download(self):
        r"""Automatically download data and extract it."""
        zip_file_path = os.path.join(self.raw_dir, "GINDataset.zip")
        download(self.url, path=zip_file_path)
        extract_archive(zip_file_path, self.raw_path)

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

    def _file_path(self):
        return os.path.join(
            self.raw_dir,
            "GINDataset",
            "dataset",
            self.name,
            "{}.txt".format(self.name),
        )

    def process(self):
        """Loads input dataset from dataset/NAME/NAME.txt file"""
        if self.verbose:
            print("loading data...")
        self.file = self._file_path()
        with open(self.file, "r") as f:
            # line_1 == N, total number of graphs
            self.N = int(f.readline().strip())

            for i in range(self.N):
                if (i + 1) % 10 == 0 and self.verbose is True:
                    print("processing graph {}...".format(i + 1))

                grow = f.readline().strip().split()
                # line_2 == [n_nodes, l] is equal to
                # [node number of a graph, class label of a graph]
                n_nodes, glabel = [int(w) for w in grow]

                # relabel graphs
                if glabel not in self.glabel_dict:
                    mapped = len(self.glabel_dict)
                    self.glabel_dict[glabel] = mapped

                self.labels.append(self.glabel_dict[glabel])

                g = dgl_graph(([], []))
                g.add_nodes(n_nodes)

                nlabels = []  # node labels
                nattrs = []  # node attributes if it has
                m_edges = 0

                for j in range(n_nodes):
                    nrow = f.readline().strip().split()

                    # handle edges and attributes(if has)
                    tmp = int(nrow[1]) + 2  # tmp == 2 + #edges
                    if tmp == len(nrow):
                        # no node attributes
                        nrow = [int(w) for w in nrow]
                    elif tmp > len(nrow):
                        nrow = [int(w) for w in nrow[:tmp]]
                        nattr = [float(w) for w in nrow[tmp:]]
                        nattrs.append(nattr)
                    else:
                        raise Exception("edge number is incorrect!")

                    # relabel nodes if it has labels
                    # if it doesn't have node labels, then every nrow[0]==0
                    if not nrow[0] in self.nlabel_dict:
                        mapped = len(self.nlabel_dict)
                        self.nlabel_dict[nrow[0]] = mapped

                    nlabels.append(self.nlabel_dict[nrow[0]])

                    m_edges += nrow[1]
                    g.add_edges(j, nrow[2:])

                    # add self loop
                    if self.self_loop:
                        m_edges += 1
                        g.add_edges(j, j)

                    if (j + 1) % 10 == 0 and self.verbose is True:
                        print(
                            "processing node {} of graph {}...".format(
                                j + 1, i + 1
                            )
                        )
                        print("this node has {} edgs.".format(nrow[1]))

                if nattrs != []:
                    nattrs = np.stack(nattrs)
                    g.ndata["attr"] = F.tensor(nattrs, F.float32)
                    self.nattrs_flag = True

                g.ndata["label"] = F.tensor(nlabels)
                if len(self.nlabel_dict) > 1:
                    self.nlabels_flag = True

                assert g.num_nodes() == n_nodes

                # update statistics of graphs
                self.n += n_nodes
                self.m += m_edges

                self.graphs.append(g)

        self.labels = F.tensor(self.labels)
        # if no attr
        if not self.nattrs_flag:
            if self.verbose:
                print("there are no node features in this dataset!")
            # generate node attr by node degree
            if self.degree_as_nlabel:
                if self.verbose:
                    print("generate node features by node degree...")
                for g in self.graphs:
                    # actually this label shouldn't be updated
                    # in case users want to keep it
                    # but usually no features means no labels, fine.
                    g.ndata["label"] = g.in_degrees()
                    # extracting unique node labels

            # in case the labels/degrees are not continuous number
            nlabel_set = set([])
            for g in self.graphs:
                nlabel_set = nlabel_set.union(
                    set([F.as_scalar(nl) for nl in g.ndata["label"]])
                )
            nlabel_set = list(nlabel_set)
            is_label_valid = all(
                [label in self.nlabel_dict for label in nlabel_set]
            )
            if (
                is_label_valid
                and len(nlabel_set) == np.max(nlabel_set) + 1
                and np.min(nlabel_set) == 0
            ):
                # Note this is different from the author's implementation. In weihua916's implementation,
                # the labels are relabeled anyway. But here we didn't relabel it if the labels are contiguous
                # to make it consistent with the original dataset
                label2idx = self.nlabel_dict
            else:
                label2idx = {nlabel_set[i]: i for i in range(len(nlabel_set))}
            # generate node attr by node label
            for g in self.graphs:
                attr = np.zeros((g.num_nodes(), len(label2idx)))
                attr[
                    range(g.num_nodes()),
                    [
                        label2idx[nl]
                        for nl in F.asnumpy(g.ndata["label"]).tolist()
                    ],
                ] = 1
                g.ndata["attr"] = F.tensor(attr, F.float32)

        # after load, get the #classes and #dim
        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata["attr"][0])

        if self.verbose:
            print("Done.")
            print(
                """
                -------- Data Statistics --------'
                #Graphs: %d
                #Graph Classes: %d
                #Nodes: %d
                #Node Classes: %d
                #Node Features Dim: %d
                #Edges: %d
                #Edge Classes: %d
                Avg. of #Nodes: %.2f
                Avg. of #Edges: %.2f
                Graph Relabeled: %s
                Node Relabeled: %s
                Degree Relabeled(If degree_as_nlabel=True): %s \n """
                % (
                    self.N,
                    self.gclasses,
                    self.n,
                    self.nclasses,
                    self.dim_nfeats,
                    self.m,
                    self.eclasses,
                    self.n / self.N,
                    self.m / self.N,
                    self.glabel_dict,
                    self.nlabel_dict,
                    self.ndegree_dict,
                )
            )

    def save(self):
        label_dict = {"labels": self.labels}
        info_dict = {
            "N": self.N,
            "n": self.n,
            "m": self.m,
            "self_loop": self.self_loop,
            "gclasses": self.gclasses,
            "nclasses": self.nclasses,
            "eclasses": self.eclasses,
            "dim_nfeats": self.dim_nfeats,
            "degree_as_nlabel": self.degree_as_nlabel,
            "glabel_dict": self.glabel_dict,
            "nlabel_dict": self.nlabel_dict,
            "elabel_dict": self.elabel_dict,
            "ndegree_dict": self.ndegree_dict,
        }
        save_graphs(str(self.graph_path), self.graphs, label_dict)
        save_info(str(self.info_path), info_dict)

    def load(self):
        graphs, label_dict = load_graphs(str(self.graph_path))
        info_dict = load_info(str(self.info_path))

        self.graphs = graphs
        self.labels = label_dict["labels"]

        self.N = info_dict["N"]
        self.n = info_dict["n"]
        self.m = info_dict["m"]
        self.self_loop = info_dict["self_loop"]
        self.gclasses = info_dict["gclasses"]
        self.nclasses = info_dict["nclasses"]
        self.eclasses = info_dict["eclasses"]
        self.dim_nfeats = info_dict["dim_nfeats"]
        self.glabel_dict = info_dict["glabel_dict"]
        self.nlabel_dict = info_dict["nlabel_dict"]
        self.elabel_dict = info_dict["elabel_dict"]
        self.ndegree_dict = info_dict["ndegree_dict"]
        self.degree_as_nlabel = info_dict["degree_as_nlabel"]

    @property
    def graph_path(self):
        return os.path.join(
            self.save_path, "gin_{}_{}.bin".format(self.name, self.hash)
        )

    @property
    def info_path(self):
        return os.path.join(
            self.save_path, "gin_{}_{}.pkl".format(self.name, self.hash)
        )

    def has_cache(self):
        if os.path.exists(self.graph_path) and os.path.exists(self.info_path):
            return True
        return False

    @property
    def num_classes(self):
        return self.gclasses
