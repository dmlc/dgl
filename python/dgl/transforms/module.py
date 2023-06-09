##
#   Copyright 2019-2021 Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""Modules for transform"""
# pylint: disable= no-member, arguments-differ, invalid-name, missing-function-docstring

from scipy.linalg import expm

from .. import backend as F, convert, function as fn, utils
from ..base import dgl_warning, DGLError
from . import functional

try:
    import torch
    from torch.distributions import Bernoulli
except ImportError:
    pass

__all__ = [
    "BaseTransform",
    "RowFeatNormalizer",
    "FeatMask",
    "RandomWalkPE",
    "LaplacianPE",
    "LapPE",
    "AddSelfLoop",
    "RemoveSelfLoop",
    "AddReverse",
    "ToSimple",
    "LineGraph",
    "KHopGraph",
    "AddMetaPaths",
    "Compose",
    "GCNNorm",
    "PPR",
    "HeatKernel",
    "GDC",
    "NodeShuffle",
    "DropNode",
    "DropEdge",
    "AddEdge",
    "SIGNDiffusion",
    "ToLevi",
    "SVDPE",
]


def update_graph_structure(g, data_dict, copy_edata=True):
    r"""Update the structure of a graph.

    Parameters
    ----------
    g : DGLGraph
        The graph to update.
    data_dict : graph data
        The dictionary data for constructing a heterogeneous graph.
    copy_edata : bool
        If True, it will copy the edge features to the updated graph.

    Returns
    -------
    DGLGraph
        The updated graph.
    """
    device = g.device
    idtype = g.idtype
    num_nodes_dict = dict()

    for ntype in g.ntypes:
        num_nodes_dict[ntype] = g.num_nodes(ntype)

    new_g = convert.heterograph(
        data_dict, num_nodes_dict=num_nodes_dict, idtype=idtype, device=device
    )

    # Copy features
    for ntype in g.ntypes:
        for key, feat in g.nodes[ntype].data.items():
            new_g.nodes[ntype].data[key] = feat

    if copy_edata:
        for c_etype in g.canonical_etypes:
            for key, feat in g.edges[c_etype].data.items():
                new_g.edges[c_etype].data[key] = feat

    return new_g


class BaseTransform:
    r"""An abstract class for writing transforms."""

    def __call__(self, g):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RowFeatNormalizer(BaseTransform):
    r"""
    Row-normalizes the features given in ``node_feat_names`` and ``edge_feat_names``.

    The row normalization formular is:

    .. math::
      x = \frac{x}{\sum_i x_i}

    where :math:`x` denotes a row of the feature tensor.

    Parameters
    ----------
    subtract_min: bool
        If True, the minimum value of whole feature tensor will be subtracted before normalization.
        Default: False.
        Subtraction will make all values non-negative. If all values are negative, after
        normalisation, the sum of each row of the feature tensor will be 1.
    node_feat_names : list[str], optional
        The names of the node feature tensors to be row-normalized. Default: `None`, which will
        not normalize any node feature tensor.
    edge_feat_names : list[str], optional
        The names of the edge feature tensors to be row-normalized. Default: `None`, which will
        not normalize any edge feature tensor.

    Example
    -------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> from dgl import RowFeatNormalizer

    Case1: Row normalize features of a homogeneous graph.

    >>> transform = RowFeatNormalizer(subtract_min=True,
    ...                               node_feat_names=['h'], edge_feat_names=['w'])
    >>> g = dgl.rand_graph(5, 20)
    >>> g.ndata['h'] = torch.randn((g.num_nodes(), 5))
    >>> g.edata['w'] = torch.randn((g.num_edges(), 5))
    >>> g = transform(g)
    >>> print(g.ndata['h'].sum(1))
    tensor([1., 1., 1., 1., 1.])
    >>> print(g.edata['w'].sum(1))
    tensor([1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1.])

    Case2: Row normalize features of a heterogeneous graph.

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (torch.tensor([1, 2]), torch.tensor([3, 4])),
    ...     ('player', 'plays', 'game'): (torch.tensor([2, 2]), torch.tensor([1, 1]))
    ... })
    >>> g.ndata['h'] = {'game': torch.randn(2, 5), 'player': torch.randn(3, 5)}
    >>> g.edata['w'] = {
    ...     ('user', 'follows', 'user'): torch.randn(2, 5),
    ...     ('player', 'plays', 'game'): torch.randn(2, 5)
    ... }
    >>> g = transform(g)
    >>> print(g.ndata['h']['game'].sum(1), g.ndata['h']['player'].sum(1))
    tensor([1., 1.]) tensor([1., 1., 1.])
    >>> print(g.edata['w'][('user', 'follows', 'user')].sum(1),
    ...     g.edata['w'][('player', 'plays', 'game')].sum(1))
    tensor([1., 1.]) tensor([1., 1.])
    """

    def __init__(
        self, subtract_min=False, node_feat_names=None, edge_feat_names=None
    ):
        self.node_feat_names = (
            [] if node_feat_names is None else node_feat_names
        )
        self.edge_feat_names = (
            [] if edge_feat_names is None else edge_feat_names
        )
        self.subtract_min = subtract_min

    def row_normalize(self, feat):
        r"""

        Description
        -----------
        Row-normalize the given feature.

        Parameters
        ----------
        feat : Tensor
            The feature to be normalized.

        Returns
        -------
        Tensor
            The normalized feature.
        """
        if self.subtract_min:
            feat = feat - feat.min()
        feat.div_(feat.sum(dim=-1, keepdim=True).clamp_(min=1.0))
        return feat

    def __call__(self, g):
        for node_feat_name in self.node_feat_names:
            if isinstance(g.ndata[node_feat_name], torch.Tensor):
                g.ndata[node_feat_name] = self.row_normalize(
                    g.ndata[node_feat_name]
                )
            else:
                for ntype in g.ndata[node_feat_name].keys():
                    g.nodes[ntype].data[node_feat_name] = self.row_normalize(
                        g.nodes[ntype].data[node_feat_name]
                    )

        for edge_feat_name in self.edge_feat_names:
            if isinstance(g.edata[edge_feat_name], torch.Tensor):
                g.edata[edge_feat_name] = self.row_normalize(
                    g.edata[edge_feat_name]
                )
            else:
                for etype in g.edata[edge_feat_name].keys():
                    g.edges[etype].data[edge_feat_name] = self.row_normalize(
                        g.edges[etype].data[edge_feat_name]
                    )

        return g


class FeatMask(BaseTransform):
    r"""Randomly mask columns of the node and edge feature tensors, as described in `Graph
    Contrastive Learning with Augmentations <https://arxiv.org/abs/2010.13902>`__.

    Parameters
    ----------
    p : float, optional
        Probability of masking a column of a feature tensor. Default: `0.5`.
    node_feat_names : list[str], optional
        The names of the node feature tensors to be masked. Default: `None`, which will
        not mask any node feature tensor.
    edge_feat_names : list[str], optional
        The names of the edge features to be masked. Default: `None`, which will not mask
        any edge feature tensor.

    Example
    -------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> from dgl import FeatMask

    Case1 : Mask node and edge feature tensors of a homogeneous graph.

    >>> transform = FeatMask(node_feat_names=['h'], edge_feat_names=['w'])
    >>> g = dgl.rand_graph(5, 10)
    >>> g.ndata['h'] = torch.ones((g.num_nodes(), 10))
    >>> g.edata['w'] = torch.ones((g.num_edges(), 10))

    >>> g = transform(g)
    >>> print(g.ndata['h'])
    tensor([[0., 0., 1., 1., 0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 0., 0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 0., 0., 1., 1., 1., 0.]])
    >>> print(g.edata['w'])
    tensor([[1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0., 1.]])

    Case2 : Mask node and edge feature tensors of a heterogeneous graph.

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (torch.tensor([1, 2]), torch.tensor([3, 4])),
    ...     ('player', 'plays', 'game'): (torch.tensor([2, 2]), torch.tensor([1, 1]))
    ... })
    >>> g.ndata['h'] = {'game': torch.ones(2, 5), 'player': torch.ones(3, 5)}
    >>> g.edata['w'] = {('user', 'follows', 'user'): torch.ones(2, 5)}
    >>> print(g.ndata['h']['game'])
    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])
    >>> print(g.edata['w'][('user', 'follows', 'user')])
    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])
    >>> g = transform(g)
    >>> print(g.ndata['h']['game'])
    tensor([[1., 1., 0., 1., 0.],
            [1., 1., 0., 1., 0.]])
    >>> print(g.edata['w'][('user', 'follows', 'user')])
    tensor([[0., 1., 0., 1., 0.],
            [0., 1., 0., 1., 0.]])
    """

    def __init__(self, p=0.5, node_feat_names=None, edge_feat_names=None):
        self.p = p
        self.node_feat_names = (
            [] if node_feat_names is None else node_feat_names
        )
        self.edge_feat_names = (
            [] if edge_feat_names is None else edge_feat_names
        )
        self.dist = Bernoulli(p)

    def __call__(self, g):
        # Fast path
        if self.p == 0:
            return g

        for node_feat_name in self.node_feat_names:
            if isinstance(g.ndata[node_feat_name], torch.Tensor):
                feat_mask = self.dist.sample(
                    torch.Size(
                        [
                            g.ndata[node_feat_name].shape[-1],
                        ]
                    )
                )
                g.ndata[node_feat_name][:, feat_mask.bool().to(g.device)] = 0

            else:
                for ntype in g.ndata[node_feat_name].keys():
                    mask_shape = g.ndata[node_feat_name][ntype].shape[-1]
                    feat_mask = self.dist.sample(
                        torch.Size(
                            [
                                mask_shape,
                            ]
                        )
                    )
                    g.ndata[node_feat_name][ntype][
                        :, feat_mask.bool().to(g.device)
                    ] = 0

        for edge_feat_name in self.edge_feat_names:
            if isinstance(g.edata[edge_feat_name], torch.Tensor):
                feat_mask = self.dist.sample(
                    torch.Size(
                        [
                            g.edata[edge_feat_name].shape[-1],
                        ]
                    )
                )
                g.edata[edge_feat_name][:, feat_mask.bool().to(g.device)] = 0

            else:
                for etype in g.edata[edge_feat_name].keys():
                    mask_shape = g.edata[edge_feat_name][etype].shape[-1]
                    feat_mask = self.dist.sample(
                        torch.Size(
                            [
                                mask_shape,
                            ]
                        )
                    )
                    g.edata[edge_feat_name][etype][
                        :, feat_mask.bool().to(g.device)
                    ] = 0
        return g


class RandomWalkPE(BaseTransform):
    r"""Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This module only works for homogeneous graphs.

    Parameters
    ----------
    k : int
        Number of random walk steps. The paper found the best value to be 16 and 20
        for two experiments.
    feat_name : str, optional
        Name to store the computed positional encodings in ndata.
    eweight_name : str, optional
        Name to retrieve the edge weights. Default: None, not using the edge weights.

    Example
    -------

    >>> import dgl
    >>> from dgl import RandomWalkPE

    >>> transform = RandomWalkPE(k=2)
    >>> g = dgl.graph(([0, 1, 1], [1, 1, 0]))
    >>> g = transform(g)
    >>> print(g.ndata['PE'])
    tensor([[0.0000, 0.5000],
            [0.5000, 0.7500]])
    """

    def __init__(self, k, feat_name="PE", eweight_name=None):
        self.k = k
        self.feat_name = feat_name
        self.eweight_name = eweight_name

    def __call__(self, g):
        PE = functional.random_walk_pe(
            g, k=self.k, eweight_name=self.eweight_name
        )
        g.ndata[self.feat_name] = F.copy_to(PE, g.device)

        return g


class LapPE(BaseTransform):
    r"""Laplacian Positional Encoding, as introduced in
    `Benchmarking Graph Neural Networks
    <https://arxiv.org/abs/2003.00982>`__

    This module only works for homogeneous bidirected graphs.

    Parameters
    ----------
    k : int
        Number of smallest non-trivial eigenvectors to use for positional encoding.
    feat_name : str, optional
        Name to store the computed positional encodings in ndata.
    eigval_name : str, optional
        If None, store laplacian eigenvectors only. Otherwise, it's the name to
        store corresponding laplacian eigenvalues in ndata. Default: None.
    padding : bool, optional
        If False, raise an exception when k>=n.
        Otherwise, add zero paddings in the end of eigenvectors and 'nan'
        paddings in the end of eigenvalues when k>=n. Default: False.
        n is the number of nodes in the given graph.

    Example
    -------
    >>> import dgl
    >>> from dgl import LapPE
    >>> transform1 = LapPE(k=3)
    >>> transform2 = LapPE(k=5, padding=True)
    >>> transform3 = LapPE(k=5, feat_name='eigvec', eigval_name='eigval', padding=True)
    >>> g = dgl.graph(([0,1,2,3,4,2,3,1,4,0], [2,3,1,4,0,0,1,2,3,4]))
    >>> g1 = transform1(g)
    >>> print(g1.ndata['PE'])
    tensor([[ 0.6325,  0.1039,  0.3489],
            [-0.5117,  0.2826,  0.6095],
            [ 0.1954,  0.6254, -0.5923],
            [-0.5117, -0.4508, -0.3938],
            [ 0.1954, -0.5612,  0.0278]])
    >>> g2 = transform2(g)
    >>> print(g2.ndata['PE'])
    tensor([[-0.6325, -0.1039,  0.3489, -0.2530,  0.0000],
            [ 0.5117, -0.2826,  0.6095,  0.4731,  0.0000],
            [-0.1954, -0.6254, -0.5923, -0.1361,  0.0000],
            [ 0.5117,  0.4508, -0.3938, -0.6295,  0.0000],
            [-0.1954,  0.5612,  0.0278,  0.5454,  0.0000]])
    >>> g3 = transform3(g)
    >>> print(g3.ndata['eigval'])
    tensor([[0.6910, 0.6910, 1.8090, 1.8090,    nan],
            [0.6910, 0.6910, 1.8090, 1.8090,    nan],
            [0.6910, 0.6910, 1.8090, 1.8090,    nan],
            [0.6910, 0.6910, 1.8090, 1.8090,    nan],
            [0.6910, 0.6910, 1.8090, 1.8090,    nan]])
    >>> print(g3.ndata['eigvec'])
    tensor([[ 0.6325, -0.1039,  0.3489,  0.2530,  0.0000],
            [-0.5117, -0.2826,  0.6095, -0.4731,  0.0000],
            [ 0.1954, -0.6254, -0.5923,  0.1361,  0.0000],
            [-0.5117,  0.4508, -0.3938,  0.6295,  0.0000],
            [ 0.1954,  0.5612,  0.0278, -0.5454,  0.0000]])
    """

    def __init__(self, k, feat_name="PE", eigval_name=None, padding=False):
        self.k = k
        self.feat_name = feat_name
        self.eigval_name = eigval_name
        self.padding = padding

    def __call__(self, g):
        if self.eigval_name:
            PE, eigval = functional.lap_pe(
                g, k=self.k, padding=self.padding, return_eigval=True
            )
            eigval = F.repeat(F.reshape(eigval, [1, -1]), g.num_nodes(), dim=0)
            g.ndata[self.eigval_name] = F.copy_to(eigval, g.device)
        else:
            PE = functional.lap_pe(g, k=self.k, padding=self.padding)
        g.ndata[self.feat_name] = F.copy_to(PE, g.device)

        return g


class LaplacianPE(LapPE):
    r"""Alias of `LapPE`."""

    def __init__(self, k, feat_name="PE", eigval_name=None, padding=False):
        super().__init__(k, feat_name, eigval_name, padding)
        dgl_warning("LaplacianPE will be deprecated. Use LapPE please.")


class AddSelfLoop(BaseTransform):
    r"""Add self-loops for each node in the graph and return a new graph.

    For heterogeneous graphs, self-loops are added only for edge types with same
    source and destination node types.

    Parameters
    ----------
    allow_duplicate : bool, optional
        If False, it will first remove self-loops to prevent duplicate self-loops.
    new_etypes : bool, optional
        If True, it will add an edge type 'self' per node type, which holds self-loops.
    edge_feat_names : list[str], optional
        The names of the self-loop features to apply `fill_data`. If None, it
        will apply `fill_data` to all self-loop features. Default: None.
    fill_data : int, float or str, optional
        The value to fill the self-loop features. Default: 1.

        * If ``fill_data`` is ``int`` or ``float``, self-loop features will be directly given by
          ``fill_data``.
        * if ``fill_data`` is ``str``, self-loop features will be generated by aggregating the
          features of the incoming edges of the corresponding nodes. The supported aggregation are:
          ``'mean'``, ``'sum'``, ``'max'``, ``'min'``.

    Example
    -------

    >>> import dgl
    >>> from dgl import AddSelfLoop

    Case1: Add self-loops for a homogeneous graph

    >>> transform = AddSelfLoop(fill_data='sum')
    >>> g = dgl.graph(([0, 0, 2], [2, 1, 0]))
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([0, 0, 2, 0, 1, 2]), tensor([2, 1, 0, 0, 1, 2]))
    >>> print(new_g.edata('he'))
    tensor([[0.],
            [1.],
            [2.],
            [2.],
            [1.],
            [0.]])

    Case2: Add self-loops for a heterogeneous graph

    >>> transform = AddSelfLoop(fill_data='sum')
    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (torch.tensor([1, 2]),
    ...                                   torch.tensor([0, 1])),
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]),
    ...                                 torch.tensor([0, 1]))})
    >>> g.edata['feat'] = {('user', 'follows', 'user'): torch.randn(2, 5),
    ...                    ('user', 'plays', 'game'): torch.randn(2, 5)}
    >>> g.edata['feat1'] = {('user', 'follows', 'user'): torch.randn(2, 15),
    ...                     ('user', 'plays', 'game'): torch.randn(2, 15)}
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype='plays'))
    (tensor([0, 1]), tensor([0, 1]))
    >>> print(new_g.edges(etype='follows'))
    (tensor([1, 2, 0, 1, 2]), tensor([0, 1, 0, 1, 2]))
    >>> print(new_g.edata['feat'][('user', 'follows', 'user')].shape)
    torch.Size([5, 5])

    Case3: Add self-etypes for a heterogeneous graph

    >>> transform = AddSelfLoop(new_etypes=True)
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype='follows'))
    (tensor([1, 2, 0, 1, 2]), tensor([0, 1, 0, 1, 2]))
    >>> print(new_g.edges(etype=('game', 'self', 'game')))
    (tensor([0, 1]), tensor([0, 1]))
    """

    def __init__(
        self,
        allow_duplicate=False,
        new_etypes=False,
        edge_feat_names=None,
        fill_data=1.0,
    ):
        self.allow_duplicate = allow_duplicate
        self.new_etypes = new_etypes
        self.edge_feat_names = edge_feat_names
        self.fill_data = fill_data

    def transform_etype(self, c_etype, g):
        r"""

        Description
        -----------
        Transform the graph corresponding to a canonical edge type.

        Parameters
        ----------
        c_etype : tuple of str
            A canonical edge type.
        g : DGLGraph
            The graph.

        Returns
        -------
        DGLGraph
            The transformed graph.
        """
        utype, _, vtype = c_etype
        if utype != vtype:
            return g

        if not self.allow_duplicate:
            g = functional.remove_self_loop(g, etype=c_etype)
        return functional.add_self_loop(
            g,
            edge_feat_names=self.edge_feat_names,
            fill_data=self.fill_data,
            etype=c_etype,
        )

    def __call__(self, g):
        for c_etype in g.canonical_etypes:
            g = self.transform_etype(c_etype, g)

        if self.new_etypes:
            device = g.device
            idtype = g.idtype
            data_dict = dict()

            # Add self etypes
            for ntype in g.ntypes:
                nids = F.arange(0, g.num_nodes(ntype), idtype, device)
                data_dict[(ntype, "self", ntype)] = (nids, nids)

            # Copy edges
            for c_etype in g.canonical_etypes:
                data_dict[c_etype] = g.edges(etype=c_etype)

            g = update_graph_structure(g, data_dict)

        return g


class RemoveSelfLoop(BaseTransform):
    r"""Remove self-loops for each node in the graph and return a new graph.

    For heterogeneous graphs, this operation only applies to edge types with same
    source and destination node types.

    Example
    -------

    >>> import dgl
    >>> from dgl import RemoveSelfLoop

    Case1: Remove self-loops for a homogeneous graph

    >>> transform = RemoveSelfLoop()
    >>> g = dgl.graph(([1, 1], [1, 2]))
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([1]), tensor([2]))

    Case2: Remove self-loops for a heterogeneous graph

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1], [1, 1]),
    ...     ('user', 'follows', 'user'): ([1, 2], [2, 2])
    ... })
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype='plays'))
    (tensor([0, 1]), tensor([1, 1]))
    >>> print(new_g.edges(etype='follows'))
    (tensor([1]), tensor([2]))
    """

    def transform_etype(self, c_etype, g):
        r"""Transform the graph corresponding to a canonical edge type.

        Parameters
        ----------
        c_etype : tuple of str
            A canonical edge type.
        g : DGLGraph
            The graph.

        Returns
        -------
        DGLGraph
            The transformed graph.
        """
        utype, _, vtype = c_etype
        if utype == vtype:
            g = functional.remove_self_loop(g, etype=c_etype)
        return g

    def __call__(self, g):
        for c_etype in g.canonical_etypes:
            g = self.transform_etype(c_etype, g)
        return g


class AddReverse(BaseTransform):
    r"""Add a reverse edge :math:`(i,j)` for each edge :math:`(j,i)` in the input graph and
    return a new graph.

    For a heterogeneous graph, it adds a "reverse" edge type for each edge type
    to hold the reverse edges. For example, for a canonical edge type ('A', 'r', 'B'),
    it adds a canonical edge type ('B', 'rev_r', 'A').

    Parameters
    ----------
    copy_edata : bool, optional
        If True, the features of the reverse edges will be identical to the original ones.
    sym_new_etype : bool, optional
        If False, it will not add a reverse edge type if the source and destination node type
        in a canonical edge type are identical. Instead, it will directly add edges to the
        original edge type.

    Example
    -------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> from dgl import AddReverse

    Case1: Add reverse edges for a homogeneous graph

    >>> transform = AddReverse()
    >>> g = dgl.graph(([0], [1]))
    >>> g.edata['w'] = torch.ones(1, 2)
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([0, 1]), tensor([1, 0]))
    >>> print(new_g.edata['w'])
    tensor([[1., 1.],
            [0., 0.]])

    Case2: Add reverse edges for a homogeneous graph and copy edata

    >>> transform = AddReverse(copy_edata=True)
    >>> new_g = transform(g)
    >>> print(new_g.edata['w'])
    tensor([[1., 1.],
            [1., 1.]])

    Case3: Add reverse edges for a heterogeneous graph

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0, 1], [1, 1]),
    ...     ('user', 'follows', 'user'): ([1, 2], [2, 2])
    ... })
    >>> new_g = transform(g)
    >>> print(new_g.canonical_etypes)
    [('game', 'rev_plays', 'user'), ('user', 'follows', 'user'), ('user', 'plays', 'game')]
    >>> print(new_g.edges(etype='rev_plays'))
    (tensor([1, 1]), tensor([0, 1]))
    >>> print(new_g.edges(etype='follows'))
    (tensor([1, 2, 2, 2]), tensor([2, 2, 1, 2]))
    """

    def __init__(self, copy_edata=False, sym_new_etype=False):
        self.copy_edata = copy_edata
        self.sym_new_etype = sym_new_etype

    def transform_symmetric_etype(self, c_etype, g, data_dict):
        r"""Transform the graph corresponding to a symmetric canonical edge type.

        Parameters
        ----------
        c_etype : tuple of str
            A canonical edge type.
        g : DGLGraph
            The graph.
        data_dict : dict
            The edge data to update.
        """
        if self.sym_new_etype:
            self.transform_asymmetric_etype(c_etype, g, data_dict)
        else:
            src, dst = g.edges(etype=c_etype)
            src, dst = F.cat([src, dst], dim=0), F.cat([dst, src], dim=0)
            data_dict[c_etype] = (src, dst)

    def transform_asymmetric_etype(self, c_etype, g, data_dict):
        r"""Transform the graph corresponding to an asymmetric canonical edge type.

        Parameters
        ----------
        c_etype : tuple of str
            A canonical edge type.
        g : DGLGraph
            The graph.
        data_dict : dict
            The edge data to update.
        """
        utype, etype, vtype = c_etype
        src, dst = g.edges(etype=c_etype)
        data_dict.update(
            {
                c_etype: (src, dst),
                (vtype, "rev_{}".format(etype), utype): (dst, src),
            }
        )

    def transform_etype(self, c_etype, g, data_dict):
        r"""Transform the graph corresponding to a canonical edge type.

        Parameters
        ----------
        c_etype : tuple of str
            A canonical edge type.
        g : DGLGraph
            The graph.
        data_dict : dict
            The edge data to update.
        """
        utype, _, vtype = c_etype
        if utype == vtype:
            self.transform_symmetric_etype(c_etype, g, data_dict)
        else:
            self.transform_asymmetric_etype(c_etype, g, data_dict)

    def __call__(self, g):
        data_dict = dict()
        for c_etype in g.canonical_etypes:
            self.transform_etype(c_etype, g, data_dict)
        new_g = update_graph_structure(g, data_dict, copy_edata=False)

        # Copy and expand edata
        for c_etype in g.canonical_etypes:
            utype, etype, vtype = c_etype
            if utype != vtype or self.sym_new_etype:
                rev_c_etype = (vtype, "rev_{}".format(etype), utype)
                for key, feat in g.edges[c_etype].data.items():
                    new_g.edges[c_etype].data[key] = feat
                    if self.copy_edata:
                        new_g.edges[rev_c_etype].data[key] = feat
            else:
                for key, feat in g.edges[c_etype].data.items():
                    new_feat = (
                        feat
                        if self.copy_edata
                        else F.zeros(
                            F.shape(feat), F.dtype(feat), F.context(feat)
                        )
                    )
                    new_g.edges[c_etype].data[key] = F.cat(
                        [feat, new_feat], dim=0
                    )

        return new_g


class ToSimple(BaseTransform):
    r"""Convert a graph to a simple graph without parallel edges and return a new graph.

    Parameters
    ----------
    return_counts : str, optional
        The edge feature name to hold the edge count in the original graph.
    aggregator : str, optional
        The way to coalesce features of duplicate edges.

        * ``'arbitrary'``: select arbitrarily from one of the duplicate edges
        * ``'sum'``: take the sum over the duplicate edges
        * ``'mean'``: take the mean over the duplicate edges

    Example
    -------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> from dgl import ToSimple

    Case1: Convert a homogeneous graph to a simple graph

    >>> transform = ToSimple()
    >>> g = dgl.graph(([0, 1, 1], [1, 2, 2]))
    >>> g.edata['w'] = torch.tensor([[0.1], [0.2], [0.3]])
    >>> sg = transform(g)
    >>> print(sg.edges())
    (tensor([0, 1]), tensor([1, 2]))
    >>> print(sg.edata['count'])
    tensor([1, 2])
    >>> print(sg.edata['w'])
    tensor([[0.1000], [0.2000]])

    Case2: Convert a heterogeneous graph to a simple graph

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2]),
    ...     ('user', 'plays', 'game'): ([0, 1, 0], [1, 1, 1])
    ... })
    >>> sg = transform(g)
    >>> print(sg.edges(etype='follows'))
    (tensor([0, 1]), tensor([1, 2]))
    >>> print(sg.edges(etype='plays'))
    (tensor([0, 1]), tensor([1, 1]))
    """

    def __init__(self, return_counts="count", aggregator="arbitrary"):
        self.return_counts = return_counts
        self.aggregator = aggregator

    def __call__(self, g):
        return functional.to_simple(
            g,
            return_counts=self.return_counts,
            copy_edata=True,
            aggregator=self.aggregator,
        )


class LineGraph(BaseTransform):
    r"""Return the line graph of the input graph.

    The line graph :math:`L(G)` of a given graph :math:`G` is a graph where
    the nodes in :math:`L(G)` correspond to the edges in :math:`G`. For a pair
    of edges :math:`(u, v)` and :math:`(v, w)` in :math:`G`, there will be an
    edge from the node corresponding to :math:`(u, v)` to the node corresponding to
    :math:`(v, w)` in :math:`L(G)`.

    This module only works for homogeneous graphs.

    Parameters
    ----------
    backtracking : bool, optional
        If False, there will be an edge from the line graph node corresponding to
        :math:`(u, v)` to the line graph node corresponding to :math:`(v, u)`.

    Example
    -------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> from dgl import LineGraph

    Case1: Backtracking is True

    >>> transform = LineGraph()
    >>> g = dgl.graph(([0, 1, 1], [1, 0, 2]))
    >>> g.ndata['h'] = torch.tensor([[0.], [1.], [2.]])
    >>> g.edata['w'] = torch.tensor([[0.], [0.1], [0.2]])
    >>> new_g = transform(g)
    >>> print(new_g)
    Graph(num_nodes=3, num_edges=3,
          ndata_schemes={'w': Scheme(shape=(1,), dtype=torch.float32)}
          edata_schemes={})
    >>> print(new_g.edges())
    (tensor([0, 0, 1]), tensor([1, 2, 0]))

    Case2: Backtracking is False

    >>> transform = LineGraph(backtracking=False)
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([0]), tensor([2]))
    """

    def __init__(self, backtracking=True):
        self.backtracking = backtracking

    def __call__(self, g):
        return functional.line_graph(
            g, backtracking=self.backtracking, shared=True
        )


class KHopGraph(BaseTransform):
    r"""Return the graph whose edges connect the :math:`k`-hop neighbors of the original graph.

    This module only works for homogeneous graphs.

    Parameters
    ----------
    k : int
        The number of hops.

    Example
    -------

    >>> import dgl
    >>> from dgl import KHopGraph

    >>> transform = KHopGraph(2)
    >>> g = dgl.graph(([0, 1], [1, 2]))
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([0]), tensor([2]))
    """

    def __init__(self, k):
        self.k = k

    def __call__(self, g):
        return functional.khop_graph(g, self.k)


class AddMetaPaths(BaseTransform):
    r"""Add new edges to an input graph based on given metapaths, as described in
    `Heterogeneous Graph Attention Network <https://arxiv.org/abs/1903.07293>`__.

    Formally, a metapath is a path of the form

    .. math::

        \mathcal{V}_1 \xrightarrow{R_1} \mathcal{V}_2 \xrightarrow{R_2} \ldots
        \xrightarrow{R_{\ell-1}} \mathcal{V}_{\ell}

    in which :math:`\mathcal{V}_i` represents a node type and :math:`\xrightarrow{R_j}`
    represents a relation type connecting its two adjacent node types. The adjacency matrix
    corresponding to the metapath is obtained by sequential multiplication of adjacency matrices
    along the metapath.

    Parameters
    ----------
    metapaths : dict[str, list]
        The metapaths to add, mapping a metapath name to a metapath. For example,
        :attr:`{'co-author': [('person', 'author', 'paper'), ('paper', 'authored by', 'person')]}`
    keep_orig_edges : bool, optional
        If True, it will keep the edges of the original graph. Otherwise, it will drop them.

    Example
    -------

    >>> import dgl
    >>> from dgl import AddMetaPaths

    >>> transform = AddMetaPaths({
    ...     'accepted': [('person', 'author', 'paper'), ('paper', 'accepted', 'venue')],
    ...     'rejected': [('person', 'author', 'paper'), ('paper', 'rejected', 'venue')]
    ... })
    >>> g = dgl.heterograph({
    ...     ('person', 'author', 'paper'): ([0, 0, 1], [1, 2, 2]),
    ...     ('paper', 'accepted', 'venue'): ([1], [0]),
    ...     ('paper', 'rejected', 'venue'): ([2], [1])
    ... })
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype=('person', 'accepted', 'venue')))
    (tensor([0]), tensor([0]))
    >>> print(new_g.edges(etype=('person', 'rejected', 'venue')))
    (tensor([0, 1]), tensor([1, 1]))
    """

    def __init__(self, metapaths, keep_orig_edges=True):
        self.metapaths = metapaths
        self.keep_orig_edges = keep_orig_edges

    def __call__(self, g):
        data_dict = dict()

        for meta_etype, metapath in self.metapaths.items():
            meta_g = functional.metapath_reachable_graph(g, metapath)
            u_type = metapath[0][0]
            v_type = metapath[-1][-1]
            data_dict[(u_type, meta_etype, v_type)] = meta_g.edges()

        if self.keep_orig_edges:
            for c_etype in g.canonical_etypes:
                data_dict[c_etype] = g.edges(etype=c_etype)
            new_g = update_graph_structure(g, data_dict, copy_edata=True)
        else:
            new_g = update_graph_structure(g, data_dict, copy_edata=False)

        return new_g


class Compose(BaseTransform):
    r"""Create a transform composed of multiple transforms in sequence.

    Parameters
    ----------
    transforms : list of Callable
        A list of transform objects to apply in order. A transform object should inherit
        :class:`~dgl.BaseTransform` and implement :func:`~dgl.BaseTransform.__call__`.

    Example
    -------

    >>> import dgl
    >>> from dgl import transforms as T

    >>> g = dgl.graph(([0, 0], [1, 1]))
    >>> transform = T.Compose([T.ToSimple(), T.AddReverse()])
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([0, 1]), tensor([1, 0]))
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, g):
        for transform in self.transforms:
            g = transform(g)
        return g

    def __repr__(self):
        args = ["  " + str(transform) for transform in self.transforms]
        return self.__class__.__name__ + "([\n" + ",\n".join(args) + "\n])"


class GCNNorm(BaseTransform):
    r"""Apply symmetric adjacency normalization to an input graph and save the result edge
    weights, as described in `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__.

    For a heterogeneous graph, this only applies to symmetric canonical edge types, whose source
    and destination node types are identical.

    Parameters
    ----------
    eweight_name : str, optional
        :attr:`edata` name to retrieve and store edge weights. The edge weights are optional.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import GCNNorm
    >>> transform = GCNNorm()
    >>> g = dgl.graph(([0, 1, 2], [0, 0, 1]))

    Case1: Transform an unweighted graph

    >>> g = transform(g)
    >>> print(g.edata['w'])
    tensor([0.5000, 0.7071, 0.0000])

    Case2: Transform a weighted graph

    >>> g.edata['w'] = torch.tensor([0.1, 0.2, 0.3])
    >>> g = transform(g)
    >>> print(g.edata['w'])
    tensor([0.3333, 0.6667, 0.0000])
    """

    def __init__(self, eweight_name="w"):
        self.eweight_name = eweight_name

    def calc_etype(self, c_etype, g):
        r"""

        Description
        -----------
        Get edge weights for an edge type.
        """
        ntype = c_etype[0]
        with g.local_scope():
            if self.eweight_name in g.edges[c_etype].data:
                g.update_all(
                    fn.copy_e(self.eweight_name, "m"),
                    fn.sum("m", "deg"),
                    etype=c_etype,
                )
                deg_inv_sqrt = 1.0 / F.sqrt(g.nodes[ntype].data["deg"])
                g.nodes[ntype].data["w"] = F.replace_inf_with_zero(deg_inv_sqrt)
                g.apply_edges(
                    lambda edge: {
                        "w": edge.src["w"]
                        * edge.data[self.eweight_name]
                        * edge.dst["w"]
                    },
                    etype=c_etype,
                )
            else:
                deg = g.in_degrees(etype=c_etype)
                deg_inv_sqrt = 1.0 / F.sqrt(F.astype(deg, F.float32))
                g.nodes[ntype].data["w"] = F.replace_inf_with_zero(deg_inv_sqrt)
                g.apply_edges(
                    lambda edges: {"w": edges.src["w"] * edges.dst["w"]},
                    etype=c_etype,
                )
            return g.edges[c_etype].data["w"]

    def __call__(self, g):
        result = dict()
        for c_etype in g.canonical_etypes:
            utype, _, vtype = c_etype
            if utype == vtype:
                result[c_etype] = self.calc_etype(c_etype, g)

        for c_etype, eweight in result.items():
            g.edges[c_etype].data[self.eweight_name] = eweight
        return g


class PPR(BaseTransform):
    r"""Apply personalized PageRank (PPR) to an input graph for diffusion, as introduced in
    `The pagerank citation ranking: Bringing order to the web
    <http://ilpubs.stanford.edu:8090/422/>`__.

    A sparsification will be applied to the weighted adjacency matrix after diffusion.
    Specifically, edges whose weight is below a threshold will be dropped.

    This module only works for homogeneous graphs.

    Parameters
    ----------
    alpha : float, optional
        Restart probability, which commonly lies in :math:`[0.05, 0.2]`.
    eweight_name : str, optional
        :attr:`edata` name to retrieve and store edge weights. If it does
        not exist in an input graph, this module initializes a weight of 1
        for all edges. The edge weights should be a tensor of shape :math:`(E)`,
        where E is the number of edges.
    eps : float, optional
        The threshold to preserve edges in sparsification after diffusion. Edges of a
        weight smaller than eps will be dropped.
    avg_degree : int, optional
        The desired average node degree of the result graph. This is the other way to
        control the sparsity of the result graph and will only be effective if
        :attr:`eps` is not given.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import PPR

    >>> transform = PPR(avg_degree=2)
    >>> g = dgl.graph(([0, 1, 2, 3, 4], [2, 3, 4, 5, 3]))
    >>> g.edata['w'] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> new_g = transform(g)
    >>> print(new_g.edata['w'])
    tensor([0.1500, 0.1500, 0.1500, 0.0255, 0.0163, 0.1500, 0.0638, 0.0383, 0.1500,
            0.0510, 0.0217, 0.1500])
    """

    def __init__(self, alpha=0.15, eweight_name="w", eps=None, avg_degree=5):
        self.alpha = alpha
        self.eweight_name = eweight_name
        self.eps = eps
        self.avg_degree = avg_degree

    def get_eps(self, num_nodes, mat):
        r"""Get the threshold for graph sparsification."""
        if self.eps is None:
            # Infer from self.avg_degree
            if self.avg_degree > num_nodes:
                return float("-inf")
            sorted_weights = torch.sort(mat.flatten(), descending=True).values
            return sorted_weights[self.avg_degree * num_nodes - 1]
        else:
            return self.eps

    def __call__(self, g):
        # Step1: PPR diffusion
        # (α - 1) A
        device = g.device
        eweight = (self.alpha - 1) * g.edata.get(
            self.eweight_name, F.ones((g.num_edges(),), F.float32, device)
        )
        num_nodes = g.num_nodes()
        mat = F.zeros((num_nodes, num_nodes), F.float32, device)
        src, dst = g.edges()
        src, dst = F.astype(src, F.int64), F.astype(dst, F.int64)
        mat[dst, src] = eweight
        # I_n + (α - 1) A
        nids = F.astype(g.nodes(), F.int64)
        mat[nids, nids] = mat[nids, nids] + 1
        # α (I_n + (α - 1) A)^-1
        diff_mat = self.alpha * F.inverse(mat)

        # Step2: sparsification
        num_nodes = g.num_nodes()
        eps = self.get_eps(num_nodes, diff_mat)
        dst, src = (diff_mat >= eps).nonzero(as_tuple=False).t()
        data_dict = {g.canonical_etypes[0]: (src, dst)}
        new_g = update_graph_structure(g, data_dict, copy_edata=False)
        new_g.edata[self.eweight_name] = diff_mat[dst, src]

        return new_g


def is_bidirected(g):
    """Return whether the graph is a bidirected graph.

    A graph is bidirected if for any edge :math:`(u, v)` in :math:`G` with weight :math:`w`,
    there exists an edge :math:`(v, u)` in :math:`G` with the same weight.
    """
    src, dst = g.edges()
    num_nodes = g.num_nodes()

    # Sort first by src then dst
    idx_src_dst = src * num_nodes + dst
    perm_src_dst = F.argsort(idx_src_dst, dim=0, descending=False)
    src1, dst1 = src[perm_src_dst], dst[perm_src_dst]

    # Sort first by dst then src
    idx_dst_src = dst * num_nodes + src
    perm_dst_src = F.argsort(idx_dst_src, dim=0, descending=False)
    src2, dst2 = src[perm_dst_src], dst[perm_dst_src]

    return F.allclose(src1, dst2) and F.allclose(src2, dst1)


# pylint: disable=C0103
class HeatKernel(BaseTransform):
    r"""Apply heat kernel to an input graph for diffusion, as introduced in
    `Diffusion kernels on graphs and other discrete structures
    <https://www.ml.cmu.edu/research/dap-papers/kondor-diffusion-kernels.pdf>`__.

    A sparsification will be applied to the weighted adjacency matrix after diffusion.
    Specifically, edges whose weight is below a threshold will be dropped.

    This module only works for homogeneous graphs.

    Parameters
    ----------
    t : float, optional
        Diffusion time, which commonly lies in :math:`[2, 10]`.
    eweight_name : str, optional
        :attr:`edata` name to retrieve and store edge weights. If it does
        not exist in an input graph, this module initializes a weight of 1
        for all edges. The edge weights should be a tensor of shape :math:`(E)`,
        where E is the number of edges.
    eps : float, optional
        The threshold to preserve edges in sparsification after diffusion. Edges of a
        weight smaller than eps will be dropped.
    avg_degree : int, optional
        The desired average node degree of the result graph. This is the other way to
        control the sparsity of the result graph and will only be effective if
        :attr:`eps` is not given.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import HeatKernel

    >>> transform = HeatKernel(avg_degree=2)
    >>> g = dgl.graph(([0, 1, 2, 3, 4], [2, 3, 4, 5, 3]))
    >>> g.edata['w'] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> new_g = transform(g)
    >>> print(new_g.edata['w'])
    tensor([0.1353, 0.1353, 0.1353, 0.0541, 0.0406, 0.1353, 0.1353, 0.0812, 0.1353,
            0.1083, 0.0541, 0.1353])
    """

    def __init__(self, t=2.0, eweight_name="w", eps=None, avg_degree=5):
        self.t = t
        self.eweight_name = eweight_name
        self.eps = eps
        self.avg_degree = avg_degree

    def get_eps(self, num_nodes, mat):
        r"""Get the threshold for graph sparsification."""
        if self.eps is None:
            # Infer from self.avg_degree
            if self.avg_degree > num_nodes:
                return float("-inf")
            sorted_weights = torch.sort(mat.flatten(), descending=True).values
            return sorted_weights[self.avg_degree * num_nodes - 1]
        else:
            return self.eps

    def __call__(self, g):
        # Step1: heat kernel diffusion
        # t A
        device = g.device
        eweight = self.t * g.edata.get(
            self.eweight_name, F.ones((g.num_edges(),), F.float32, device)
        )
        num_nodes = g.num_nodes()
        mat = F.zeros((num_nodes, num_nodes), F.float32, device)
        src, dst = g.edges()
        src, dst = F.astype(src, F.int64), F.astype(dst, F.int64)
        mat[dst, src] = eweight
        # t (A - I_n)
        nids = F.astype(g.nodes(), F.int64)
        mat[nids, nids] = mat[nids, nids] - self.t

        if is_bidirected(g):
            e, V = torch.linalg.eigh(mat, UPLO="U")
            diff_mat = V @ torch.diag(e.exp()) @ V.t()
        else:
            diff_mat_np = expm(mat.cpu().numpy())
            diff_mat = torch.Tensor(diff_mat_np).to(device)

        # Step2: sparsification
        num_nodes = g.num_nodes()
        eps = self.get_eps(num_nodes, diff_mat)
        dst, src = (diff_mat >= eps).nonzero(as_tuple=False).t()
        data_dict = {g.canonical_etypes[0]: (src, dst)}
        new_g = update_graph_structure(g, data_dict, copy_edata=False)
        new_g.edata[self.eweight_name] = diff_mat[dst, src]

        return new_g


class GDC(BaseTransform):
    r"""Apply graph diffusion convolution (GDC) to an input graph, as introduced in
    `Diffusion Improves Graph Learning <https://www.in.tum.de/daml/gdc/>`__.

    A sparsification will be applied to the weighted adjacency matrix after diffusion.
    Specifically, edges whose weight is below a threshold will be dropped.

    This module only works for homogeneous graphs.

    Parameters
    ----------
    coefs : list[float], optional
        List of coefficients. :math:`\theta_k` for each power of the adjacency matrix.
    eweight_name : str, optional
        :attr:`edata` name to retrieve and store edge weights. If it does
        not exist in an input graph, this module initializes a weight of 1
        for all edges. The edge weights should be a tensor of shape :math:`(E)`,
        where E is the number of edges.
    eps : float, optional
        The threshold to preserve edges in sparsification after diffusion. Edges of a
        weight smaller than eps will be dropped.
    avg_degree : int, optional
        The desired average node degree of the result graph. This is the other way to
        control the sparsity of the result graph and will only be effective if
        :attr:`eps` is not given.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import GDC

    >>> transform = GDC([0.3, 0.2, 0.1], avg_degree=2)
    >>> g = dgl.graph(([0, 1, 2, 3, 4], [2, 3, 4, 5, 3]))
    >>> g.edata['w'] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> new_g = transform(g)
    >>> print(new_g.edata['w'])
    tensor([0.3000, 0.3000, 0.0200, 0.3000, 0.0400, 0.3000, 0.1000, 0.0600, 0.3000,
            0.0800, 0.0200, 0.3000])
    """

    def __init__(self, coefs, eweight_name="w", eps=None, avg_degree=5):
        self.coefs = coefs
        self.eweight_name = eweight_name
        self.eps = eps
        self.avg_degree = avg_degree

    def get_eps(self, num_nodes, mat):
        r"""Get the threshold for graph sparsification."""
        if self.eps is None:
            # Infer from self.avg_degree
            if self.avg_degree > num_nodes:
                return float("-inf")
            sorted_weights = torch.sort(mat.flatten(), descending=True).values
            return sorted_weights[self.avg_degree * num_nodes - 1]
        else:
            return self.eps

    def __call__(self, g):
        # Step1: diffusion
        # A
        device = g.device
        eweight = g.edata.get(
            self.eweight_name, F.ones((g.num_edges(),), F.float32, device)
        )
        num_nodes = g.num_nodes()
        adj = F.zeros((num_nodes, num_nodes), F.float32, device)
        src, dst = g.edges()
        src, dst = F.astype(src, F.int64), F.astype(dst, F.int64)
        adj[dst, src] = eweight

        # theta_0 I_n
        mat = torch.eye(num_nodes, device=device)
        diff_mat = self.coefs[0] * mat
        # add theta_k A^k
        for coef in self.coefs[1:]:
            mat = mat @ adj
            diff_mat += coef * mat

        # Step2: sparsification
        num_nodes = g.num_nodes()
        eps = self.get_eps(num_nodes, diff_mat)
        dst, src = (diff_mat >= eps).nonzero(as_tuple=False).t()
        data_dict = {g.canonical_etypes[0]: (src, dst)}
        new_g = update_graph_structure(g, data_dict, copy_edata=False)
        new_g.edata[self.eweight_name] = diff_mat[dst, src]

        return new_g


class NodeShuffle(BaseTransform):
    r"""Randomly shuffle the nodes.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import NodeShuffle

    >>> transform = NodeShuffle()
    >>> g = dgl.graph(([0, 1], [1, 2]))
    >>> g.ndata['h1'] = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    >>> g.ndata['h2'] = torch.tensor([[7., 8.], [9., 10.], [11., 12.]])
    >>> g = transform(g)
    >>> print(g.ndata['h1'])
    tensor([[5., 6.],
            [3., 4.],
            [1., 2.]])
    >>> print(g.ndata['h2'])
    tensor([[11., 12.],
            [ 9., 10.],
            [ 7.,  8.]])
    """

    def __call__(self, g):
        g = g.clone()
        for ntype in g.ntypes:
            nids = F.astype(g.nodes(ntype), F.int64)
            perm = F.rand_shuffle(nids)
            for key, feat in g.nodes[ntype].data.items():
                g.nodes[ntype].data[key] = feat[perm]
        return g


# pylint: disable=C0103
class DropNode(BaseTransform):
    r"""Randomly drop nodes, as described in
    `Graph Contrastive Learning with Augmentations <https://arxiv.org/abs/2010.13902>`__.

    Parameters
    ----------
    p : float, optional
        Probability of a node to be dropped.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import DropNode

    >>> transform = DropNode()
    >>> g = dgl.rand_graph(5, 20)
    >>> g.ndata['h'] = torch.arange(g.num_nodes())
    >>> g.edata['h'] = torch.arange(g.num_edges())
    >>> new_g = transform(g)
    >>> print(new_g)
    Graph(num_nodes=3, num_edges=7,
          ndata_schemes={'h': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'h': Scheme(shape=(), dtype=torch.int64)})
    >>> print(new_g.ndata['h'])
    tensor([0, 1, 2])
    >>> print(new_g.edata['h'])
    tensor([0, 6, 14, 5, 17, 3, 11])
    """

    def __init__(self, p=0.5):
        self.p = p
        self.dist = Bernoulli(p)

    def __call__(self, g):
        g = g.clone()

        # Fast path
        if self.p == 0:
            return g

        for ntype in g.ntypes:
            samples = self.dist.sample(torch.Size([g.num_nodes(ntype)]))
            nids_to_remove = g.nodes(ntype)[samples.bool().to(g.device)]
            g.remove_nodes(nids_to_remove, ntype=ntype)
        return g


# pylint: disable=C0103
class DropEdge(BaseTransform):
    r"""Randomly drop edges, as described in
    `DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
    <https://arxiv.org/abs/1907.10903>`__ and `Graph Contrastive Learning with Augmentations
    <https://arxiv.org/abs/2010.13902>`__.

    Parameters
    ----------
    p : float, optional
        Probability of an edge to be dropped.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import DropEdge

    >>> transform = DropEdge()
    >>> g = dgl.rand_graph(5, 20)
    >>> g.edata['h'] = torch.arange(g.num_edges())
    >>> new_g = transform(g)
    >>> print(new_g)
    Graph(num_nodes=5, num_edges=12,
          ndata_schemes={}
          edata_schemes={'h': Scheme(shape=(), dtype=torch.int64)})
    >>> print(new_g.edata['h'])
    tensor([0, 1, 3, 7, 8, 10, 11, 12, 13, 15, 18, 19])
    """

    def __init__(self, p=0.5):
        self.p = p
        self.dist = Bernoulli(p)

    def __call__(self, g):
        g = g.clone()

        # Fast path
        if self.p == 0:
            return g

        for c_etype in g.canonical_etypes:
            samples = self.dist.sample(torch.Size([g.num_edges(c_etype)]))
            eids_to_remove = g.edges(form="eid", etype=c_etype)[
                samples.bool().to(g.device)
            ]
            g.remove_edges(eids_to_remove, etype=c_etype)
        return g


class AddEdge(BaseTransform):
    r"""Randomly add edges, as described in `Graph Contrastive Learning with Augmentations
    <https://arxiv.org/abs/2010.13902>`__.

    Parameters
    ----------
    ratio : float, optional
        Number of edges to add divided by the number of existing edges.

    Example
    -------

    >>> import dgl
    >>> from dgl import AddEdge

    >>> transform = AddEdge()
    >>> g = dgl.rand_graph(5, 20)
    >>> new_g = transform(g)
    >>> print(new_g.num_edges())
    24
    """

    def __init__(self, ratio=0.2):
        self.ratio = ratio

    def __call__(self, g):
        # Fast path
        if self.ratio == 0.0:
            return g

        device = g.device
        idtype = g.idtype
        g = g.clone()
        for c_etype in g.canonical_etypes:
            utype, _, vtype = c_etype
            num_edges_to_add = int(g.num_edges(c_etype) * self.ratio)
            src = F.randint(
                [num_edges_to_add],
                idtype,
                device,
                low=0,
                high=g.num_nodes(utype),
            )
            dst = F.randint(
                [num_edges_to_add],
                idtype,
                device,
                low=0,
                high=g.num_nodes(vtype),
            )
            g.add_edges(src, dst, etype=c_etype)
        return g


class SIGNDiffusion(BaseTransform):
    r"""The diffusion operator from `SIGN: Scalable Inception Graph Neural Networks
    <https://arxiv.org/abs/2004.11198>`__

    It performs node feature diffusion with :math:`TX, \cdots, T^{k}X`, where :math:`T`
    is a diffusion matrix and :math:`X` is the input node features.

    Specifically, this module provides four options for :math:`T`.

    **raw**: raw adjacency matrix :math:`A`

    **rw**: random walk (row-normalized) adjacency matrix :math:`D^{-1}A`, where
    :math:`D` is the degree matrix.

    **gcn**: symmetrically normalized adjacency matrix used by
    `GCN <https://arxiv.org/abs/1609.02907>`__, :math:`D^{-1/2}AD^{-1/2}`

    **ppr**: approximate personalized PageRank used by
    `APPNP <https://arxiv.org/abs/1810.05997>`__

    .. math::
        H^{0} &= X

        H^{l+1} &= (1-\alpha)\left(D^{-1/2}AD^{-1/2} H^{l}\right) + \alpha X

    This module only works for homogeneous graphs.

    Parameters
    ----------
    k : int
        The maximum number of times for node feature diffusion.
    in_feat_name : str, optional
        :attr:`g.ndata[{in_feat_name}]` should store the input node features. Default: 'feat'
    out_feat_name : str, optional
        :attr:`g.ndata[{out_feat_name}_i]` will store the result of diffusing
        input node features for i times. Default: 'out_feat'
    eweight_name : str, optional
        Name to retrieve edge weights from :attr:`g.edata`. Default: None,
        treating the graph as unweighted.
    diffuse_op : str, optional
        The diffusion operator to use, which can be 'raw', 'rw', 'gcn', or 'ppr'.
        Default: 'raw'
    alpha : float, optional
        Restart probability if :attr:`diffuse_op` is :attr:`'ppr'`,
        which commonly lies in :math:`[0.05, 0.2]`. Default: 0.2

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import SIGNDiffusion

    >>> transform = SIGNDiffusion(k=2, eweight_name='w')
    >>> num_nodes = 5
    >>> num_edges = 20
    >>> g = dgl.rand_graph(num_nodes, num_edges)
    >>> g.ndata['feat'] = torch.randn(num_nodes, 10)
    >>> g.edata['w'] = torch.randn(num_edges)
    >>> transform(g)
    Graph(num_nodes=5, num_edges=20,
          ndata_schemes={'feat': Scheme(shape=(10,), dtype=torch.float32),
                         'out_feat_1': Scheme(shape=(10,), dtype=torch.float32),
                         'out_feat_2': Scheme(shape=(10,), dtype=torch.float32)}
          edata_schemes={'w': Scheme(shape=(), dtype=torch.float32)})
    """

    def __init__(
        self,
        k,
        in_feat_name="feat",
        out_feat_name="out_feat",
        eweight_name=None,
        diffuse_op="raw",
        alpha=0.2,
    ):
        self.k = k
        self.in_feat_name = in_feat_name
        self.out_feat_name = out_feat_name
        self.eweight_name = eweight_name
        self.diffuse_op = diffuse_op
        self.alpha = alpha

        if diffuse_op == "raw":
            self.diffuse = self.raw
        elif diffuse_op == "rw":
            self.diffuse = self.rw
        elif diffuse_op == "gcn":
            self.diffuse = self.gcn
        elif diffuse_op == "ppr":
            self.diffuse = self.ppr
        else:
            raise DGLError(
                "Expect diffuse_op to be from ['raw', 'rw', 'gcn', 'ppr'], \
                got {}".format(
                    diffuse_op
                )
            )

    def __call__(self, g):
        feat_list = self.diffuse(g)

        for i in range(1, self.k + 1):
            g.ndata[self.out_feat_name + "_" + str(i)] = feat_list[i - 1]
        return g

    def raw(self, g):
        use_eweight = False
        if (self.eweight_name is not None) and self.eweight_name in g.edata:
            use_eweight = True

        feat_list = []
        with g.local_scope():
            if use_eweight:
                message_func = fn.u_mul_e(
                    self.in_feat_name, self.eweight_name, "m"
                )
            else:
                message_func = fn.copy_u(self.in_feat_name, "m")
            for _ in range(self.k):
                g.update_all(message_func, fn.sum("m", self.in_feat_name))
                feat_list.append(g.ndata[self.in_feat_name])
        return feat_list

    def rw(self, g):
        use_eweight = False
        if (self.eweight_name is not None) and self.eweight_name in g.edata:
            use_eweight = True

        feat_list = []
        with g.local_scope():
            g.ndata["h"] = g.ndata[self.in_feat_name]
            if use_eweight:
                message_func = fn.u_mul_e("h", self.eweight_name, "m")
                reduce_func = fn.sum("m", "h")
                # Compute the diagonal entries of D from the weighted A
                g.update_all(
                    fn.copy_e(self.eweight_name, "m"), fn.sum("m", "z")
                )
            else:
                message_func = fn.copy_u("h", "m")
                reduce_func = fn.mean("m", "h")

            for _ in range(self.k):
                g.update_all(message_func, reduce_func)
                if use_eweight:
                    g.ndata["h"] = g.ndata["h"] / F.reshape(
                        g.ndata["z"], (g.num_nodes(), 1)
                    )
                feat_list.append(g.ndata["h"])
        return feat_list

    def gcn(self, g):
        feat_list = []
        with g.local_scope():
            if self.eweight_name is None:
                eweight_name = "w"
                if eweight_name in g.edata:
                    g.edata.pop(eweight_name)
            else:
                eweight_name = self.eweight_name

            transform = GCNNorm(eweight_name=eweight_name)
            transform(g)

            for _ in range(self.k):
                g.update_all(
                    fn.u_mul_e(self.in_feat_name, eweight_name, "m"),
                    fn.sum("m", self.in_feat_name),
                )
                feat_list.append(g.ndata[self.in_feat_name])
        return feat_list

    def ppr(self, g):
        feat_list = []
        with g.local_scope():
            if self.eweight_name is None:
                eweight_name = "w"
                if eweight_name in g.edata:
                    g.edata.pop(eweight_name)
            else:
                eweight_name = self.eweight_name
            transform = GCNNorm(eweight_name=eweight_name)
            transform(g)

            in_feat = g.ndata[self.in_feat_name]
            for _ in range(self.k):
                g.update_all(
                    fn.u_mul_e(self.in_feat_name, eweight_name, "m"),
                    fn.sum("m", self.in_feat_name),
                )
                g.ndata[self.in_feat_name] = (1 - self.alpha) * g.ndata[
                    self.in_feat_name
                ] + self.alpha * in_feat
                feat_list.append(g.ndata[self.in_feat_name])
        return feat_list


class ToLevi(BaseTransform):
    r"""This function transforms the original graph to its heterogeneous Levi graph,
    by converting edges to intermediate nodes, only support homogeneous directed graph.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl import ToLevi

    >>> transform = ToLevi()
    >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
    >>> g.ndata['h'] = th.randn((g.num_nodes(), 2))
    >>> g.edata['w'] = th.randn((g.num_edges(), 2))
    >>> lg = transform(g)
    >>> lg
    Grpah(num_nodes={'edge': 4, 'node': 4},
          num_edges={('edge', 'e2n', 'node'): 4,
                     ('node', 'n2e', 'edge'): 4},
          metagraph=[('edge', 'node', 'e2n'),
                     ('node', 'edge', 'n2e')])
    >>> lg.nodes('node')
    tensor([0, 1, 2, 3])
    >>> lg.nodes('edge')
    tensor([0, 1, 2, 3])
    >>> lg.nodes['node'].data['h'].shape
    torch.Size([4, 2])
    >>> lg.nodes['edge'].data['w'].shape
    torch.Size([4, 2])
    """

    def __init__(self):
        pass

    def __call__(self, g):
        r"""
        Parameters
        ----------
        g : DGLGraph
            The input graph, should be a homogeneous directed graph.

        Returns
        -------
        DGLGraph
            The Levi graph of input, will be a heterogeneous graph, where nodes of
            ntypes ``'node'`` and ``'edge'`` have corresponding IDs of nodes and edges
            in the original graph. Edge features of the input graph are copied to
            corresponding new nodes of ntype ``'edge'``.
        """
        device = g.device
        idtype = g.idtype

        edge_list = g.edges()
        n2e = edge_list[0], F.arange(0, g.num_edges(), idtype, device)
        e2n = F.arange(0, g.num_edges(), idtype, device), edge_list[1]
        graph_data = {
            ("node", "n2e", "edge"): n2e,
            ("edge", "e2n", "node"): e2n,
        }
        levi_g = convert.heterograph(graph_data, idtype=idtype, device=device)

        # Copy ndata and edata
        # Since the node types in dgl.heterograph are in alphabetical order
        # ('edge' < 'node'), edge_frames should be in front of node_frames.
        node_frames = utils.extract_node_subframes(g, nodes_or_device=device)
        edge_frames = utils.extract_edge_subframes(g, edges_or_device=device)
        utils.set_new_frames(levi_g, node_frames=edge_frames + node_frames)

        return levi_g


class SVDPE(BaseTransform):
    r"""SVD-based Positional Encoding, as introduced in
    `Global Self-Attention as a Replacement for Graph Convolution
    <https://arxiv.org/pdf/2108.03348.pdf>`__

    This function computes the largest :math:`k` singular values and
    corresponding left and right singular vectors to form positional encodings,
    which could be stored in ndata.

    Parameters
    ----------
    k : int
        Number of largest singular values and corresponding singular vectors
        used for positional encoding.
    feat_name : str, optional
        Name to store the computed positional encodings in ndata.
        Default : ``svd_pe``
    padding : bool, optional
        If False, raise an error when :math:`k > N`,
        where :math:`N` is the number of nodes in :attr:`g`.
        If True, add zero paddings in the end of encodings when :math:`k > N`.
        Default : False.
    random_flip : bool, optional
        If True, randomly flip the signs of encoding vectors.
        Proposed to be activated during training for better generalization.
        Default : True.

    Example
    -------
    >>> import dgl
    >>> from dgl import SVDPE

    >>> transform = SVDPE(k=2, feat_name="svd_pe")
    >>> g = dgl.graph(([0,1,2,3,4,2,3,1,4,0], [2,3,1,4,0,0,1,2,3,4]))
    >>> g_ = transform(g)
    >>> print(g_.ndata['svd_pe'])
    tensor([[-6.3246e-01, -1.1373e-07, -6.3246e-01,  0.0000e+00],
            [-6.3246e-01,  7.6512e-01, -6.3246e-01, -7.6512e-01],
            [ 6.3246e-01,  4.7287e-01,  6.3246e-01, -4.7287e-01],
            [-6.3246e-01, -7.6512e-01, -6.3246e-01,  7.6512e-01],
            [ 6.3246e-01, -4.7287e-01,  6.3246e-01,  4.7287e-01]])
    """

    def __init__(self, k, feat_name="svd_pe", padding=False, random_flip=True):
        self.k = k
        self.feat_name = feat_name
        self.padding = padding
        self.random_flip = random_flip

    def __call__(self, g):
        encoding = functional.svd_pe(
            g, k=self.k, padding=self.padding, random_flip=self.random_flip
        )
        g.ndata[self.feat_name] = F.copy_to(encoding, g.device)

        return g
