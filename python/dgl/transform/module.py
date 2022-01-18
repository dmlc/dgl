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

from .. import convert
from .. import backend as F
from . import functional

__all__ = [
    'BaseTransform',
    'AddSelfLoop',
    'RemoveSelfLoop',
    'AddReverse',
    'ToSimple',
    'LineGraph',
    'KHopGraph',
    'AddMetaPaths',
    'Compose'
]

def update_graph_structure(g, data_dict, copy_edata=True):
    r"""

    Description
    -----------
    Update the structure of a graph.

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

    new_g = convert.heterograph(data_dict, num_nodes_dict=num_nodes_dict,
                                idtype=idtype, device=device)

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
    r"""

    Description
    -----------
    An abstract class for writing transforms.
    """
    def __call__(self, g):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'

class AddSelfLoop(BaseTransform):
    r"""

    Description
    -----------
    Add self-loops for each node in the graph and return a new graph.

    For heterogeneous graphs, self-loops are added only for edge types with same
    source and destination node types.

    Parameters
    ----------
    allow_duplicate : bool, optional
        If False, it will first remove self-loops to prevent duplicate self-loops.
    new_etypes : bool, optional
        If True, it will add an edge type 'self' per node type, which holds self-loops.

    Example
    -------

    >>> import dgl
    >>> from dgl import AddSelfLoop

    Case1: Add self-loops for a homogeneous graph

    >>> transform = AddSelfLoop()
    >>> g = dgl.graph(([1, 1], [1, 2]))
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([1, 0, 1, 2]), tensor([2, 0, 1, 2]))

    Case2: Add self-loops for a heterogeneous graph

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0], [1]),
    ...     ('user', 'follows', 'user'): ([1], [2])
    ... })
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype='plays'))
    (tensor([0]), tensor([1]))
    >>> print(new_g.edges(etype='follows'))
    (tensor([1, 0, 1, 2]), tensor([2, 0, 1, 2]))

    Case3: Add self-etypes for a heterogeneous graph

    >>> transform = AddSelfLoop(new_etypes=True)
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype='follows'))
    (tensor([1, 0, 1, 2]), tensor([2, 0, 1, 2]))
    >>> print(new_g.edges(etype=('game', 'self', 'game')))
    (tensor([0, 1]), tensor([0, 1]))
    """
    def __init__(self, allow_duplicate=False, new_etypes=False):
        self.allow_duplicate = allow_duplicate
        self.new_etypes = new_etypes

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
        return functional.add_self_loop(g, etype=c_etype)

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
                data_dict[(ntype, 'self', ntype)] = (nids, nids)

            # Copy edges
            for c_etype in g.canonical_etypes:
                data_dict[c_etype] = g.edges(etype=c_etype)

            g = update_graph_structure(g, data_dict)
        return g

class RemoveSelfLoop(BaseTransform):
    r"""

    Description
    -----------
    Remove self-loops for each node in the graph and return a new graph.

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
        if utype == vtype:
            g = functional.remove_self_loop(g, etype=c_etype)
        return g

    def __call__(self, g):
        for c_etype in g.canonical_etypes:
            g = self.transform_etype(c_etype, g)
        return g

class AddReverse(BaseTransform):
    r"""

    Description
    -----------
    Add a reverse edge :math:`(i,j)` for each edge :math:`(j,i)` in the input graph and
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
        r"""

        Description
        -----------
        Transform the graph corresponding to a symmetric canonical edge type.

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
        r"""

        Description
        -----------
        Transform the graph corresponding to an asymmetric canonical edge type.

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
        data_dict.update({
            c_etype: (src, dst),
            (vtype, 'rev_{}'.format(etype), utype): (dst, src)
        })

    def transform_etype(self, c_etype, g, data_dict):
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
                rev_c_etype = (vtype, 'rev_{}'.format(etype), utype)
                for key, feat in g.edges[c_etype].data.items():
                    new_g.edges[c_etype].data[key] = feat
                    if self.copy_edata:
                        new_g.edges[rev_c_etype].data[key] = feat
            else:
                for key, feat in g.edges[c_etype].data.items():
                    new_feat = feat if self.copy_edata else F.zeros(
                        F.shape(feat), F.dtype(feat), F.context(feat))
                    new_g.edges[c_etype].data[key] = F.cat([feat, new_feat], dim=0)

        return new_g

class ToSimple(BaseTransform):
    r"""

    Description
    -----------
    Convert a graph to a simple graph without parallel edges and return a new graph.

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
    def __init__(self, return_counts='count', aggregator='arbitrary'):
        self.return_counts = return_counts
        self.aggregator = aggregator

    def __call__(self, g):
        return functional.to_simple(g,
                                    return_counts=self.return_counts,
                                    copy_edata=True,
                                    aggregator=self.aggregator)

class LineGraph(BaseTransform):
    r"""

    Description
    -----------
    Return the line graph of the input graph.

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
        return functional.line_graph(g, backtracking=self.backtracking, shared=True)

class KHopGraph(BaseTransform):
    r"""

    Description
    -----------
    Return the graph whose edges connect the :math:`k`-hop neighbors of the original graph.

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
    r"""

    Description
    -----------
    Add new edges to an input graph based on given metapaths, as described in
    `Heterogeneous Graph Attention Network <https://arxiv.org/abs/1903.07293>`__. Formally,
    a metapath is a path of the form

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
    r"""

    Description
    -----------
    Create a transform composed of multiple transforms in sequence.

    Parameters
    ----------
    transforms : list of Callable
        A list of transform objects to apply in order. A transform object should inherit
        :class:`~dgl.BaseTransform` and implement :func:`~dgl.BaseTransform.__call__`.

    Example
    -------

    >>> import dgl
    >>> from dgl import transform as T

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
        args = ['  ' + str(transform) for transform in self.transforms]
        return self.__class__.__name__ + '([\n' + ',\n'.join(args) + '\n])'
