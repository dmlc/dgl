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
    'KNNGraph',
    'Compose'
]

# pylint: disable=E0001
class BaseTransform:
    r"""

    Description
    -----------
    An abstract class for writing transforms.
    """
    def transform_symmetric_etype(self, c_etype, *args):
        r"""

        Description
        -----------
        Transform a relation graph whose source and destination node types are identical.

        Parameters
        ----------
        c_etype : 3-tuple of str
            A canonical edge type.
        """
        raise NotImplementedError

    def transform_asymmetric_etype(self, c_etype, *args):
        r"""

        Description
        -----------
        Transform a relation graph whose source and destination node types are distinct.

        Parameters
        ----------
        c_etype : 3-tuple of str
            A canonical edge type.
        """
        raise NotImplementedError

    def transform_etype(self, c_etype, *args):
        r"""

        Description
        -----------
        Transform a relation graph.

        Parameters
        ----------
        c_etype : 3-tuple of str
            A canonical edge type.
        """
        utype, _, vtype = c_etype
        if utype == vtype:
            return self.transform_symmetric_etype(c_etype, *args)
        else:
            return self.transform_asymmetric_etype(c_etype, *args)

    def update_graph_structure(self, g, data_dict, copy_edata=True):
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

    def __call__(self, g):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class AddSelfLoop(BaseTransform):
    r"""

    Description
    -----------
    Add self-loops for each node in the graph and return a new graph.

    For heterogeneous graphs, self-loops are added only for edge types with same
    source and destination node types.

    Parameters
    ----------
    remove_first : bool, optional
        If True, it will first remove self-loops to prevent duplicate self-loops.
    add_self_etypes : bool, optional
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
    (tensor([1, 1, 0, 1, 2]), tensor([1, 2, 0, 1, 2]))

    Case2: Remove self-loops first to avoid duplicate self-loops

    >>> transform = AddSelfLoop(remove_first=True)
    >>> new_g = transform(g)
    >>> print(new_g.edges())
    (tensor([1, 0, 1, 2]), tensor([2, 0, 1, 2]))

    Case3: Add self-loops for a heterogeneous graph

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): ([0], [1]),
    ...     ('user', 'follows', 'user'): ([1], [2])
    ... })
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype='plays'))
    (tensor([0]), tensor([1]))
    >>> print(new_g.edges(etype='follows'))
    (tensor([1, 0, 1, 2]), tensor([2, 0, 1, 2]))

    Case4: Add self-etypes for a heterogeneous graph

    >>> transform = AddSelfLoop(self_etypes=True)
    >>> new_g = transform(g)
    >>> print(new_g.edges(etype='follows'))
    (tensor([1, 0, 1, 2]), tensor([2, 0, 1, 2]))
    >>> print(new_g.edges(etype=('game', 'self', 'game')))
    (tensor([0, 1]), tensor([0, 1]))
    """
    def __init__(self, remove_first=False, self_etypes=False):
        self.remove_first = remove_first
        self.self_etypes = self_etypes

    def transform_symmetric_etype(self, c_etype, g):
        if self.remove_first:
            g = functional.remove_self_loop(g, etype=c_etype)
        g = functional.add_self_loop(g, etype=c_etype)
        return g

    def transform_asymmetric_etype(self, c_etype, g):
        return g

    def __call__(self, g):
        for c_etype in g.canonical_etypes:
            g = self.transform_etype(c_etype, g)

        if self.self_etypes:
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

            g = self.update_graph_structure(g, data_dict)
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
    def transform_symmetric_etype(self, c_etype, g):
        return functional.remove_self_loop(g, etype=c_etype)

    def transform_asymmetric_etype(self, c_etype, g):
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
    combine_like : bool, optional
        If True, it will not add a reverse edge type if the source and destination node type
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
    def __init__(self, copy_edata=False, combine_like=True):
        self.copy_edata = copy_edata
        self.combine_like = combine_like

    def transform_symmetric_etype(self, c_etype, g, data_dict):
        if self.combine_like:
            src, dst = g.edges(etype=c_etype)
            src, dst = F.cat([src, dst], dim=0), F.cat([dst, src], dim=0)
            data_dict[c_etype] = (src, dst)
            return data_dict
        else:
            return self.transform_asymmetric_etype(c_etype, g, data_dict)

    def transform_asymmetric_etype(self, c_etype, g, data_dict):
        utype, etype, vtype = c_etype
        src, dst = g.edges(etype=c_etype)
        data_dict.update({
            c_etype: (src, dst),
            (vtype, 'rev_{}'.format(etype), utype): (dst, src)
        })
        return data_dict

    def __call__(self, g):
        data_dict = dict()
        for c_etype in g.canonical_etypes:
            data_dict = self.transform_etype(c_etype, g, data_dict)
        new_g = self.update_graph_structure(g, data_dict, copy_edata=False)

        # Copy and expand edata
        for c_etype in g.canonical_etypes:
            utype, etype, vtype = c_etype
            if utype == vtype and self.combine_like:
                for key, feat in g.edges[c_etype].data.items():
                    new_feat = feat if self.copy_edata else F.zeros(
                        F.shape(feat), F.dtype(feat), F.context(feat))
                    new_g.edges[c_etype].data[key] = F.cat([feat, new_feat], dim=0)
            else:
                rev_c_etype = (vtype, 'rev_{}'.format(etype), utype)
                for key, feat in g.edges[c_etype].data.items():
                    new_g.edges[c_etype].data[key] = feat
                    if self.copy_edata:
                        new_g.edges[rev_c_etype].data[key] = feat

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
    writeback_mapping : bool, optional
        If True, it returns an extra write-back mapping for each edge type.

        * If the input graph has a single edge type, the mapping is a tensor
          recording the mapping from the edge IDs in the input graph to the edge
          IDs in the returned graph.
        * If the input graph has multiple edge types, it returns a dictionary
          mapping edge types to tensors in the above format.
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

    Case2: Convert a homogeneous graph to a simple graph with writeback mapping

    >>> transform = ToSimple(writeback_mapping=True)
    >>> sg, wm = transform(g)
    >>> print(wm)
    tensor([0, 1, 1])

    Case3: Convert a heterogeneous graph to a simple graph

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): ([0, 1, 1], [1, 2, 2]),
    ...     ('user', 'plays', 'game'): ([0, 1, 0], [1, 1, 1])
    ... })
    >>> sg, wm = transform(g)
    >>> print(sg.edges(etype='follows'))
    (tensor([0, 1]), tensor([1, 2]))
    >>> print(sg.edges(etype='plays'))
    (tensor([0, 1]), tensor([1, 1]))
    >>> print(wm)
    {('user', 'follows', 'user'): tensor([0, 1, 1]), ('user', 'plays', 'game'): tensor([0, 1, 0])}
    """
    def __init__(self, return_counts='count', writeback_mapping=False, aggregator='arbitrary'):
        self.return_counts = return_counts
        self.writeback_mapping = writeback_mapping
        self.aggregator = aggregator

    def __call__(self, g):
        return functional.to_simple(g,
                                    return_counts=self.return_counts,
                                    writeback_mapping=self.writeback_mapping,
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
        If True, the line graph node corresponding to :math:`(u, v)` will not have an
        edge connecting to the line graph node corresponding to :math:`(v, u)`.

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
    Add canonical edge types to an input graph based on given metapaths, as described in
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
            new_g = self.update_graph_structure(g, data_dict, copy_edata=True)
        else:
            new_g = self.update_graph_structure(g, data_dict, copy_edata=False)

        return new_g

class KNNGraph(BaseTransform):
    r"""

    Description
    -----------
    Construct a graph from a set of points according to k-nearest-neighbor (KNN)
    and return.

    The function transforms the coordinates/features of a point set
    into a directed homogeneous graph. The coordinates of the point
    set is specified as a matrix whose rows correspond to points and
    columns correspond to coordinate/feature dimensions.

    The nodes of the returned graph correspond to the points, where the predecessors
    of each point are its k-nearest neighbors measured by the chosen distance.

    Parameters
    ----------
    ndata_name : str
        The ndata name to store the node features for KNN computation.
    k : int
        The number of nearest neighbors per node.
    algorithm : str, optional
        Algorithm used to compute the k-nearest neighbors.

        * 'bruteforce-blas' will first compute the distance matrix
          using BLAS matrix multiplication operation provided by
          backend frameworks. Then use topk algorithm to get
          k-nearest neighbors. This method is fast when the point
          set is small but has :math:`O(N^2)` memory complexity where
          :math:`N` is the number of points.

        * 'bruteforce' will compute distances pair by pair and
          directly select the k-nearest neighbors during distance
          computation. This method is slower than 'bruteforce-blas'
          but has less memory overhead (i.e., :math:`O(Nk)` where :math:`N`
          is the number of points, :math:`k` is the number of nearest
          neighbors per node) since we do not need to store all distances.

        * 'bruteforce-sharemem' (CUDA only) is similar to 'bruteforce'
          but use shared memory in CUDA devices for buffer. This method is
          faster than 'bruteforce' when the dimension of input points
          is not large. This method is only available on CUDA device.

        * 'kd-tree' will use the kd-tree algorithm (CPU only).
          This method is suitable for low-dimensional data (e.g. 3D
          point clouds)

        * 'nn-descent' is an approximate approach from paper
          `Efficient k-nearest neighbor graph construction for generic similarity
          measures <https://www.cs.princeton.edu/cass/papers/www11.pdf>`_. This method
          will search for nearest neighbor candidates in "neighbors' neighbors".

        (default: 'bruteforce-blas')
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')

    Example
    -------

    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> from dgl import KNNGraph

    >>> g = dgl.rand_graph(5, 20)
    >>> g.ndata['h'] = torch.randn(g.num_nodes(), 3)
    >>> transform = KNNGraph(ndata_name='h', k=3)
    >>> new_g = transform(g)
    """
    def __init__(self, ndata_name, k, algorithm='bruteforce-blas', dist='euclidean'):
        self.ndata_name = ndata_name
        self.k = k
        self.algorithm = algorithm
        self.dist = dist

    def __call__(self, g):
        knn_g = functional.knn_graph(g.ndata[self.ndata_name],
                                     self.k, self.algorithm, self.dist)
        for key, feat in g.ndata.items():
            knn_g.ndata[key] = feat

        return knn_g

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
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))
