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
    'ToSimple'
]

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
    Add a reverse edge for each edge in the input graph and return a new graph.

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

"""
class LineGraph(BaseTransform):
    raise NotImplementedError

class KHopGraph(BaseTransform):
    raise NotImplementedError

class MetapathGraph(BaseTransform):
    raise NotImplementedError

class KNNGraph(BaseTransform):
    raise NotImplementedError

class Compose(BaseTransform):
    raise NotImplementedError

class GDC(BaseTransform):
    raise NotImplementedError

class PPR(BaseTransform):
    raise NotImplementedError

class MDK(BaseTransform):
    raise NotImplementedError
"""
