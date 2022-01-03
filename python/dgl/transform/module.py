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
    'AddSelfLoop'
]

class BaseTransform:
    r"""

    Description
    -----------
    An abstract class for writing transforms.
    """
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
    >>>
    """
    def __init__(self, remove_first=False, self_etypes=False):
        self.remove_first = remove_first
        self.self_etypes = self_etypes

    def __call__(self, g):
        for c_etype in g.canonical_etypes:
            utype, _, vtype = c_etype
            if utype != vtype:
                continue
            if self.remove_first:
                g = functional.remove_self_loop(g, etype=c_etype)
            g = functional.add_self_loop(g, etype=c_etype)

        if self.self_etypes:
            device = g.device
            idtype = g.idtype
            edge_dict = dict()
            num_nodes_dict = dict()

            # Add self etypes
            for ntype in g.ntypes:
                num_nodes = g.num_nodes(ntype)
                nids = F.arange(0, num_nodes, idtype, device)
                edge_dict[(ntype, 'self', ntype)] = (nids, nids)
                num_nodes_dict[ntype] = num_nodes

            # Copy edges
            for c_etype in g.canonical_etypes:
                edge_dict[c_etype] = g.edges(etype=c_etype)

            new_g = convert.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

            # Copy features
            for ntype in g.ntypes:
                for key, feat in g.nodes[ntype].data.items():
                    new_g.nodes[ntype].data[key] = feat

            for c_etype in g.canonical_etypes:
                for key, feat in g.edges[c_etype].data.items():
                    new_g.edges[c_etype].data[key] = feat
            g = new_g
        return g
