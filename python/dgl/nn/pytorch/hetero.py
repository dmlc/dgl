"""Heterograph NN modules"""
from collections import defaultdict

import torch as th
import torch.nn as nn

__all__ = ['HeteroGraphConv']

class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.

    Examples
    --------

    Create a heterograph with three types of relations and nodes.

    >>> import dgl
    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user') : edges1, 
    ...     ('user', 'plays', 'game') : edges2,
    ...     ('store', 'sells', 'game')  : edges3})

    Create a ``HeteroGraphConv`` that applies different convolution modules to
    different relations. Note that the modules for ``'follows'`` and ``'plays'`` 
    do not share weights.

    >>> import dgl.nn.pytorch as dglnn
    >>> conv = dglnn.HeteroGraphConv({
    ...     'follows' : dglnn.GraphConv(...),
    ...     'plays' : dglnn.GraphConv(...),
    ...     'sells' : dglnn.SAGEConv(...)},
    ...     aggregate='sum')

    Call forward with some ``'user'`` features. This computes new features for both
    ``'user'`` and ``'game'`` nodes.

    >>> import torch as th
    >>> h1 = {'user' : th.randn((g.number_of_nodes('user'), 5))}
    >>> h2 = conv(g, h1)
    >>> print(h2.keys())
    dict_keys(['user', 'game'])

    Call forward with both ``'user'`` and ``'store'`` features. Because both the
    ``'plays'`` and ``'sells'`` relations will update the ``'game'`` features,
    their results are aggregated by the specified method (i.e., summation here).

    >>> f1 = {'user' : ..., 'store' : ...}
    >>> f2 = conv(g, f1)
    >>> print(f2.keys())
    dict_keys(['user', 'game'])

    Call forward with some ``'store'`` features. This only computes new features
    for ``'game'`` nodes.

    >>> g1 = {'store' : ...}
    >>> g2 = conv(g, g1)
    >>> print(g2.keys())
    dict_keys(['game'])

    Call forward with a pair of inputs is allowed and each submodule will also
    be invoked with a pair of inputs.

    >>> x_src = {'user' : ..., 'store' : ...}
    >>> x_dst = {'user' : ..., 'game' : ...}
    >>> y_dst = conv(g, (x_src, x_dst))
    >>> print(y_dst.keys())
    dict_keys(['user', 'game'])

    Parameters
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types. The forward function of each
        module must have a `DGLHeteroGraph` object as the first argument, and
        its second argument is either a tensor object representing the node
        features or a pair of tensor object representing the source and destination
        node features.
    aggregate : str, optional
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.
    
    Attributes
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        self.agg_fn = get_aggregate_fn(aggregate)

    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        outputs = {ty : [] for ty in g.dsttypes}
        if isinstance(inputs, tuple):
            src_inputs, dst_inputs = inputs
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs and dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](rel_graph, (src_inputs[stype], dst_inputs[dtype]))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](rel_graph, inputs[stype])
                outputs[dtype].append(dstdata)
        rsts = {}
        for ty, alist in outputs.items():
            agged = self.agg_fn(alist)
            if agged is not None:
                rsts[ty] = agged
        return rsts

def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.

    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.

    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == 'sum':
        fn = th.sum
    elif agg == 'max':
        fn = th.max
    elif agg == 'min':
        fn = th.min
    elif agg == 'mean':
        fn = th.mean
    elif agg == 'stack':
        fn = lambda inputs, dim : inputs
    else:
        raise DGLError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    def aggfn(inputs):
        if len(inputs) == 0:
            return None
        stacked = th.stack(inputs, dim=0)
        return fn(stacked, dim=0)
    return aggfn
