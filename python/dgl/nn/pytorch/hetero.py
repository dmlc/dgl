"""Heterograph NN modules"""
from functools import partial
import torch as th
import torch.nn as nn
from ...base import DGLError

__all__ = ['HeteroGraphConv', 'HeteroLinear', 'HeteroEmbedding']

class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.
    If the relation graph has no edge, the corresponding module will not be called.

    Pseudo-code:

    .. code::

        outputs = {nty : [] for nty in g.dsttypes}
        # Apply sub-modules on their associating relation graphs in parallel
        for relation in g.canonical_etypes:
            stype, etype, dtype = relation
            dstdata = relation_submodule(g[relation], ...)
            outputs[dtype].append(dstdata)

        # Aggregate the results for each destination node type
        rsts = {}
        for ntype, ntype_outputs in outputs.items():
            if len(ntype_outputs) != 0:
                rsts[ntype] = aggregate(ntype_outputs)
        return rsts

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
    aggregate : str, callable, optional
        Method for aggregating node features generated by different relations.
        Allowed string values are 'sum', 'max', 'min', 'mean', 'stack'.
        The 'stack' aggregation is performed along the second dimension, whose order
        is deterministic.
        User can also customize the aggregator by providing a callable instance.
        For example, aggregation by summation is equivalent to the follows:

        .. code::

            def my_agg_func(tensors, dsttype):
                # tensors: is a list of tensors to aggregate
                # dsttype: string name of the destination node type for which the
                #          aggregation is performed
                stacked = torch.stack(tensors, dim=0)
                return torch.sum(stacked, dim=0)

    Attributes
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

def _max_reduce_func(inputs, dim):
    return th.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return th.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return th.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return th.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return th.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = th.stack(inputs, dim=0)
    return fn(stacked, dim=0)

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
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        raise DGLError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)

class HeteroLinear(nn.Module):
    """Apply linear transformations on heterogeneous inputs.

    Parameters
    ----------
    in_size : dict[key, int]
        Input feature size for heterogeneous inputs. A key can be a string or a tuple of strings.
    out_size : int
        Output feature size.

    Examples
    --------

    >>> import dgl
    >>> import torch
    >>> from dgl.nn import HeteroLinear
    >>> g = dgl.heterograph({
    ...     ('A', 'r1', 'B'): ([0, 1], [1, 2]),
    ...     ('B', 'r2', 'B'): ([1], [2])
    ... })
    >>> g.nodes['A'].data['h'] = torch.randn(2, 3)
    >>> g.nodes['B'].data['h'] = torch.randn(3, 4)
    >>> layer = HeteroLinear(g, 5, 'h')
    >>> out_feats = layer({nty: g.nodes[nty].data['h'] for nty in g.ntypes})
    >>> print(out_feats['A'].shape)
    torch.Size([2, 5])
    >>> print(out_feats['B'].shape)
    torch.Size([3, 5])
    """
    def __init__(self, in_size, out_size):
        super(HeteroLinear, self).__init__()

        self.linears = nn.ModuleDict()
        for typ, typ_in_size in in_size.items():
            self.linears[str(typ)] = nn.Linear(typ_in_size, out_size)

    def forward(self, feat):
        """Forward function

        Parameters
        ----------
        feat : dict[key, Tensor]
            Heterogeneous input features. It maps keys to features.

        Returns
        -------
        dict[key, Tensor]
            Transformed features.
        """
        out_feat = dict()
        for typ, typ_feat in feat.items():
            out_feat[typ] = self.linears[str(typ)](typ_feat)

        return out_feat

class HeteroEmbedding(nn.Module):
    """Create a heterogeneous embedding table.

    It internally contains multiple ``torch.nn.Embedding`` with different dictionary sizes.

    Parameters
    ----------
    num_embeddings : dict[key, int]
        Size of the dictionaries. A key can be a string or a tuple of strings.
    embedding_dim : int
        Size of each embedding vector.

    Examples
    --------

    >>> import dgl
    >>> import torch
    >>> from dgl.nn import HeteroEmbedding
    >>> g = dgl.heterograph({
    ...     ('A', 'r1', 'B'): ([0, 1], [1, 2]),
    ...     ('B', 'r2', 'B'): ([1], [2])
    ... })
    >>> layer = HeteroEmbedding(g, 5)

    >>> # Get the embeddings of all nodes
    >>> embeds = layer.weight
    >>> print(embeds['A'].shape)
    torch.Size([2, 5])
    >>> print(embeds['B'].shape)
    torch.Size([3, 5])

    >>> # Get the embeddings for a node subset
    >>> embeds = layer({'A': torch.LongTensor([0]), 'B': torch.LongTensor([0, 2])})
    >>> print(embeds['A'].shape)
    torch.Size([1, 5])
    >>> print(embeds['B'].shape)
    torch.Size([2, 5])
    """
    def __init__(self, num_embeddings, embedding_dim):
        super(HeteroEmbedding, self).__init__()

        self.embeds = nn.ModuleDict()
        self.raw_keys = dict()
        for typ, typ_num_rows in num_embeddings.items():
            self.embeds[str(typ)] = nn.Embedding(typ_num_rows, embedding_dim)
            self.raw_keys[str(typ)] = typ

    @property
    def weight(self):
        """Get the heterogeneous embedding table

        Returns
        -------
        dict[key, Tensor]
            Heterogeneous embedding table
        """
        return {self.raw_keys[ntype]: emb.weight for ntype, emb in self.embeds.items()}

    def forward(self, input_ids):
        """Forward function

        Parameters
        ----------
        input_ids : dict[key, Tensor]
            The row IDs to retrieve embeddings. It maps a key to key-specific IDs.

        Returns
        -------
        dict[key, Tensor]
            The retrieved embeddings.
        """
        embeds = dict()
        for typ, typ_ids in input_ids.items():
            embeds[typ] = self.embeds[str(typ)](typ_ids)

        return embeds
