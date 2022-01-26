"""Heterograph NN modules"""
from functools import partial
import torch as th
import torch.nn as nn
from ...base import DGLError
from ...convert import to_homogeneous

__all__ = ['HeteroGraphConv', 'HeteroLinearLayer', 'HeteroEmbedding']

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

class HeteroLinearLayer(nn.Module):
    """Apply a linear transformation on the node features of a
    heterogeneous graph and return a homogeneous graph representation.

    The underlying implementation invokes :func:`~dgl.to_homogeneous`. Therefore,
    the returned graph stores the following extra attributes:
    
      - ret_g.ndata[dgl.NID]: the original node IDs.
      - ret_g.edata[dgl.EID]: the original edge IDs.
      - ret_g.ndata[dgl.NTYPE]: the type ID of each node.
      - ret_g.edata[dgl.ETYPE]: the type ID of each edge.

    Parameters
    ----------
    hg : DGLGraph
        A heterogeneous DGL graph.
    out_size : int
        Output feature size.
    feat_name : str
        Which feature to transform.

    Examples
    --------
    >>> hg = dgl.heterograph({('user', 'rate', 'movie') : ...,
                              ('user', 'follows', 'user') : ...})
    >>> print(hg.num_nodes('user'), hg.num_nodes('movie'))
    300 1000
    >>> hg.nodes['user'].data['feat'] = torch.randn(hg.num_nodes('user'), 32)
    >>> hg.nodes['movie'].data['feat'] = torch.randn(hg.num_nodes('movie'), 64)
    >>> layer = dgl.nn.HeteroLinearLayer(hg, 100, 'feat')
    >>> g, feat = layer(hg)
    >>> print(g.num_nodes())
    1300
    >>> print(feat.shape)
    (1300, 100)
    """
    def __init__(self, hg, out_size, feat_name):
        super(HeteroLinearLayer, self).__init__()
        self.feat_name = feat_name
        self.linears = nn.ModuleDict()
        for ntype in hg.ntypes:
            linear = nn.Linear(hg.nodes[ntype].data[feat_name].shape[1], out_size)
            self.linears[ntype] = linear
    
    def forward(self, hg):
        """Forward function
        
        Parameters
        ----------
        hg : DGLGraph
            The input heterogeneous DGL graph.

        Returns
        -------
        g : DGLGraph
            The homogenized DGL graph.
        feat : Tensor
            Transformed node features.
        """
        feat = th.tensor([])
        for ntype in hg.ntypes:
            features = self.linears[ntype](hg.nodes[ntype].data[self.feat_name])
            feat = th.cat((feat, features))
        g = to_homogeneous(hg)   
        
        return g, feat          
    
class HeteroEmbedding(nn.Module):
    """Create node embeddings for each node type and return a homogeneous
    graph representation.

    The underlying implementation invokes :func:`~dgl.to_homogeneous`. Therefore,
    the returned graph stores the following extra attributes:
    
      - ret_g.ndata[dgl.NID]: the original node IDs.
      - ret_g.edata[dgl.EID]: the original edge IDs.
      - ret_g.ndata[dgl.NTYPE]: the type ID of each node.
      - ret_g.edata[dgl.ETYPE]: the type ID of each edge.

    Parameters
    ----------
    hg : DGLGraph
        A heterogeneous DGL graph.
    embed_size : int
        Node embedding size.

    Examples
    --------
    
    >>> hg = dgl.heterograph({('user', 'rate', 'movie') : ...,
                              ('user', 'follows', 'user') : ...})
    >>> print(hg.num_nodes('user'), hg.num_nodes('movie'))
    300 1000
    >>> layer = dgl.nn.HeteroEmbedding(hg, 100)
    >>> g, embed = layer(hg)
    >>> print(g.num_nodes())
    1300
    >>> print(embed.shape)
    (1300, 100)
    >>> print(embed.requires_grad)
    True
    
    """
    def __init__(self, hg, embed_size):
        super(HeteroEmbedding, self).__init__()
        self.embed_size = embed_size
        nodes = hg.num_nodes()      
        self.embed = nn.Parameter(th.FloatTensor(nodes, self.embed_size))
        nn.init.xavier_uniform_(self.embed, gain=nn.init.calculate_gain('relu'))
        
    def forward(self, hg):
        """Forward function
        
        Parameters
        ----------
        hg : DGLGraph
            The input heterogeneous DGL graph.

        Returns
        -------
        g : DGLGraph
            The homogenized DGL graph.
        embed : Tensor
            Node embeddings.
        """
        g = to_homogeneous(hg)
            
        return g, self.embed