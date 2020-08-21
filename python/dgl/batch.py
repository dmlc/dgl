"""Utilities for batching/unbatching graphs."""
from collections.abc import Mapping
from collections import defaultdict

from . import backend as F
from .base import ALL, is_all, DGLError, dgl_warning
from . import convert
from . import utils

__all__ = ['batch', 'unbatch', 'batch_hetero', 'unbatch_hetero']

def batch(graphs, ndata=ALL, edata=ALL, *, node_attrs=None, edge_attrs=None):
    r"""Batch a collection of :class:`DGLGraph` s into one graph for more efficient
    graph computation.

    Each input graph becomes one disjoint component of the batched graph. The nodes
    and edges are relabeled to be disjoint segments:

    =================  =========  =================  ===  =========
                       graphs[0]  graphs[1]          ...  graphs[k]
    =================  =========  =================  ===  =========
    Original node ID   0 ~ N_0    0 ~ N_1            ...  0 ~ N_k
    New node ID        0 ~ N_0    N_0+1 ~ N_0+N_1+1  ...  1+\sum_{i=0}^{k-1} N_i ~
                                                          1+\sum_{i=0}^k N_i
    =================  =========  =================  ===  =========

    Because of this, many of the computations on a batched graph are the same as if
    performed on each graph individually, but become much more efficient
    since they can be parallelized easily. This makes ``dgl.batch`` very useful
    for tasks dealing with many graph samples such as graph classification tasks.

    For heterograph inputs, they must share the same set of relations (i.e., node types
    and edge types) and the function will perform batching on each relation one by one.
    Thus, the result is also a heterograph and has the same set of relations as the inputs.

    The numbers of nodes and edges of the input graphs are accessible via the
    :func:`DGLGraph.batch_num_nodes` and :func:`DGLGraph.batch_num_edges` attributes
    of the resulting graph. For homogeneous graphs, they are 1D integer tensors,
    with each element being the number of nodes/edges of the corresponding input graph. For
    heterographs, they are dictionaries of 1D integer tensors, with node
    type or edge type as the keys.

    The function supports batching batched graphs. The batch size of the result
    graph is the sum of the batch sizes of all the input graphs.

    By default, node/edge features are batched by concatenating the feature tensors
    of all input graphs. This thus requires features of the same name to have
    the same data type and feature size. One can pass ``None`` to the ``ndata``
    or ``edata`` argument to prevent feature batching, or pass a list of strings
    to specify which features to batch.

    To unbatch the graph back to a list, use the :func:`dgl.unbatch` function.

    Parameters
    ----------
    graphs : list[DGLGraph]
        Input graphs.
    ndata : list[str], None, optional
        Node features to batch.
    edata : list[str], None, optional
        Edge features to batch.

    Returns
    -------
    DGLGraph
        Batched graph.

    Examples
    --------

    Batch homogeneous graphs

    >>> import dgl
    >>> import torch as th
    >>> # 4 nodes, 3 edges
    >>> g1 = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 3])))
    >>> # 3 nodes, 4 edges
    >>> g2 = dgl.graph((th.tensor([0, 0, 0, 1]), th.tensor([0, 1, 2, 0])))
    >>> bg = dgl.batch([g1, g2])
    >>> bg
    Graph(num_nodes=7, num_edges=7,
          ndata_schemes={}
          edata_schemes={})
    >>> bg.batch_size
    2
    >>> bg.batch_num_nodes()
    tensor([4, 3])
    >>> bg.batch_num_edges()
    tensor([3, 4])
    >>> bg.edges()
    (tensor([0, 1, 2, 4, 4, 4, 5], tensor([1, 2, 3, 4, 5, 6, 4]))

    Batch batched graphs

    >>> bbg = dgl.batch([bg, bg])
    >>> bbg.batch_size
    4
    >>> bbg.batch_num_nodes()
    tensor([4, 3, 4, 3])
    >>> bbg.batch_num_edges()
    tensor([3, 4, 3, 4])

    Batch graphs with feature data

    >>> g1.ndata['x'] = th.zeros(g1.num_nodes(), 3)
    >>> g1.edata['w'] = th.ones(g1.num_edges(), 2)
    >>> g2.ndata['x'] = th.ones(g2.num_nodes(), 3)
    >>> g2.edata['w'] = th.zeros(g2.num_edges(), 2)
    >>> bg = dgl.batch([g1, g2])
    >>> bg.ndata['x']
    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])
    >>> bg.edata['w']
    tensor([[1, 1],
            [1, 1],
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]])

    Batch heterographs

    >>> hg1 = dgl.heterograph({
    ...     ('user', 'plays', 'game') : (th.tensor([0, 1]), th.tensor([0, 0]))})
    >>> hg2 = dgl.heterograph({
    ...     ('user', 'plays', 'game') : (th.tensor([0, 0, 0]), th.tensor([1, 0, 2]))})
    >>> bhg = dgl.batch([hg1, hg2])
    >>> bhg
    Graph(num_nodes={'user': 3, 'game': 4},
          num_edges={('user', 'plays', 'game'): 5},
          metagraph=[('drug', 'game')])
    >>> bhg.batch_size
    2
    >>> bhg.batch_num_nodes()
    {'user' : tensor([2, 1]), 'game' : tensor([1, 3])}
    >>> bhg.batch_num_edges()
    {('user', 'plays', 'game') : tensor([2, 3])}

    See Also
    --------
    unbatch
    """
    if len(graphs) == 0:
        raise DGLError('The input list of graphs cannot be empty.')
    if node_attrs is not None:
        dgl_warning('Arguments node_attrs has been deprecated. Please use'
                    ' ndata instead.')
        ndata = node_attrs
    if edge_attrs is not None:
        dgl_warning('Arguments edge_attrs has been deprecated. Please use'
                    ' edata instead.')
        edata = edge_attrs
    if not (is_all(ndata) or isinstance(ndata, list)):
        raise DGLError('Invalid argument ndata: must be a string list but got {}.'.format(
            type(ndata)))
    if not (is_all(edata) or isinstance(edata, list)):
        raise DGLError('Invalid argument edata: must be a string list but got {}.'.format(
            type(edata)))
    if any(g.is_block for g in graphs):
        raise DGLError("Batching a block is not supported.")

    utils.check_all_same_device(graphs, 'graphs')
    utils.check_all_same_idtype(graphs, 'graphs')
    relations = graphs[0].canonical_etypes
    ntypes = graphs[0].ntypes
    idtype = graphs[0].idtype
    device = graphs[0].device

    # Batch graph structure for each relation graph
    edge_dict = defaultdict(list)
    num_nodes_dict = defaultdict(int)
    for g in graphs:
        for rel in relations:
            srctype, etype, dsttype = rel
            u, v = g.edges(order='eid', etype=rel)
            src = u + num_nodes_dict[srctype]
            dst = v + num_nodes_dict[dsttype]
            edge_dict[rel].append((src, dst))
        for ntype in ntypes:
            num_nodes_dict[ntype] += g.number_of_nodes(ntype)
    for rel in relations:
        src, dst = zip(*edge_dict[rel])
        edge_dict[rel] = (F.cat(src, 0), F.cat(dst, 0))
    retg = convert.heterograph(edge_dict, num_nodes_dict, idtype=idtype, device=device)

    # Compute batch num nodes
    bnn = {}
    for ntype in graphs[0].ntypes:
        bnn[ntype] = F.cat([g.batch_num_nodes(ntype) for g in graphs], 0)
    retg.set_batch_num_nodes(bnn)

    # Compute batch num edges
    bne = {}
    for etype in graphs[0].canonical_etypes:
        bne[etype] = F.cat([g.batch_num_edges(etype) for g in graphs], 0)
    retg.set_batch_num_edges(bne)

    # Batch node feature
    if ndata is not None:
        for ntype in graphs[0].ntypes:
            feat_dicts = [g.nodes[ntype].data for g in graphs if g.number_of_nodes(ntype) > 0]
            ret_feat = _batch_feat_dicts(feat_dicts, ndata, 'nodes["{}"].data'.format(ntype))
            retg.nodes[ntype].data.update(ret_feat)

    # Batch edge feature
    if edata is not None:
        for etype in graphs[0].canonical_etypes:
            feat_dicts = [g.edges[etype].data for g in graphs if g.number_of_edges(etype) > 0]
            ret_feat = _batch_feat_dicts(feat_dicts, edata, 'edges[{}].data'.format(etype))
            retg.edges[etype].data.update(ret_feat)

    return retg

def _batch_feat_dicts(feat_dicts, keys, feat_dict_name):
    """Internal function to batch feature dictionaries.

    Parameters
    ----------
    feat_dicts : list[dict[str, Tensor]]
        Feature dictionary list.
    keys : list[str]
        Feature keys. Can be '__ALL__', meaning batching all features.
    feat_dict_name : str
        Name of the feature dictionary for reporting errors.

    Returns
    -------
    dict[str, Tensor]
        New feature dict.
    """
    if len(feat_dicts) == 0:
        return {}
    # sanity checks
    if is_all(keys):
        utils.check_all_same_keys(feat_dicts, feat_dict_name)
        keys = feat_dicts[0].keys()
    else:
        utils.check_all_have_keys(feat_dicts, keys, feat_dict_name)
    utils.check_all_same_schema(feat_dicts, keys, feat_dict_name)
    # concat features
    ret_feat = {k : F.cat([fd[k] for fd in feat_dicts], 0) for k in keys}
    return ret_feat

def unbatch(g, node_split=None, edge_split=None):
    """Revert the batch operation by split the given graph into a list of small ones.

    This is the reverse operation of :func:``dgl.batch``. If the ``node_split``
    or the ``edge_split`` is not given, it calls :func:`DGLGraph.batch_num_nodes`
    and :func:`DGLGraph.batch_num_edges` of the input graph to get the information.

    If the ``node_split`` or the ``edge_split`` arguments are given,
    it will partition the graph according to the given segments. One must assure
    that the partition is valid -- edges of the i^th graph only connect nodes
    belong to the i^th graph. Otherwise, DGL will throw an error.

    The function supports heterograph input, in which case the two split
    section arguments shall be of dictionary type -- similar to the
    :func:`DGLGraph.batch_num_nodes`
    and :func:`DGLGraph.batch_num_edges` attributes of a heterograph.

    Parameters
    ----------
    g : DGLGraph
        Input graph to unbatch.
    node_split : Tensor, dict[str, Tensor], optional
        Number of nodes of each result graph.
    edge_split : Tensor, dict[str, Tensor], optional
        Number of edges of each result graph.

    Returns
    -------
    list[DGLGraph]
        Unbatched list of graphs.

    Examples
    --------

    Unbatch a batched graph

    >>> import dgl
    >>> import torch as th
    >>> # 4 nodes, 3 edges
    >>> g1 = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 3])))
    >>> # 3 nodes, 4 edges
    >>> g2 = dgl.graph((th.tensor([0, 0, 0, 1]), th.tensor([0, 1, 2, 0])))
    >>> # add features
    >>> g1.ndata['x'] = th.zeros(g1.num_nodes(), 3)
    >>> g1.edata['w'] = th.ones(g1.num_edges(), 2)
    >>> g2.ndata['x'] = th.ones(g2.num_nodes(), 3)
    >>> g2.edata['w'] = th.zeros(g2.num_edges(), 2)
    >>> bg = dgl.batch([g1, g2])
    >>> f1, f2 = dgl.unbatch(bg)
    >>> f1
    Graph(num_nodes=4, num_edges=3,
          ndata_schemes={‘x’ : Scheme(shape=(3,), dtype=torch.float32)}
          edata_schemes={‘w’ : Scheme(shape=(2,), dtype=torch.float32)})
    >>> f2
    Graph(num_nodes=3, num_edges=4,
          ndata_schemes={‘x’ : Scheme(shape=(3,), dtype=torch.float32)}
          edata_schemes={‘w’ : Scheme(shape=(2,), dtype=torch.float32)})

    With provided split arguments:

    >>> g1 = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 3])))
    >>> g2 = dgl.graph((th.tensor([0, 0, 0, 1]), th.tensor([0, 1, 2, 0])))
    >>> g3 = dgl.graph((th.tensor([0]), th.tensor([1])))
    >>> bg = dgl.batch([g1, g2, g3])
    >>> bg.batch_num_nodes()
    tensor([4, 3, 2])
    >>> bg.batch_num_edges()
    tensor([3, 4, 1])
    >>> # unbatch but merge g2 and g3
    >>> f1, f2 = dgl.unbatch(bg, th.tensor([4, 5]), th.tensor([3, 5]))
    >>> f1
    Graph(num_nodes=4, num_edges=3,
          ndata_schemes={}
          edata_schemes={})
    >>> f2
    Graph(num_nodes=5, num_edges=5,
          ndata_schemes={}
          edata_schemes={})

    Heterograph input

    >>> hg1 = dgl.heterograph({
    ...     ('user', 'plays', 'game') : (th.tensor([0, 1]), th.tensor([0, 0]))})
    >>> hg2 = dgl.heterograph({
    ...     ('user', 'plays', 'game') : (th.tensor([0, 0, 0]), th.tensor([1, 0, 2]))})
    >>> bhg = dgl.batch([hg1, hg2])
    >>> f1, f2 = dgl.unbatch(bhg)
    >>> f1
    Graph(num_nodes={'user': 2, 'game': 1},
          num_edges={('user', 'plays', 'game'): 2},
          metagraph=[('drug', 'game')])
    >>> f2
    Graph(num_nodes={'user': 1, 'game': 3},
          num_edges={('user', 'plays', 'game'): 3},
          metagraph=[('drug', 'game')])

    See Also
    --------
    batch
    """
    num_split = None
    # Parse node_split
    if node_split is None:
        node_split = {ntype : g.batch_num_nodes(ntype) for ntype in g.ntypes}
    elif not isinstance(node_split, Mapping):
        if len(g.ntypes) != 1:
            raise DGLError('Must provide a dictionary for argument node_split when'
                           ' there are multiple node types.')
        node_split = {g.ntypes[0] : node_split}
    if node_split.keys() != set(g.ntypes):
        raise DGLError('Must specify node_split for each node type.')
    for split in node_split.values():
        if num_split is not None and num_split != len(split):
            raise DGLError('All node_split and edge_split must specify the same number'
                           ' of split sizes.')
        num_split = len(split)

    # Parse edge_split
    if edge_split is None:
        edge_split = {etype : g.batch_num_edges(etype) for etype in g.canonical_etypes}
    elif not isinstance(edge_split, Mapping):
        if len(g.etypes) != 1:
            raise DGLError('Must provide a dictionary for argument edge_split when'
                           ' there are multiple edge types.')
        edge_split = {g.canonical_etypes[0] : edge_split}
    if edge_split.keys() != set(g.canonical_etypes):
        raise DGLError('Must specify edge_split for each canonical edge type.')
    for split in edge_split.values():
        if num_split is not None and num_split != len(split):
            raise DGLError('All edge_split and edge_split must specify the same number'
                           ' of split sizes.')
        num_split = len(split)

    node_split = {k : F.asnumpy(split).tolist() for k, split in node_split.items()}
    edge_split = {k : F.asnumpy(split).tolist() for k, split in edge_split.items()}

    # Split edges for each relation
    edge_dict_per = [{} for i in range(num_split)]
    for rel in g.canonical_etypes:
        srctype, etype, dsttype = rel
        srcnid_off = dstnid_off = 0
        u, v = g.edges(order='eid', etype=rel)
        us = F.split(u, edge_split[rel], 0)
        vs = F.split(v, edge_split[rel], 0)
        for i, (subu, subv) in enumerate(zip(us, vs)):
            edge_dict_per[i][rel] = (subu - srcnid_off, subv - dstnid_off)
            srcnid_off += node_split[srctype][i]
            dstnid_off += node_split[dsttype][i]
    num_nodes_dict_per = [{k : split[i] for k, split in node_split.items()}
                          for i in range(num_split)]

    # Create graphs
    gs = [convert.heterograph(edge_dict, num_nodes_dict, idtype=g.idtype)
          for edge_dict, num_nodes_dict in zip(edge_dict_per, num_nodes_dict_per)]

    # Unbatch node features
    for ntype in g.ntypes:
        for key, feat in g.nodes[ntype].data.items():
            subfeats = F.split(feat, node_split[ntype], 0)
            for subg, subf in zip(gs, subfeats):
                subg.nodes[ntype].data[key] = subf

    # Unbatch edge features
    for etype in g.canonical_etypes:
        for key, feat in g.edges[etype].data.items():
            subfeats = F.split(feat, edge_split[etype], 0)
            for subg, subf in zip(gs, subfeats):
                subg.edges[etype].data[key] = subf

    return gs


#### DEPRECATED APIS ####
def batch_hetero(*args, **kwargs):
    """DEPREACTED: please use dgl.batch """
    dgl_warning('From v0.5, DGLHeteroGraph is merged into DGLGraph. You can safely'
                ' replace dgl.batch_hetero with dgl.batch')
    return batch(*args, **kwargs)

def unbatch_hetero(*args, **kwargs):
    """DEPREACTED: please use dgl.unbatch """
    dgl_warning('From v0.5, DGLHeteroGraph is merged into DGLGraph. You can safely'
                ' replace dgl.unbatch_hetero with dgl.unbatch')
    return batch(*args, **kwargs)
