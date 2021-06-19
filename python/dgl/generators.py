"""Module for various graph generator functions."""

from . import backend as F
from . import convert
from . import random

__all__ = ['rand_graph', 'rand_bipartite']

def rand_graph(num_nodes, num_edges, idtype=F.int64, device=F.cpu()):
    """Generate a random graph of the given number of nodes/edges and return.

    It uniformly chooses ``num_edges`` from all possible node pairs and form a graph.
    The random choice is without replacement, which means there will be no multi-edge
    in the resulting graph.

    To control the randomness, set the random seed via :func:`dgl.seed`.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    num_edges : int
        The number of edges
    idtype : int32, int64, optional
        The data type for storing the structure-related graph information
        such as node and edge IDs. It should be a framework-specific data type object
        (e.g., torch.int32). By default, DGL uses int64.
    device : Device context, optional
        The device of the resulting graph. It should be a framework-specific device
        object (e.g., torch.device). By default, DGL stores the graph on CPU.

    Returns
    -------
    DGLGraph
        The generated random graph.

    See Also
    --------
    rand_bipartite

    Examples
    --------
    >>> import dgl
    >>> dgl.rand_graph(100, 10)
    Graph(num_nodes=100, num_edges=10,
          ndata_schemes={}
          edata_schemes={})
    """
    #TODO(minjie): support RNG as one of the arguments.
    eids = random.choice(num_nodes * num_nodes, num_edges, replace=False)
    eids = F.zerocopy_to_numpy(eids)
    rows = F.zerocopy_from_numpy(eids // num_nodes)
    cols = F.zerocopy_from_numpy(eids % num_nodes)
    rows = F.copy_to(F.astype(rows, idtype), device)
    cols = F.copy_to(F.astype(cols, idtype), device)
    return convert.graph((rows, cols),
                         num_nodes=num_nodes,
                         idtype=idtype, device=device)

def rand_bipartite(utype, etype, vtype,
                   num_src_nodes, num_dst_nodes, num_edges,
                   idtype=F.int64, device=F.cpu()):
    """Generate a random uni-directional bipartite graph and return.

    It uniformly chooses ``num_edges`` from all possible node pairs and form a graph.
    The random choice is without replacement, which means there will be no multi-edge
    in the resulting graph.

    To control the randomness, set the random seed via :func:`dgl.seed`.

    Parameters
    ----------
    utype : str, optional
        The name of the source node type.
    etype : str, optional
        The name of the edge type.
    vtype : str, optional
        The name of the destination node type.
    num_src_nodes : int
        The number of source nodes.
    num_dst_nodes : int
        The number of destination nodes.
    num_edges : int
        The number of edges
    idtype : int32, int64, optional
        The data type for storing the structure-related graph information
        such as node and edge IDs. It should be a framework-specific data type object
        (e.g., torch.int32). By default, DGL uses int64.
    device : Device context, optional
        The device of the resulting graph. It should be a framework-specific device
        object (e.g., torch.device). By default, DGL stores the graph on CPU.

    Returns
    -------
    DGLGraph
        The generated random bipartite graph.

    See Also
    --------
    rand_graph

    Examples
    --------
    >>> import dgl
    >>> dgl.rand_bipartite('user', 'buys', 'game', 50, 100, 10)
    Graph(num_nodes={'game': 100, 'user': 50},
          num_edges={('user', 'buys', 'game'): 10},
          metagraph=[('user', 'game', 'buys')])
    """
    #TODO(minjie): support RNG as one of the arguments.
    eids = random.choice(num_src_nodes * num_dst_nodes, num_edges, replace=False)
    eids = F.zerocopy_to_numpy(eids)
    rows = F.zerocopy_from_numpy(eids // num_dst_nodes)
    cols = F.zerocopy_from_numpy(eids % num_dst_nodes)
    rows = F.copy_to(F.astype(rows, idtype), device)
    cols = F.copy_to(F.astype(cols, idtype), device)
    return convert.heterograph({(utype, etype, vtype): (rows, cols)},
                               {utype: num_src_nodes, vtype: num_dst_nodes},
                               idtype=idtype, device=device)
