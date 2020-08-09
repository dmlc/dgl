"""Module for various graph generator functions."""
# pylint: disable= dangerous-default-value

from . import backend as F
from . import convert
from . import random

__all__ = ['rand_graph', 'rand_bipartite']

def rand_graph(num_nodes, num_edges, idtype=F.int64, device=F.cpu(),
               formats=['coo', 'csr', 'csc']):
    """Generate a random graph of the given number of nodes/edges.

    It uniformly chooses ``num_edges`` from all pairs and form a graph.

    TODO(minjie): support RNG as one of the arguments.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    num_edges : int
        The number of edges
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64. Default: int64.
    device : Device context, optional
        Device on which the graph is created. Default: CPU.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.

    Returns
    -------
    DGLHeteroGraph
        Generated random graph.
    """
    eids = random.choice(num_nodes * num_nodes, num_edges, replace=False)
    rows = F.copy_to(F.astype(eids / num_nodes, idtype), device)
    cols = F.copy_to(F.astype(eids % num_nodes, idtype), device)
    g = convert.graph((rows, cols),
                      num_nodes=num_nodes, validate=False,
                      formats=formats,
                      idtype=idtype, device=device)
    return g

def rand_bipartite(num_src_nodes, num_dst_nodes, num_edges,
                   idtype=F.int64, device=F.cpu(),
                   formats=['csr', 'coo', 'csc']):
    """Generate a random bipartite graph of the given number of src/dst nodes and
    number of edges.

    It uniformly chooses ``num_edges`` from all pairs and form a graph.

    Parameters
    ----------
    num_src_nodes : int
        The number of source nodes, the :math:`|U|` in :math:`G=(U,V,E)`.
    num_dst_nodes : int
        The number of destination nodes, the :math:`|V|` in :math:`G=(U,V,E)`.
    num_edges : int
        The number of edges
    idtype : int32, int64, optional
        Integer ID type. Must be int32 or int64. Default: int64.
    device : Device context, optional
        Device on which the graph is created. Default: CPU.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.

    Returns
    -------
    DGLHeteroGraph
        Generated random bipartite graph.
    """
    eids = random.choice(num_src_nodes * num_dst_nodes, num_edges, replace=False)
    rows = F.copy_to(F.astype(eids / num_dst_nodes, idtype), device)
    cols = F.copy_to(F.astype(eids % num_dst_nodes, idtype), device)
    g = convert.bipartite((rows, cols),
                          num_nodes=(num_src_nodes, num_dst_nodes), validate=False,
                          idtype=idtype, device=device,
                          formats=formats)
    return g
