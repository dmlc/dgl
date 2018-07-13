from dgl.graph import DGLGraph

__NUM_GRAPHS__ = "__num_graphs__"

def batch(graph_list):
    """Batch a list of DGLGraphs into one single graph.

    Parameters
    ----------
    graph_list : iterable
        A list of DGLGraphs to be batched.
    """
    attrs = {__NUM_GRAPHS__ : len(graph_list)}
    newgrh = DGLGraph(**attrs)
    # TODO(minjie): tensorize the following loop.
    for i, grh in enumerate(graph_list):
        for u in grh.nodes:
            newgrh.add_node((i, u), **grh.nodes[u])
        for u, v in grh.edges:
            newgrh.add_edge((i, u), (i, v), **grh.edges[u, v])
        # TODO(minjie): handle glb func
        assert len(grh._glb_func) == 0
    return newgrh

def unbatch(graph_batch):
    """Unbatch the graph and return a list of subgraphs.

    Parameters
    ----------
    graph_batch : DGLGraph
        The batched graph.
    """
    num_graphs = graph_batch.graph[__NUM_GRAPHS__]
    graph_list = [DGLGraph() for i in range(num_graphs)]
    # TODO(minjie): tensorize the following loop.
    for bu in graph_batch.nodes:
        i, u = bu
        graph_list[i].add_node(u, **graph_batch.nodes[bu])
    for bu, bv in graph_batch.edges:
        i, u = bu
        j, v = bv
        assert i == j
        graph_list[i].add_edge(u, v, **graph_batch.edges[bu, bv])
    return graph_list
