from dgl import DGLError
from dgl.utils import toindex
from dgl.graph_index import create_graph_index

def test_node_subgraph():
    gi = create_graph_index()
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(0, 3)

    sub2par_nodemap = [2, 0, 3]
    sgi, sub2par_edgemap = gi.node_subgraph(toindex(sub2par_nodemap))

    for s, d, e in zip(*sgi.edges()):
        assert sub2par_edgemap[e] == gi.edge_id(
                sub2par_nodemap[s], sub2par_nodemap[d])
