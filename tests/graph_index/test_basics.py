from dgl import DGLError
from dgl.utils import toindex
from dgl.graph_index import create_graph_index
from unittest import TestCase

def test_edge_id():
    gi = create_graph_index()

    gi.add_nodes(4)
    gi.add_edge(0, 1)
    eid = gi.edge_id(0, 1).tolist()
    assert len(eid) == 1
    assert eid[0] == 0
    assert not gi.is_multigraph()

    # multiedges
    gi.add_edge(0, 1)
    eid = gi.edge_id(0, 1).tolist()
    assert len(eid) == 2
    assert 1 in eid
    assert 0 in eid
    assert gi.is_multigraph()

    gi.add_edges(toindex([0, 1, 1, 2]), toindex([2, 2, 2, 3]))
    true_eids = {
            (0, 1): [0, 1],
            (0, 2): [2],
            (1, 2): [3, 4],
            (2, 3): [5],
    }

    src, dst, eid = gi.edge_ids(toindex([0, 0, 1, 2]), toindex([1, 2, 2, 3]))
    for s, d, e in zip(src, dst, eid):
        assert e in true_eids[s, d]

    # source broadcasting
    src, dst, eid = gi.edge_ids(toindex([0]), toindex([1, 2]))
    for s, d, e in zip(src, dst, eid):
        assert e in true_eids[s, d]

    # destination broadcasting
    src, dst, eid = gi.edge_ids(toindex([0, 1]), toindex([2]))
    for s, d, e in zip(src, dst, eid):
        assert e in true_eids[s, d]

    gi.clear()
    assert not gi.is_multigraph()
    # the following assumes that grabbing nonexistent edge will throw an error
    try:
        gi.edge_id(0, 1)
        fail = True
    except DGLError:
        fail = False
    finally:
        assert not fail

    gi.add_nodes(4)
    gi.add_edge(0, 1)
    eid = gi.edge_id(0, 1).tolist()
    assert len(eid) == 1
    assert eid[0] == 0


if __name__ == '__main__':
    test_edge_id()
