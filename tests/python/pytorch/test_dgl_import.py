import sys


def test_graphbolt_is_not_imported():
    import dgl

    assert "dgl.graphbolt" not in sys.modules
