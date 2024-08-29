import sys


def test_graphbolt_is_not_imported():
    assert (
        "dgl.graphbolt" not in sys.modules
    ), "dgl.graphbolt is already imported"
    import dgl

    assert "dgl.graphbolt" not in sys.modules, "dgl.graphbolt is imported"
