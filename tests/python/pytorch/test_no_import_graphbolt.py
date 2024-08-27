import sys

import dgl


def test_no_import_graphbolt():
    assert "dgl.graphbolt" not in sys.modules


if __name__ == "__main__":
    test_no_import_graphbolt()
