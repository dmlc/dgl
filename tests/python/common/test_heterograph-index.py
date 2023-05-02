import unittest

import backend as F

import dgl
import pytest
from dgl import DGLError
from utils import parametrize_idtype


def create_test_heterograph(idtype):
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1], [1, 2]),
            ("user", "plays", "game"): ([0, 1, 2, 1], [0, 0, 1, 1]),
            ("user", "wishes", "game"): ([0, 2], [1, 0]),
            ("developer", "develops", "game"): ([0, 1], [0, 1]),
        },
        idtype=idtype,
        device=F.ctx(),
    )
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


@unittest.skipIf(
    F._default_context_str == "cpu", reason="Need gpu for this test"
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch",
    reason="Pinning graph outplace only supported for PyTorch",
)
@parametrize_idtype
def test_pin_memory(idtype):
    g = create_test_heterograph(idtype)
    g.nodes["user"].data["h"] = F.ones((3, 5))
    g.nodes["game"].data["i"] = F.ones((2, 5))
    g.edges["plays"].data["e"] = F.ones((4, 4))
    g = g.to(F.cpu())
    assert not g.is_pinned()

    # Test pinning a CPU graph.
    g._graph.pin_memory()
    assert not g.is_pinned()
    g._graph = g._graph.pin_memory()
    assert g.is_pinned()
    assert g.device == F.cpu()

    # when clone with a new (different) formats, e.g., g.formats("csc")
    # ensure the new graphs are not pinned
    assert not g.formats("csc").is_pinned()
    assert not g.formats("csr").is_pinned()
    # 'coo' formats is the default and thus not cloned
    assert g.formats("coo").is_pinned()

    # Test pinning a GPU graph will cause error raised.
    g1 = g.to(F.cuda())
    with pytest.raises(DGLError):
        g1._graph.pin_memory()

    # Test pinning an empty homograph
    g2 = dgl.graph(([], []))
    assert not g2.is_pinned()
    g2._graph = g2._graph.pin_memory()
    assert g2.is_pinned()

    # Test pinning heterograph with 0 edge of one relation type
    g3 = dgl.heterograph(
        {("a", "b", "c"): ([0, 1], [1, 2]), ("c", "d", "c"): ([], [])}
    ).astype(idtype)
    g3._graph = g3._graph.pin_memory()
    assert g3.is_pinned()


if __name__ == "__main__":
    pass
