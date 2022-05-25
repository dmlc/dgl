import pytest
import backend as F
import dgl
from dgl.dataloading import pin_graph_for_uva
from test_utils import parametrize_dtype


def create_test_heterograph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    }, idtype=idtype, device=F.cpu())
    assert g.idtype == idtype
    return g


@pytest.mark.skipif(F._default_context_str == 'cpu', reason="Need gpu for this test")
@parametrize_dtype
def test_uva_graph(idtype):
    g = create_test_heterograph(idtype)
    assert not g._graph.is_pinned()
    with pin_graph_for_uva(g, 'cuda') as g_pinned:
        assert g_pinned._graph.is_pinned()
    assert not g._graph.is_pinned()
    
