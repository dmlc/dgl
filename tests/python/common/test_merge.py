import backend as F

import dgl
from utils import parametrize_idtype


@parametrize_idtype
def test_heterograph_merge(idtype):
    g1 = (
        dgl.heterograph({("a", "to", "b"): ([0, 1], [1, 0])})
        .astype(idtype)
        .to(F.ctx())
    )
    g1_n_edges = g1.num_edges(etype="to")
    g1.nodes["a"].data["nh"] = F.randn((2, 3))
    g1.nodes["b"].data["nh"] = F.randn((2, 3))
    g1.edges["to"].data["eh"] = F.randn((2, 3))

    g2 = (
        dgl.heterograph({("a", "to", "b"): ([1, 2, 3], [2, 3, 5])})
        .astype(idtype)
        .to(F.ctx())
    )
    g2.nodes["a"].data["nh"] = F.randn((4, 3))
    g2.nodes["b"].data["nh"] = F.randn((6, 3))
    g2.edges["to"].data["eh"] = F.randn((3, 3))
    g2.add_nodes(3, ntype="a")
    g2.add_nodes(3, ntype="b")

    m = dgl.merge([g1, g2])

    # Check g2's edges and nodes were added to g1's in m.
    m_us = F.asnumpy(m.edges()[0][g1_n_edges:])
    g2_us = F.asnumpy(g2.edges()[0])
    assert all(m_us == g2_us)
    m_vs = F.asnumpy(m.edges()[1][g1_n_edges:])
    g2_vs = F.asnumpy(g2.edges()[1])
    assert all(m_vs == g2_vs)
    for ntype in m.ntypes:
        assert m.num_nodes(ntype=ntype) == max(
            g1.num_nodes(ntype=ntype), g2.num_nodes(ntype=ntype)
        )

        # Check g1's node data was updated with g2's in m.
        for key in m.nodes[ntype].data:
            g2_n_nodes = g2.num_nodes(ntype=ntype)
            updated_g1_ndata = F.asnumpy(m.nodes[ntype].data[key][:g2_n_nodes])
            g2_ndata = F.asnumpy(g2.nodes[ntype].data[key])
            assert all((updated_g1_ndata == g2_ndata).flatten())

    # Check g1's edge data was updated with g2's in m.
    for key in m.edges["to"].data:
        updated_g1_edata = F.asnumpy(m.edges["to"].data[key][g1_n_edges:])
        g2_edata = F.asnumpy(g2.edges["to"].data[key])
        assert all((updated_g1_edata == g2_edata).flatten())
