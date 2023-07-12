import dgl.graphbolt as gb
import torch


def test_unique_and_compact_node_pairs_hetero():
    N1 = torch.randint(0, 50, (30,))
    N2 = torch.randint(0, 50, (20,))
    N3 = torch.randint(0, 50, (10,))
    unique_N1, compacted_N1 = torch.unique(N1, return_inverse=True)
    unique_N2, compacted_N2 = torch.unique(N2, return_inverse=True)
    unique_N3, compacted_N3 = torch.unique(N3, return_inverse=True)
    expected_unique_nodes = {
        "n1": unique_N1,
        "n2": unique_N2,
        "n3": unique_N3,
    }
    expected_compacted_pairs = {
        ("n1", "e1", "n2"): (
            compacted_N1[:20],
            compacted_N2,
        ),
        ("n1", "e2", "n3"): (
            compacted_N1[20:30],
            compacted_N3,
        ),
        ("n2", "e3", "n3"): (
            compacted_N2[10:],
            compacted_N3,
        ),
    }
    node_pairs = {
        ("n1", "e1", "n2"): (
            N1[:20],
            N2,
        ),
        ("n1", "e2", "n3"): (
            N1[20:30],
            N3,
        ),
        ("n2", "e3", "n3"): (
            N2[10:],
            N3,
        ),
    }

    unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
        node_pairs
    )
    for etype, pair in compacted_node_pairs.items():
        expected_u, expected_v = expected_compacted_pairs[etype]
        u, v = pair
        assert torch.equal(u, expected_u)
        assert torch.equal(v, expected_v)
    for ntype, nodes in unique_nodes.items():
        expected_nodes = expected_unique_nodes[ntype]
        assert torch.equal(nodes, expected_nodes)


def test_unique_and_compact_node_pairs_homo():
    N = torch.randint(0, 50, (20,))
    expected_unique_N, compacted_N = torch.unique(N, return_inverse=True)
    expected_compacted_pairs = tuple(compacted_N.split(10))

    node_pairs = tuple(N.split(10))
    unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
        node_pairs
    )
    expected_u, expected_v = expected_compacted_pairs
    u, v = compacted_node_pairs
    assert torch.equal(u, expected_u)
    assert torch.equal(v, expected_v)
    assert torch.equal(unique_nodes, expected_unique_N)
