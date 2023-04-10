import unittest

import backend as F

import dgl
import numpy as np
from utils import parametrize_idtype


def tree1(idtype):
    """Generate a tree
         0
        / \
       1   2
      / \
     3   4
    Edges are from leaves to root.
    """
    g = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g.add_nodes(5)
    g.add_edges(3, 1)
    g.add_edges(4, 1)
    g.add_edges(1, 0)
    g.add_edges(2, 0)
    g.ndata["h"] = F.tensor([0, 1, 2, 3, 4])
    g.edata["h"] = F.randn((4, 10))
    return g


def tree2(idtype):
    """Generate a tree
         1
        / \
       4   3
      / \
     2   0
    Edges are from leaves to root.
    """
    g = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g.add_nodes(5)
    g.add_edges(2, 4)
    g.add_edges(0, 4)
    g.add_edges(4, 1)
    g.add_edges(3, 1)
    g.ndata["h"] = F.tensor([0, 1, 2, 3, 4])
    g.edata["h"] = F.randn((4, 10))
    return g


@parametrize_idtype
def test_batch_unbatch(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)

    bg = dgl.batch([t1, t2])
    assert bg.num_nodes() == 10
    assert bg.num_edges() == 8
    assert bg.batch_size == 2
    assert F.allclose(bg.batch_num_nodes(), F.tensor([5, 5]))
    assert F.allclose(bg.batch_num_edges(), F.tensor([4, 4]))

    tt1, tt2 = dgl.unbatch(bg)
    assert F.allclose(t1.ndata["h"], tt1.ndata["h"])
    assert F.allclose(t1.edata["h"], tt1.edata["h"])
    assert F.allclose(t2.ndata["h"], tt2.ndata["h"])
    assert F.allclose(t2.edata["h"], tt2.edata["h"])


@parametrize_idtype
def test_batch_unbatch1(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)
    b1 = dgl.batch([t1, t2])
    b2 = dgl.batch([t2, b1])
    assert b2.num_nodes() == 15
    assert b2.num_edges() == 12
    assert b2.batch_size == 3
    assert F.allclose(b2.batch_num_nodes(), F.tensor([5, 5, 5]))
    assert F.allclose(b2.batch_num_edges(), F.tensor([4, 4, 4]))

    s1, s2, s3 = dgl.unbatch(b2)
    assert F.allclose(t2.ndata["h"], s1.ndata["h"])
    assert F.allclose(t2.edata["h"], s1.edata["h"])
    assert F.allclose(t1.ndata["h"], s2.ndata["h"])
    assert F.allclose(t1.edata["h"], s2.edata["h"])
    assert F.allclose(t2.ndata["h"], s3.ndata["h"])
    assert F.allclose(t2.edata["h"], s3.edata["h"])


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support inplace update",
)
@parametrize_idtype
def test_batch_unbatch_frame(idtype):
    """Test module of node/edge frames of batched/unbatched DGLGraphs.
    Also address the bug mentioned in https://github.com/dmlc/dgl/issues/1475.
    """
    t1 = tree1(idtype)
    t2 = tree2(idtype)
    N1 = t1.num_nodes()
    E1 = t1.num_edges()
    N2 = t2.num_nodes()
    E2 = t2.num_edges()
    D = 10
    t1.ndata["h"] = F.randn((N1, D))
    t1.edata["h"] = F.randn((E1, D))
    t2.ndata["h"] = F.randn((N2, D))
    t2.edata["h"] = F.randn((E2, D))

    b1 = dgl.batch([t1, t2])
    b2 = dgl.batch([t2])
    b1.ndata["h"][:N1] = F.zeros((N1, D))
    b1.edata["h"][:E1] = F.zeros((E1, D))
    b2.ndata["h"][:N2] = F.zeros((N2, D))
    b2.edata["h"][:E2] = F.zeros((E2, D))
    assert not F.allclose(t1.ndata["h"], F.zeros((N1, D)))
    assert not F.allclose(t1.edata["h"], F.zeros((E1, D)))
    assert not F.allclose(t2.ndata["h"], F.zeros((N2, D)))
    assert not F.allclose(t2.edata["h"], F.zeros((E2, D)))

    g1, g2 = dgl.unbatch(b1)
    (_g2,) = dgl.unbatch(b2)
    assert F.allclose(g1.ndata["h"], F.zeros((N1, D)))
    assert F.allclose(g1.edata["h"], F.zeros((E1, D)))
    assert F.allclose(g2.ndata["h"], t2.ndata["h"])
    assert F.allclose(g2.edata["h"], t2.edata["h"])
    assert F.allclose(_g2.ndata["h"], F.zeros((N2, D)))
    assert F.allclose(_g2.edata["h"], F.zeros((E2, D)))


@parametrize_idtype
def test_batch_unbatch2(idtype):
    # test setting/getting features after batch
    a = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    a.add_nodes(4)
    a.add_edges(0, [1, 2, 3])
    b = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    b.add_nodes(3)
    b.add_edges(0, [1, 2])
    c = dgl.batch([a, b])
    c.ndata["h"] = F.ones((7, 1))
    c.edata["w"] = F.ones((5, 1))
    assert F.allclose(c.ndata["h"], F.ones((7, 1)))
    assert F.allclose(c.edata["w"], F.ones((5, 1)))


@parametrize_idtype
def test_batch_send_and_recv(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)

    bg = dgl.batch([t1, t2])
    _mfunc = lambda edges: {"m": edges.src["h"]}
    _rfunc = lambda nodes: {"h": F.sum(nodes.mailbox["m"], 1)}
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]

    bg.send_and_recv((u, v), _mfunc, _rfunc)

    t1, t2 = dgl.unbatch(bg)
    assert F.asnumpy(t1.ndata["h"][1]) == 7
    assert F.asnumpy(t2.ndata["h"][4]) == 2


@parametrize_idtype
def test_batch_propagate(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)

    bg = dgl.batch([t1, t2])
    _mfunc = lambda edges: {"m": edges.src["h"]}
    _rfunc = lambda nodes: {"h": F.sum(nodes.mailbox["m"], 1)}
    # get leaves.

    order = []

    # step 1
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]
    order.append((u, v))

    # step 2
    u = [1, 2, 4 + 5, 3 + 5]
    v = [0, 0, 1 + 5, 1 + 5]
    order.append((u, v))

    bg.prop_edges(order, _mfunc, _rfunc)
    t1, t2 = dgl.unbatch(bg)

    assert F.asnumpy(t1.ndata["h"][0]) == 9
    assert F.asnumpy(t2.ndata["h"][1]) == 5


@parametrize_idtype
def test_batched_edge_ordering(idtype):
    g1 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    e1 = F.randn((5, 10))
    g1.edata["h"] = e1
    g2 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g2.add_nodes(6)
    g2.add_edges([0, 1, 2, 5, 4, 5], [1, 2, 3, 4, 3, 0])
    e2 = F.randn((6, 10))
    g2.edata["h"] = e2
    g = dgl.batch([g1, g2])
    r1 = g.edata["h"][g.edge_ids(4, 5)]
    r2 = g1.edata["h"][g1.edge_ids(4, 5)]
    assert F.array_equal(r1, r2)


@parametrize_idtype
def test_batch_no_edge(idtype):
    g1 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    g2 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g2.add_nodes(6)
    g2.add_edges([0, 1, 2, 5, 4, 5], [1, 2, 3, 4, 3, 0])
    g3 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g3.add_nodes(1)  # no edges
    g = dgl.batch([g1, g3, g2])  # should not throw an error


@parametrize_idtype
def test_batch_keeps_empty_data(idtype):
    g1 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g1.ndata["nh"] = F.tensor([])
    g1.edata["eh"] = F.tensor([])
    g2 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g2.ndata["nh"] = F.tensor([])
    g2.edata["eh"] = F.tensor([])
    g = dgl.batch([g1, g2])
    assert "nh" in g.ndata
    assert "eh" in g.edata


def _get_subgraph_batch_info(keys, induced_indices_arr, batch_num_objs):
    """Internal function to compute batch information for subgraphs.
    Parameters
    ----------
    keys : List[str]
        The node/edge type keys.
    induced_indices_arr : List[Tensor]
        The induced node/edge index tensor for all node/edge types.
    batch_num_objs : Tensor
        Number of nodes/edges for each graph in the original batch.
    Returns
    -------
    Mapping[str, Tensor]
        A dictionary mapping all node/edge type keys to the ``batch_num_objs``
        array of corresponding graph.
    """
    bucket_offset = np.expand_dims(
        np.cumsum(F.asnumpy(batch_num_objs), 0), -1
    )  # (num_bkts, 1)
    ret = {}
    for key, induced_indices in zip(keys, induced_indices_arr):
        # NOTE(Zihao): this implementation is not efficient and we can replace it with
        # binary search in the future.
        induced_indices = np.expand_dims(
            F.asnumpy(induced_indices), 0
        )  # (1, num_nodes)
        new_offset = np.sum((induced_indices < bucket_offset), 1)  # (num_bkts,)
        # start_offset = [0] + [new_offset[i-1] for i in range(1, n_bkts)]
        start_offset = np.concatenate([np.zeros((1,)), new_offset[:-1]], 0)
        new_batch_num_objs = new_offset - start_offset
        ret[key] = F.tensor(new_batch_num_objs, dtype=F.dtype(batch_num_objs))
    return ret


@parametrize_idtype
def test_set_batch_info(idtype):
    ctx = F.ctx()

    g1 = dgl.rand_graph(30, 100).astype(idtype).to(F.ctx())
    g2 = dgl.rand_graph(40, 200).astype(idtype).to(F.ctx())
    bg = dgl.batch([g1, g2])
    batch_num_nodes = F.astype(bg.batch_num_nodes(), idtype)
    batch_num_edges = F.astype(bg.batch_num_edges(), idtype)

    # test homogeneous node subgraph
    sg_n = dgl.node_subgraph(bg, list(range(10, 20)) + list(range(50, 60)))
    induced_nodes = sg_n.ndata["_ID"]
    induced_edges = sg_n.edata["_ID"]
    new_batch_num_nodes = _get_subgraph_batch_info(
        bg.ntypes, [induced_nodes], batch_num_nodes
    )
    new_batch_num_edges = _get_subgraph_batch_info(
        bg.canonical_etypes, [induced_edges], batch_num_edges
    )
    sg_n.set_batch_num_nodes(new_batch_num_nodes)
    sg_n.set_batch_num_edges(new_batch_num_edges)
    subg_n1, subg_n2 = dgl.unbatch(sg_n)
    subg1 = dgl.node_subgraph(g1, list(range(10, 20)))
    subg2 = dgl.node_subgraph(g2, list(range(20, 30)))
    assert subg_n1.num_edges() == subg1.num_edges()
    assert subg_n2.num_edges() == subg2.num_edges()

    # test homogeneous edge subgraph
    sg_e = dgl.edge_subgraph(
        bg, list(range(40, 70)) + list(range(150, 200)), relabel_nodes=False
    )
    induced_nodes = F.arange(0, bg.num_nodes(), idtype)
    induced_edges = sg_e.edata["_ID"]
    new_batch_num_nodes = _get_subgraph_batch_info(
        bg.ntypes, [induced_nodes], batch_num_nodes
    )
    new_batch_num_edges = _get_subgraph_batch_info(
        bg.canonical_etypes, [induced_edges], batch_num_edges
    )
    sg_e.set_batch_num_nodes(new_batch_num_nodes)
    sg_e.set_batch_num_edges(new_batch_num_edges)
    subg_e1, subg_e2 = dgl.unbatch(sg_e)
    subg1 = dgl.edge_subgraph(g1, list(range(40, 70)), relabel_nodes=False)
    subg2 = dgl.edge_subgraph(g2, list(range(50, 100)), relabel_nodes=False)
    assert subg_e1.num_nodes() == subg1.num_nodes()
    assert subg_e2.num_nodes() == subg2.num_nodes()


if __name__ == "__main__":
    # test_batch_unbatch()
    # test_batch_unbatch1()
    # test_batch_unbatch_frame()
    # test_batch_unbatch2()
    # test_batched_edge_ordering()
    # test_batch_send_then_recv()
    # test_batch_send_and_recv()
    # test_batch_propagate()
    # test_batch_no_edge()
    test_set_batch_info(F.int32)
