import backend as F

import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
from utils import parametrize_idtype

D = 5


def generate_graph(idtype):
    g = dgl.graph([])
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(10)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edges(0, i)
        g.add_edges(i, 9)
    # add a back flow from 9 to 0
    g.add_edges(9, 0)
    g.ndata.update({"f1": F.randn((10,)), "f2": F.randn((10, D))})
    weights = F.randn((17,))
    g.edata.update({"e1": weights, "e2": F.unsqueeze(weights, 1)})
    return g


@parametrize_idtype
def test_v2v_update_all(idtype):
    def _test(fld):
        def message_func(edges):
            return {"m": edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {"m": edges.src[fld] * edges.data["e1"]}
            else:
                return {"m": edges.src[fld] * edges.data["e2"]}

        def reduce_func(nodes):
            return {fld: F.sum(nodes.mailbox["m"], 1)}

        def apply_func(nodes):
            return {fld: 2 * nodes.data[fld]}

        g = generate_graph(idtype)
        # update all
        v1 = g.ndata[fld]
        g.update_all(
            fn.copy_u(u=fld, out="m"), fn.sum(msg="m", out=fld), apply_func
        )
        v2 = g.ndata[fld]
        g.ndata.update({fld: v1})
        g.update_all(message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert F.allclose(v2, v3)
        # update all with edge weights
        v1 = g.ndata[fld]
        g.update_all(
            fn.u_mul_e(fld, "e1", "m"), fn.sum(msg="m", out=fld), apply_func
        )
        v2 = g.ndata[fld]
        g.ndata.update({fld: v1})
        g.update_all(message_func_edge, reduce_func, apply_func)
        v4 = g.ndata[fld]
        assert F.allclose(v2, v4)

    # test 1d node features
    _test("f1")
    # test 2d node features
    _test("f2")


@parametrize_idtype
def test_v2v_snr(idtype):
    u = F.tensor([0, 0, 0, 3, 4, 9], idtype)
    v = F.tensor([1, 2, 3, 9, 9, 0], idtype)

    def _test(fld):
        def message_func(edges):
            return {"m": edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {"m": edges.src[fld] * edges.data["e1"]}
            else:
                return {"m": edges.src[fld] * edges.data["e2"]}

        def reduce_func(nodes):
            return {fld: F.sum(nodes.mailbox["m"], 1)}

        def apply_func(nodes):
            return {fld: 2 * nodes.data[fld]}

        g = generate_graph(idtype)
        # send and recv
        v1 = g.ndata[fld]
        g.send_and_recv(
            (u, v),
            fn.copy_u(u=fld, out="m"),
            fn.sum(msg="m", out=fld),
            apply_func,
        )
        v2 = g.ndata[fld]
        g.ndata.update({fld: v1})
        g.send_and_recv((u, v), message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert F.allclose(v2, v3)
        # send and recv with edge weights
        v1 = g.ndata[fld]
        g.send_and_recv(
            (u, v),
            fn.u_mul_e(fld, "e1", "m"),
            fn.sum(msg="m", out=fld),
            apply_func,
        )
        v2 = g.ndata[fld]
        g.ndata.update({fld: v1})
        g.send_and_recv((u, v), message_func_edge, reduce_func, apply_func)
        v4 = g.ndata[fld]
        assert F.allclose(v2, v4)

    # test 1d node features
    _test("f1")
    # test 2d node features
    _test("f2")


@parametrize_idtype
def test_v2v_pull(idtype):
    nodes = F.tensor([1, 2, 3, 9], idtype)

    def _test(fld):
        def message_func(edges):
            return {"m": edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {"m": edges.src[fld] * edges.data["e1"]}
            else:
                return {"m": edges.src[fld] * edges.data["e2"]}

        def reduce_func(nodes):
            return {fld: F.sum(nodes.mailbox["m"], 1)}

        def apply_func(nodes):
            return {fld: 2 * nodes.data[fld]}

        g = generate_graph(idtype)
        # send and recv
        v1 = g.ndata[fld]
        g.pull(
            nodes,
            fn.copy_u(u=fld, out="m"),
            fn.sum(msg="m", out=fld),
            apply_func,
        )
        v2 = g.ndata[fld]
        g.ndata[fld] = v1
        g.pull(nodes, message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert F.allclose(v2, v3)
        # send and recv with edge weights
        v1 = g.ndata[fld]
        g.pull(
            nodes,
            fn.u_mul_e(fld, "e1", "m"),
            fn.sum(msg="m", out=fld),
            apply_func,
        )
        v2 = g.ndata[fld]
        g.ndata[fld] = v1
        g.pull(nodes, message_func_edge, reduce_func, apply_func)
        v4 = g.ndata[fld]
        assert F.allclose(v2, v4)

    # test 1d node features
    _test("f1")
    # test 2d node features
    _test("f2")


@parametrize_idtype
def test_update_all_multi_fallback(idtype):
    # create a graph with zero in degree nodes
    g = dgl.graph([])
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(10)
    for i in range(1, 9):
        g.add_edges(0, i)
        g.add_edges(i, 9)
    g.ndata["h"] = F.randn((10, D))
    g.edata["w1"] = F.randn((16,))
    g.edata["w2"] = F.randn((16, D))

    def _mfunc_hxw1(edges):
        return {"m1": edges.src["h"] * F.unsqueeze(edges.data["w1"], 1)}

    def _mfunc_hxw2(edges):
        return {"m2": edges.src["h"] * edges.data["w2"]}

    def _rfunc_m1(nodes):
        return {"o1": F.sum(nodes.mailbox["m1"], 1)}

    def _rfunc_m2(nodes):
        return {"o2": F.sum(nodes.mailbox["m2"], 1)}

    def _rfunc_m1max(nodes):
        return {"o3": F.max(nodes.mailbox["m1"], 1)}

    def _afunc(nodes):
        ret = {}
        for k, v in nodes.data.items():
            if k.startswith("o"):
                ret[k] = 2 * v
        return ret

    # compute ground truth
    g.update_all(_mfunc_hxw1, _rfunc_m1, _afunc)
    o1 = g.ndata.pop("o1")
    g.update_all(_mfunc_hxw2, _rfunc_m2, _afunc)
    o2 = g.ndata.pop("o2")
    g.update_all(_mfunc_hxw1, _rfunc_m1max, _afunc)
    o3 = g.ndata.pop("o3")
    # v2v spmv
    g.update_all(
        fn.u_mul_e("h", "w1", "m1"), fn.sum(msg="m1", out="o1"), _afunc
    )
    assert F.allclose(o1, g.ndata.pop("o1"))
    # v2v fallback to e2v
    g.update_all(
        fn.u_mul_e("h", "w2", "m2"), fn.sum(msg="m2", out="o2"), _afunc
    )
    assert F.allclose(o2, g.ndata.pop("o2"))


@parametrize_idtype
def test_pull_multi_fallback(idtype):
    # create a graph with zero in degree nodes
    g = dgl.graph([])
    g = g.astype(idtype).to(F.ctx())
    g.add_nodes(10)
    for i in range(1, 9):
        g.add_edges(0, i)
        g.add_edges(i, 9)
    g.ndata["h"] = F.randn((10, D))
    g.edata["w1"] = F.randn((16,))
    g.edata["w2"] = F.randn((16, D))

    def _mfunc_hxw1(edges):
        return {"m1": edges.src["h"] * F.unsqueeze(edges.data["w1"], 1)}

    def _mfunc_hxw2(edges):
        return {"m2": edges.src["h"] * edges.data["w2"]}

    def _rfunc_m1(nodes):
        return {"o1": F.sum(nodes.mailbox["m1"], 1)}

    def _rfunc_m2(nodes):
        return {"o2": F.sum(nodes.mailbox["m2"], 1)}

    def _rfunc_m1max(nodes):
        return {"o3": F.max(nodes.mailbox["m1"], 1)}

    def _afunc(nodes):
        ret = {}
        for k, v in nodes.data.items():
            if k.startswith("o"):
                ret[k] = 2 * v
        return ret

    # nodes to pull
    def _pull_nodes(nodes):
        # compute ground truth
        g.pull(nodes, _mfunc_hxw1, _rfunc_m1, _afunc)
        o1 = g.ndata.pop("o1")
        g.pull(nodes, _mfunc_hxw2, _rfunc_m2, _afunc)
        o2 = g.ndata.pop("o2")
        g.pull(nodes, _mfunc_hxw1, _rfunc_m1max, _afunc)
        o3 = g.ndata.pop("o3")
        # v2v spmv
        g.pull(
            nodes,
            fn.u_mul_e("h", "w1", "m1"),
            fn.sum(msg="m1", out="o1"),
            _afunc,
        )
        assert F.allclose(o1, g.ndata.pop("o1"))
        # v2v fallback to e2v
        g.pull(
            nodes,
            fn.u_mul_e("h", "w2", "m2"),
            fn.sum(msg="m2", out="o2"),
            _afunc,
        )
        assert F.allclose(o2, g.ndata.pop("o2"))

    # test#1: non-0deg nodes
    nodes = [1, 2, 9]
    _pull_nodes(nodes)
    # test#2: 0deg nodes + non-0deg nodes
    nodes = [0, 1, 2, 9]
    _pull_nodes(nodes)


@parametrize_idtype
def test_spmv_3d_feat(idtype):
    def src_mul_edge_udf(edges):
        return {
            "sum": edges.src["h"]
            * F.unsqueeze(F.unsqueeze(edges.data["h"], 1), 1)
        }

    def sum_udf(nodes):
        return {"h": F.sum(nodes.mailbox["sum"], 1)}

    n = 100
    p = 0.1
    a = sp.random(n, n, p, data_rvs=lambda n: np.ones(n))
    g = dgl.from_scipy(a)
    g = g.astype(idtype).to(F.ctx())
    m = g.num_edges()

    # test#1: v2v with adj data
    h = F.randn((n, 5, 5))
    e = F.randn((m,))

    g.ndata["h"] = h
    g.edata["h"] = e
    g.update_all(
        message_func=fn.u_mul_e("h", "h", "sum"), reduce_func=fn.sum("sum", "h")
    )  # 1
    ans = g.ndata["h"]

    g.ndata["h"] = h
    g.edata["h"] = e
    g.update_all(
        message_func=src_mul_edge_udf, reduce_func=fn.sum("sum", "h")
    )  # 2
    assert F.allclose(g.ndata["h"], ans)

    g.ndata["h"] = h
    g.edata["h"] = e
    g.update_all(message_func=src_mul_edge_udf, reduce_func=sum_udf)  # 3
    assert F.allclose(g.ndata["h"], ans)

    # test#2: e2v
    def src_mul_edge_udf(edges):
        return {"sum": edges.src["h"] * edges.data["h"]}

    h = F.randn((n, 5, 5))
    e = F.randn((m, 5, 5))

    g.ndata["h"] = h
    g.edata["h"] = e
    g.update_all(
        message_func=fn.u_mul_e("h", "h", "sum"), reduce_func=fn.sum("sum", "h")
    )  # 1
    ans = g.ndata["h"]

    g.ndata["h"] = h
    g.edata["h"] = e
    g.update_all(
        message_func=src_mul_edge_udf, reduce_func=fn.sum("sum", "h")
    )  # 2
    assert F.allclose(g.ndata["h"], ans)

    g.ndata["h"] = h
    g.edata["h"] = e
    g.update_all(message_func=src_mul_edge_udf, reduce_func=sum_udf)  # 3
    assert F.allclose(g.ndata["h"], ans)


if __name__ == "__main__":
    test_v2v_update_all()
    test_v2v_snr()
    test_v2v_pull()
    test_v2v_update_all_multi_fn()
    test_v2v_snr_multi_fn()
    test_e2v_update_all_multi_fn()
    test_e2v_snr_multi_fn()
    test_e2v_recv_multi_fn()
    test_update_all_multi_fallback()
    test_pull_multi_fallback()
    test_spmv_3d_feat()
