import os
import tempfile
import time
import unittest
import warnings

import backend as F

import dgl
import dgl.ndarray as nd
import numpy as np
import pytest
import scipy as sp
from dgl.data.utils import load_labels, load_tensors, save_tensors

np.random.seed(44)


def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format="coo") != 0).astype(
        np.int64
    )
    return dgl.from_scipy(arr)


def construct_graph(n):
    g_list = []
    for _ in range(n):
        g = generate_rand_graph(30)
        g.edata["e1"] = F.randn((g.num_edges(), 32))
        g.edata["e2"] = F.ones((g.num_edges(), 32))
        g.ndata["n1"] = F.randn((g.num_nodes(), 64))
        g_list.append(g)
    return g_list


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_graph_serialize_with_feature():
    num_graphs = 100

    t0 = time.time()

    g_list = construct_graph(num_graphs)

    t1 = time.time()

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    dgl.save_graphs(path, g_list)

    t2 = time.time()
    idx_list = np.random.permutation(np.arange(num_graphs)).tolist()
    loadg_list, _ = dgl.load_graphs(path, idx_list)

    t3 = time.time()
    idx = idx_list[0]
    load_g = loadg_list[0]
    print("Save time: {} s".format(t2 - t1))
    print("Load time: {} s".format(t3 - t2))
    print("Graph Construction time: {} s".format(t1 - t0))

    assert F.allclose(load_g.nodes(), g_list[idx].nodes())

    load_edges = load_g.all_edges("uv", "eid")
    g_edges = g_list[idx].all_edges("uv", "eid")
    assert F.allclose(load_edges[0], g_edges[0])
    assert F.allclose(load_edges[1], g_edges[1])
    assert F.allclose(load_g.edata["e1"], g_list[idx].edata["e1"])
    assert F.allclose(load_g.edata["e2"], g_list[idx].edata["e2"])
    assert F.allclose(load_g.ndata["n1"], g_list[idx].ndata["n1"])

    os.unlink(path)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_graph_serialize_without_feature():
    num_graphs = 100
    g_list = [generate_rand_graph(30) for _ in range(num_graphs)]

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    dgl.save_graphs(path, g_list)

    idx_list = np.random.permutation(np.arange(num_graphs)).tolist()
    loadg_list, _ = dgl.load_graphs(path, idx_list)

    idx = idx_list[0]
    load_g = loadg_list[0]

    assert F.allclose(load_g.nodes(), g_list[idx].nodes())

    load_edges = load_g.all_edges("uv", "eid")
    g_edges = g_list[idx].all_edges("uv", "eid")
    assert F.allclose(load_edges[0], g_edges[0])
    assert F.allclose(load_edges[1], g_edges[1])

    os.unlink(path)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_graph_serialize_with_labels():
    num_graphs = 100
    g_list = [generate_rand_graph(30) for _ in range(num_graphs)]
    labels = {"label": F.zeros((num_graphs, 1))}

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    dgl.save_graphs(path, g_list, labels)

    idx_list = np.random.permutation(np.arange(num_graphs)).tolist()
    loadg_list, l_labels0 = dgl.load_graphs(path, idx_list)
    l_labels = load_labels(path)
    assert F.allclose(l_labels["label"], labels["label"])
    assert F.allclose(l_labels0["label"], labels["label"])

    idx = idx_list[0]
    load_g = loadg_list[0]

    assert F.allclose(load_g.nodes(), g_list[idx].nodes())

    load_edges = load_g.all_edges("uv", "eid")
    g_edges = g_list[idx].all_edges("uv", "eid")
    assert F.allclose(load_edges[0], g_edges[0])
    assert F.allclose(load_edges[1], g_edges[1])

    os.unlink(path)


def test_serialize_tensors():
    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    tensor_dict = {
        "a": F.tensor([1, 3, -1, 0], dtype=F.int64),
        "1@1": F.tensor([1.5, 2], dtype=F.float32),
    }

    save_tensors(path, tensor_dict)

    load_tensor_dict = load_tensors(path)

    for key in tensor_dict:
        assert key in load_tensor_dict
        assert np.array_equal(
            F.asnumpy(load_tensor_dict[key]), F.asnumpy(tensor_dict[key])
        )

    load_nd_dict = load_tensors(path, return_dgl_ndarray=True)

    for key in tensor_dict:
        assert key in load_nd_dict
        assert isinstance(load_nd_dict[key], nd.NDArray)
        assert np.array_equal(
            load_nd_dict[key].asnumpy(), F.asnumpy(tensor_dict[key])
        )

    os.unlink(path)


def test_serialize_empty_dict():
    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    tensor_dict = {}

    save_tensors(path, tensor_dict)

    load_tensor_dict = load_tensors(path)
    assert isinstance(load_tensor_dict, dict)
    assert len(load_tensor_dict) == 0

    os.unlink(path)


def load_old_files(files):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return dgl.load_graphs(os.path.join(os.path.dirname(__file__), files))


def test_load_old_files1():
    loadg_list, _ = load_old_files("data/1.bin")
    idx, num_nodes, edge0, edge1, edata_e1, edata_e2, ndata_n1 = np.load(
        os.path.join(os.path.dirname(__file__), "data/1.npy"), allow_pickle=True
    )

    load_g = loadg_list[idx]
    load_edges = load_g.all_edges("uv", "eid")

    assert np.allclose(F.asnumpy(load_edges[0]), edge0)
    assert np.allclose(F.asnumpy(load_edges[1]), edge1)
    assert np.allclose(F.asnumpy(load_g.edata["e1"]), edata_e1)
    assert np.allclose(F.asnumpy(load_g.edata["e2"]), edata_e2)
    assert np.allclose(F.asnumpy(load_g.ndata["n1"]), ndata_n1)


def test_load_old_files2():
    loadg_list, labels0 = load_old_files("data/2.bin")
    labels1 = load_labels(os.path.join(os.path.dirname(__file__), "data/2.bin"))
    idx, edges0, edges1, np_labels = np.load(
        os.path.join(os.path.dirname(__file__), "data/2.npy"), allow_pickle=True
    )
    assert np.allclose(F.asnumpy(labels0["label"]), np_labels)
    assert np.allclose(F.asnumpy(labels1["label"]), np_labels)

    load_g = loadg_list[idx]
    print(load_g)
    load_edges = load_g.all_edges("uv", "eid")
    assert np.allclose(F.asnumpy(load_edges[0]), edges0)
    assert np.allclose(F.asnumpy(load_edges[1]), edges1)


def create_heterographs(idtype):
    g_x = dgl.heterograph(
        {("user", "follows", "user"): ([0, 1, 2], [1, 2, 3])}, idtype=idtype
    )
    g_y = dgl.heterograph(
        {("user", "knows", "user"): ([0, 2], [2, 3])}, idtype=idtype
    ).formats("csr")
    g_x.ndata["h"] = F.randn((4, 3))
    g_x.edata["w"] = F.randn((3, 2))
    g_y.ndata["hh"] = F.ones((4, 5))
    g_y.edata["ww"] = F.randn((2, 10))
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1, 2], [1, 2, 3]),
            ("user", "knows", "user"): ([0, 2], [2, 3]),
        },
        idtype=idtype,
    )
    g.nodes["user"].data["h"] = g_x.ndata["h"]
    g.nodes["user"].data["hh"] = g_y.ndata["hh"]
    g.edges["follows"].data["w"] = g_x.edata["w"]
    g.edges["knows"].data["ww"] = g_y.edata["ww"]
    return [g, g_x, g_y]


def create_heterographs2(idtype):
    g_x = dgl.heterograph(
        {("user", "follows", "user"): ([0, 1, 2], [1, 2, 3])}, idtype=idtype
    )
    g_y = dgl.heterograph(
        {("user", "knows", "user"): ([0, 2], [2, 3])}, idtype=idtype
    ).formats("csr")
    g_z = dgl.heterograph(
        {("user", "knows", "knowledge"): ([0, 1, 3], [2, 3, 4])}, idtype=idtype
    )
    g_x.ndata["h"] = F.randn((4, 3))
    g_x.edata["w"] = F.randn((3, 2))
    g_y.ndata["hh"] = F.ones((4, 5))
    g_y.edata["ww"] = F.randn((2, 10))
    g = dgl.heterograph(
        {
            ("user", "follows", "user"): ([0, 1, 2], [1, 2, 3]),
            ("user", "knows", "user"): ([0, 2], [2, 3]),
            ("user", "knows", "knowledge"): ([0, 1, 3], [2, 3, 4]),
        },
        idtype=idtype,
    )
    g.nodes["user"].data["h"] = g_x.ndata["h"]
    g.edges["follows"].data["w"] = g_x.edata["w"]
    g.nodes["user"].data["hh"] = g_y.ndata["hh"]
    g.edges[("user", "knows", "user")].data["ww"] = g_y.edata["ww"]
    return [g, g_x, g_y, g_z]


def test_deserialize_old_heterograph_file():
    path = os.path.join(os.path.dirname(__file__), "data/hetero1.bin")
    g_list, label_dict = dgl.load_graphs(path)
    assert g_list[0].idtype == F.int64
    assert g_list[3].idtype == F.int32
    assert np.allclose(
        F.asnumpy(g_list[2].nodes["user"].data["hh"]), np.ones((4, 5))
    )
    assert np.allclose(
        F.asnumpy(g_list[5].nodes["user"].data["hh"]), np.ones((4, 5))
    )
    edges = g_list[0]["follows"].edges()
    assert np.allclose(F.asnumpy(edges[0]), np.array([0, 1, 2]))
    assert np.allclose(F.asnumpy(edges[1]), np.array([1, 2, 3]))
    assert F.allclose(label_dict["graph_label"], F.ones(54))


def create_old_heterograph_files():
    path = os.path.join(os.path.dirname(__file__), "data/hetero1.bin")
    g_list0 = create_heterographs(F.int64) + create_heterographs(F.int32)
    labels_dict = {"graph_label": F.ones(54)}
    dgl.save_graphs(path, g_list0, labels_dict)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_serialize_heterograph():
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()
    g_list0 = create_heterographs2(F.int64) + create_heterographs2(F.int32)
    dgl.save_graphs(path, g_list0)

    g_list, _ = dgl.load_graphs(path)
    assert g_list[0].idtype == F.int64
    assert len(g_list[0].canonical_etypes) == 3
    for i in range(len(g_list0)):
        for j, etypes in enumerate(g_list0[i].canonical_etypes):
            assert g_list[i].canonical_etypes[j] == etypes
    # assert g_list[1].restrict_format() == 'any'
    # assert g_list[2].restrict_format() == 'csr'

    assert g_list[4].idtype == F.int32
    assert np.allclose(
        F.asnumpy(g_list[2].nodes["user"].data["hh"]), np.ones((4, 5))
    )
    assert np.allclose(
        F.asnumpy(g_list[6].nodes["user"].data["hh"]), np.ones((4, 5))
    )
    edges = g_list[0]["follows"].edges()
    assert np.allclose(F.asnumpy(edges[0]), np.array([0, 1, 2]))
    assert np.allclose(F.asnumpy(edges[1]), np.array([1, 2, 3]))
    for i in range(len(g_list)):
        assert g_list[i].ntypes == g_list0[i].ntypes
        assert g_list[i].etypes == g_list0[i].etypes

    # test set feature after load_graph
    g_list[3].nodes["user"].data["test"] = F.tensor([0, 1, 2, 4])
    g_list[3].edata["test"] = F.tensor([0, 1, 2])

    os.unlink(path)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
@pytest.mark.skip(reason="lack of permission on CI")
def test_serialize_heterograph_s3():
    path = "s3://dglci-data-test/graph2.bin"
    g_list0 = create_heterographs(F.int64) + create_heterographs(F.int32)
    dgl.save_graphs(path, g_list0)

    g_list = dgl.load_graphs(path, [0, 2, 5])
    assert g_list[0].idtype == F.int64
    # assert g_list[1].restrict_format() == 'csr'
    assert np.allclose(
        F.asnumpy(g_list[1].nodes["user"].data["hh"]), np.ones((4, 5))
    )
    assert np.allclose(
        F.asnumpy(g_list[2].nodes["user"].data["hh"]), np.ones((4, 5))
    )
    edges = g_list[0]["follows"].edges()
    assert np.allclose(F.asnumpy(edges[0]), np.array([0, 1, 2]))
    assert np.allclose(F.asnumpy(edges[1]), np.array([1, 2, 3]))


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
@pytest.mark.parametrize(
    "formats",
    [
        "coo",
        "csr",
        "csc",
        ["coo", "csc"],
        ["coo", "csr"],
        ["csc", "csr"],
        ["coo", "csr", "csc"],
    ],
)
def test_graph_serialize_with_formats(formats):
    num_graphs = 100
    g_list = [generate_rand_graph(30) for _ in range(num_graphs)]

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    dgl.save_graphs(path, g_list, formats=formats)

    idx_list = np.random.permutation(np.arange(num_graphs)).tolist()
    loadg_list, _ = dgl.load_graphs(path, idx_list)

    idx = idx_list[0]
    load_g = loadg_list[0]
    g_formats = load_g.formats()

    # verify formats
    if not isinstance(formats, list):
        formats = [formats]
    for fmt in formats:
        assert fmt in g_formats["created"]

    assert F.allclose(load_g.nodes(), g_list[idx].nodes())

    load_edges = load_g.all_edges("uv", "eid")
    g_edges = g_list[idx].all_edges("uv", "eid")
    assert F.allclose(load_edges[0], g_edges[0])
    assert F.allclose(load_edges[1], g_edges[1])

    os.unlink(path)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_graph_serialize_with_restricted_formats():
    g = dgl.rand_graph(100, 200)
    g = g.formats(["coo"])
    g_list = [g]

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    expect_except = False
    try:
        dgl.save_graphs(path, g_list, formats=["csr"])
    except:
        expect_except = True
    assert expect_except

    os.unlink(path)


@unittest.skipIf(F._default_context_str == "gpu", reason="GPU not implemented")
def test_deserialize_old_graph():
    num_nodes = 100
    num_edges = 200
    path = os.path.join(os.path.dirname(__file__), "data/graph_0.9a220622.dgl")
    g_list, _ = dgl.load_graphs(path)
    g = g_list[0]
    assert "coo" in g.formats()["created"]
    assert "csr" in g.formats()["not created"]
    assert "csc" in g.formats()["not created"]
    assert num_nodes == g.num_nodes()
    assert num_edges == g.num_edges()
