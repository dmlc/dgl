import backend as F
import numpy as np
import scipy as sp
import time
import tempfile
import os
import pytest

from dgl import DGLGraph
import dgl
import dgl.ndarray as nd
from dgl.data.utils import save_graphs, load_graphs, load_labels, save_tensors, load_tensors

np.random.seed(44)


def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1,
                            format='coo') != 0).astype(np.int64)
    return DGLGraph(arr, readonly=True)


def construct_graph(n, readonly=True):
    g_list = []
    for i in range(n):
        g = generate_rand_graph(30)
        g.edata['e1'] = F.randn((g.number_of_edges(), 32))
        g.edata['e2'] = F.ones((g.number_of_edges(), 32))
        g.ndata['n1'] = F.randn((g.number_of_nodes(), 64))
        g.readonly(i % 2 == 0)
        g_list.append(g)
    return g_list


def test_graph_serialize_with_feature():
    num_graphs = 100

    t0 = time.time()

    g_list = construct_graph(num_graphs)

    t1 = time.time()

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    save_graphs(path, g_list)

    t2 = time.time()
    idx_list = np.random.permutation(np.arange(num_graphs)).tolist()
    loadg_list, _ = load_graphs(path, idx_list)

    t3 = time.time()
    idx = idx_list[0]
    load_g = loadg_list[0]
    print("Save time: {} s".format(t2 - t1))
    print("Load time: {} s".format(t3 - t2))
    print("Graph Construction time: {} s".format(t1 - t0))

    assert F.allclose(load_g.nodes(), g_list[idx].nodes())

    load_edges = load_g.all_edges('uv', 'eid')
    g_edges = g_list[idx].all_edges('uv', 'eid')
    assert F.allclose(load_edges[0], g_edges[0])
    assert F.allclose(load_edges[1], g_edges[1])
    assert F.allclose(load_g.edata['e1'], g_list[idx].edata['e1'])
    assert F.allclose(load_g.edata['e2'], g_list[idx].edata['e2'])
    assert F.allclose(load_g.ndata['n1'], g_list[idx].ndata['n1'])

    t4 = time.time()
    bg = dgl.batch(loadg_list)
    t5 = time.time()
    print("Batch time: {} s".format(t5 - t4))

    os.unlink(path)


def test_graph_serialize_without_feature():
    num_graphs = 100
    g_list = [generate_rand_graph(30) for _ in range(num_graphs)]

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    save_graphs(path, g_list)

    idx_list = np.random.permutation(np.arange(num_graphs)).tolist()
    loadg_list, _ = load_graphs(path, idx_list)

    idx = idx_list[0]
    load_g = loadg_list[0]

    assert F.allclose(load_g.nodes(), g_list[idx].nodes())

    load_edges = load_g.all_edges('uv', 'eid')
    g_edges = g_list[idx].all_edges('uv', 'eid')
    assert F.allclose(load_edges[0], g_edges[0])
    assert F.allclose(load_edges[1], g_edges[1])

    os.unlink(path)


def test_graph_serialize_with_labels():
    num_graphs = 100
    g_list = [generate_rand_graph(30) for _ in range(num_graphs)]
    labels = {"label": F.zeros((num_graphs, 1))}

    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    save_graphs(path, g_list, labels)

    idx_list = np.random.permutation(np.arange(num_graphs)).tolist()
    loadg_list, l_labels0 = load_graphs(path, idx_list)
    l_labels = load_labels(path)
    assert F.allclose(l_labels['label'], labels['label'])
    assert F.allclose(l_labels0['label'], labels['label'])

    idx = idx_list[0]
    load_g = loadg_list[0]

    assert F.allclose(load_g.nodes(), g_list[idx].nodes())

    load_edges = load_g.all_edges('uv', 'eid')
    g_edges = g_list[idx].all_edges('uv', 'eid')
    assert F.allclose(load_edges[0], g_edges[0])
    assert F.allclose(load_edges[1], g_edges[1])

    os.unlink(path)


def test_serialize_tensors():
    # create a temporary file and immediately release it so DGL can open it.
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()

    tensor_dict = {"a": F.tensor(
        [1, 3, -1, 0], dtype=F.int64), "1@1": F.tensor([1.5, 2], dtype=F.float32)}

    save_tensors(path, tensor_dict)

    load_tensor_dict = load_tensors(path)

    for key in tensor_dict:
        assert key in load_tensor_dict
        assert np.array_equal(
            F.asnumpy(load_tensor_dict[key]), F.asnumpy(tensor_dict[key]))

    load_nd_dict = load_tensors(path, return_dgl_ndarray=True)

    for key in tensor_dict:
        assert key in load_nd_dict
        assert isinstance(load_nd_dict[key], nd.NDArray)
        assert np.array_equal(
            load_nd_dict[key].asnumpy(), F.asnumpy(tensor_dict[key]))

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


def test_load_old_files1():
    loadg_list, _ = load_graphs(os.path.join(
        os.path.dirname(__file__), "data/1.bin"))
    idx, num_nodes, edge0, edge1, edata_e1, edata_e2, ndata_n1 = np.load(
        os.path.join(os.path.dirname(__file__), "data/1.npy"), allow_pickle=True)

    load_g = loadg_list[idx]
    load_edges = load_g.all_edges('uv', 'eid')

    assert np.allclose(F.asnumpy(load_edges[0]), edge0)
    assert np.allclose(F.asnumpy(load_edges[1]), edge1)
    assert np.allclose(F.asnumpy(load_g.edata['e1']), edata_e1)
    assert np.allclose(F.asnumpy(load_g.edata['e2']), edata_e2)
    assert np.allclose(F.asnumpy(load_g.ndata['n1']), ndata_n1)


def test_load_old_files2():
    loadg_list, labels0 = load_graphs(os.path.join(
        os.path.dirname(__file__), "data/2.bin"))
    labels1 = load_labels(os.path.join(
        os.path.dirname(__file__), "data/2.bin"))
    idx, edges0, edges1, np_labels = np.load(os.path.join(
        os.path.dirname(__file__), "data/2.npy"), allow_pickle=True)
    assert np.allclose(F.asnumpy(labels0['label']), np_labels)
    assert np.allclose(F.asnumpy(labels1['label']), np_labels)

    load_g = loadg_list[idx]
    load_edges = load_g.all_edges('uv', 'eid')
    assert np.allclose(F.asnumpy(load_edges[0]), edges0)
    assert np.allclose(F.asnumpy(load_edges[1]), edges1)


def create_heterographs(index_dtype):
    g_x = dgl.graph(([0, 1, 2], [1, 2, 3]), 'user',
                    'follows', index_dtype=index_dtype, restrict_format='any')
    g_y = dgl.graph(([0, 2], [2, 3]), 'user', 'knows', index_dtype=index_dtype, restrict_format='csr')
    g_x.nodes['user'].data['h'] = F.randn((4, 3))
    g_x.edges['follows'].data['w'] = F.randn((3, 2))
    g_y.nodes['user'].data['hh'] = F.ones((4, 5))
    g_y.edges['knows'].data['ww'] = F.randn((2, 10))
    g = dgl.hetero_from_relations([g_x, g_y])
    return [g, g_x, g_y]


def test_serialize_old_heterograph_file():
    path = os.path.join(
        os.path.dirname(__file__), "data/hetero1.bin")
    g_list = load_graphs(path)
    assert g_list[0].idtype == F.int64
    assert g_list[3].idtype == F.int32
    assert np.allclose(
        F.asnumpy(g_list[2].nodes['user'].data['hh']), np.ones((4, 5)))
    assert np.allclose(
        F.asnumpy(g_list[5].nodes['user'].data['hh']), np.ones((4, 5)))
    edges = g_list[0]['follows'].edges()
    assert np.allclose(F.asnumpy(edges[0]), np.array([0, 1, 2]))
    assert np.allclose(F.asnumpy(edges[1]), np.array([1, 2, 3]))

def create_old_heterograph_files():
    path = os.path.join(
        os.path.dirname(__file__), "data/hetero1.bin")
    g_list0 = create_heterographs("int64") + create_heterographs("int32")
    save_graphs(path, g_list0)


def test_serialize_heterograph():
    f = tempfile.NamedTemporaryFile(delete=False)
    path = f.name
    f.close()
    g_list0 = create_heterographs("int64") + create_heterographs("int32")
    save_graphs(path, g_list0)

    g_list = load_graphs(path)
    assert g_list[0].idtype == F.int64
    assert g_list[1].restrict_format() == 'any'
    assert g_list[2].restrict_format() == 'csr'
    assert g_list[3].idtype == F.int32
    assert np.allclose(
        F.asnumpy(g_list[2].nodes['user'].data['hh']), np.ones((4, 5)))
    assert np.allclose(
        F.asnumpy(g_list[5].nodes['user'].data['hh']), np.ones((4, 5)))
    edges = g_list[0]['follows'].edges()
    assert np.allclose(F.asnumpy(edges[0]), np.array([0, 1, 2]))
    assert np.allclose(F.asnumpy(edges[1]), np.array([1, 2, 3]))
    for i in range(len(g_list)):
        assert g_list[i].ntypes == g_list0[i].ntypes
        assert g_list[i].etypes == g_list0[i].etypes

    os.unlink(path)

@pytest.mark.skip(reason="lack of permission on CI")
def test_serialize_heterograph_s3():
    # f = tempfile.NamedTemporaryFile(delete=False)
    path = "s3://dglci-data-test/graph2.bin"
    # f.close()
    g_list0 = create_heterographs("int64") + create_heterographs("int32")
    save_graphs(path, g_list0)

    g_list = load_graphs(path, [0, 2, 5])
    assert g_list[0].idtype == F.int64
    assert g_list[1].restrict_format() == 'csr'
    assert np.allclose(
        F.asnumpy(g_list[1].nodes['user'].data['hh']), np.ones((4, 5)))
    assert np.allclose(
        F.asnumpy(g_list[2].nodes['user'].data['hh']), np.ones((4, 5)))
    edges = g_list[0]['follows'].edges()
    assert np.allclose(F.asnumpy(edges[0]), np.array([0, 1, 2]))
    assert np.allclose(F.asnumpy(edges[1]), np.array([1, 2, 3]))



if __name__ == "__main__":
    # test_graph_serialize_with_feature()
    # test_graph_serialize_without_feature()
    # test_graph_serialize_with_labels()
    # test_serialize_tensors()
    # test_serialize_empty_dict()
    # test_load_old_files1()
    # test_load_old_files2()
    # test_serialize_heterograph()
    # test_serialize_heterograph_s3()
    # test_serialize_old_heterograph_file()
