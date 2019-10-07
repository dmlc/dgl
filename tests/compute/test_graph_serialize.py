import backend as F
import numpy as np
import scipy as sp
import time
import tempfile
import os

from dgl import DGLGraph
import dgl
from dgl.data.utils import save_graphs, load_graphs, load_labels

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


if __name__ == "__main__":
    test_graph_serialize_with_feature()
    test_graph_serialize_without_feature()
    test_graph_serialize_with_labels()
