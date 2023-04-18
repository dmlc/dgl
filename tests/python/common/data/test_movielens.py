import unittest

import backend as F

import dgl


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_actor():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    movielens = dgl.data.MovieLensDataset(name='ml-100k', valid_ratio=0.2, test_ratio=0.1, force_reload=True)
    train_graph, valid_graph, test_graph = movielens[0]
    assert train_graph.num_nodes() + valid_graph.num_nodes() + test_graph.num_nodes() == 943 + 1682
    assert train_graph.num_edges() + valid_graph.num_edges() + test_graph.num_edges() == 100000 * 2

    movielens1 = dgl.data.MovieLensDataset(name='ml-100k', valid_ratio=0.2, test_ratio=0.1, force_reload=True, transform=transform)
    train_graph1, valid_graph1, test_graph1 = movielens1[0]
    assert train_graph1.num_edges() - train_graph.num_edges() == train_graph.num_nodes()
    assert valid_graph1.num_edges() - valid_graph.num_edges() == valid_graph.num_nodes()
    assert test_graph1.num_edges() - test_graph.num_edges() == test_graph.num_nodes()
    


    

    

