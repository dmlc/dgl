import numpy as np

from dgl.state import NodeDict

def test_node_dict():
    # Make sure the semantics should be the same as a normal dict.
    nodes = NodeDict()
    nodes[0] = {'k1' : 'n01'}
    nodes[0]['k2'] = 'n02'
    nodes[1] = {}
    nodes[1]['k1'] = 'n11'
    print(nodes)
    for k in nodes.items():
        print(k)
    print(nodes.items())
    nodes.clear()
    print(nodes)

def test_node_dict_batched():
    nodes = NodeDict()
    n0 = 0
    n1 = 1
    n2 = 2
    # Set node 0, 1, 2 attrs in a batch
    nodes[[n0, n1, n2]] = {'k1' : [0, 1, 2], 'k2': [0, 1, 2]}
    # Query in a batch
    assert nodes[[n0, n1]]['k1'] == [0, 1]
    assert nodes[[n2, n1]]['k2'] == [2, 1]
    # Set all nodes with the same attribute
    nodes[[n0, n1, n2]]['k1'] = 10
    assert nodes[[n0, n1, n2]]['k1'] == [10, 10, 10]
    print(nodes)

def test_node_dict_batched_tensor():
    nodes = NodeDict()
    n0 = 0
    n1 = 1
    n2 = 2
    # Set node 0, 1, 2 attrs in a batch
    # Each node has a feature vector of shape (10,)
    all_node_features = np.ones((3, 10))
    nodes[[n0, n1, n2]] = {'k' : all_node_features}
    assert nodes[[n0, n1]]['k'].shape == (2, 10)
    assert nodes[[n2, n1, n2, n0]]['k'].shape == (4, 10)

def test_node_dict_tensor_arg():
    nodes = NodeDict()
    # Set node 0, 1, 2 attrs in a batch
    # Each node has a feature vector of shape (10,)
    all_nodes = np.arange(3)
    all_node_features = np.ones((3, 10))
    nodes[all_nodes] = {'k' : all_node_features}
    assert nodes[[0, 1]]['k'].shape == (2, 10)
    assert nodes[[2, 1, 2, 0]]['k'].shape == (4, 10)
    query = np.array([2, 1, 2, 0, 1])
    assert nodes[query]['k'].shape == (5, 10)

test_node_dict()
test_node_dict_batched()
test_node_dict_batched_tensor()
test_node_dict_tensor_arg()
