# import numpy as F
import torch as F

from dgl.state import NodeDict

# TODO(gaiyu): more test cases

def test_node_dict():
    # Make sure the semantics should be the same as a normal dict.
    nodes = NodeDict()
    nodes[0] = {'k1' : 'n01'}
    nodes[0]['k2'] = 'n02'
    nodes[1] = {}
    nodes[1]['k1'] = 'n11'
    print(nodes)
    for key, value in nodes.items():
        print(key, value)
    print(nodes.items())
    nodes.clear()
    print(nodes)

def test_node_dict_batched():
    nodes = NodeDict()
    n0 = 0
    n1 = 1
    n2 = 2
    # Set node 0, 1, 2 attrs in a batch
    nodes[[n0, n1, n2]] = {'k1' : F.tensor([0, 1, 2]), 'k2' : F.tensor([0, 1, 2])}
    # Query in a batch
    assert F.prod(nodes[[n0, n1]]['k1'] == F.tensor([0, 1]))
    assert F.prod(nodes[[n2, n1]]['k2'] == F.tensor([2, 1]))
    # Set all nodes with the same attribute (not supported, having to be a Python loop)
    # nodes[[n0, n1, n2]]['k1'] = 10
    # assert F.prod(nodes[[n0, n1, n2]]['k1'] == F.tensor([10, 10, 10]))
    print(nodes)

def test_node_dict_batched_tensor():
    nodes = NodeDict()
    n0 = 0
    n1 = 1
    n2 = 2
    # Set node 0, 1, 2 attrs in a batch
    # Each node has a feature vector of shape (10,)
    all_node_features = F.ones([3, 10])
    nodes[[n0, n1, n2]] = {'k' : all_node_features}
    assert nodes[[n0, n1]]['k'].shape == (2, 10)
    assert nodes[[n2, n1, n2, n0]]['k'].shape == (4, 10)

def test_node_dict_tensor_arg():
    nodes = NodeDict()
    # Set node 0, 1, 2 attrs in a batch
    # Each node has a feature vector of shape (10,)
    all_nodes = F.arange(3).int()
    all_node_features = F.ones([3, 10])
    nodes[all_nodes] = {'k' : all_node_features}
    assert nodes[[0, 1]]['k'].shape == (2, 10)
    assert nodes[[2, 1, 2, 0]]['k'].shape == (4, 10)
    query = F.tensor([2, 1, 2, 0, 1])
    assert nodes[query]['k'].shape == (5, 10)

test_node_dict()
test_node_dict_batched()
test_node_dict_batched_tensor()
test_node_dict_tensor_arg()
