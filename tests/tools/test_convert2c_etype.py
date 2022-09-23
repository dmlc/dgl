import dgl
import unittest, pytest
from scipy import sparse as spsp
from random import randint
from dgl.distributed import partition_graph
from convert_etype2canonical_etype import etype2canonical_etypes
from collections import Counter

def create_random_hetero(type_n, n, balance=True):
    num_nodes = {}
    for i in range(1, type_n+1):
        num_nodes[f'n{i}'] = n if balance else randint(1, n)
    c_etypes = []
    count = 0
    for i in range(1, type_n):
        for j in range(i+1, type_n+1):
            count += 1
            c_etypes.append((f'n{i}', f'r{count}', f'n{j}'))
    edges = {}
    for etype in c_etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(num_nodes[src_ntype], num_nodes[dst_ntype], density=0.001, format='coo',
                          random_state=100)
        edges[etype] = (arr.row, arr.col)
    return dgl.heterograph(edges, num_nodes), c_etypes

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
@pytest.mark.parametrize("type_n, n, balance", [[5, 100, True]])
def test_get_canonical_etypes(type_n, n, balance):
    # Create random graph
    g, c_etypes = create_random_hetero(type_n, n, balance)
    # Partition the graph
    num_parts = 2
    graph_name = 'convert2c_etype'
    path = '/tmp/random_graph/'
    partition_graph(g, graph_name, num_parts, path)
    # Call function
    # test_c_etypes = etype2canonical_etypes(f'{path}/{graph_name}.json', 1)
    # for c_etype in test_c_etypes:
    #     print(c_etype)
    #     print(test_c_etypes[c_etype])
    #assert Counter(c_etypes) == Counter(test_c_etypes)

if __name__ == "__main__":
    test_get_canonical_etypes(5, 100, True)




