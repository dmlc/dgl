import dgl
import unittest, pytest
from scipy import sparse as spsp
from dgl.distributed import partition_graph
from convert_etype2canonical_etype import etype2canonical_etypes
from collections import Counter
import json

def create_random_hetero(type_n, n):
    num_nodes = {}
    for i in range(1, type_n+1):
        num_nodes[f'n{i}'] = n
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
    return dgl.heterograph(edges, num_nodes), [','.join(c_etype) for c_etype in c_etypes]

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
@pytest.mark.parametrize("type_n, n, num_parts", [[3, 100, 2], [10, 500, 4], [10, 1000, 8]])
def test_get_canonical_etypes(type_n, n, num_parts):
    # Create random graph
    g, expected_c_etypes = create_random_hetero(type_n, n)

    # Partition the graph
    graph_name = 'convert2c_etype'
    path = '/tmp/random_graph/'
    partition_graph(g, graph_name, num_parts, path)
    
    # New version partition code generate config file contains canonical_etypes bornly,
    # to test our code, convert it to old version
    part_config = f'{path}{graph_name}.json'
    expected_etypes = _convert_config2old_version(part_config)

    # Call convert function
    c_etypes = etype2canonical_etypes(f'{path}/{graph_name}.json', 1)
    
    # Check we get all canonical etypes
    assert Counter(expected_c_etypes) == Counter(c_etypes.keys())
    
    # Check the id is match after transform from etypes -> canonical
    etypes = _extract_etype_from_c_etype(c_etypes)
    assert expected_etypes == etypes

def _convert_config2old_version(part_config):
    with open(part_config, 'r+') as config_f:
        config = json.load(config_f)
        if 'etypes' not in config:
            etypes = _extract_etype_from_c_etype(config['canonical_etypes'])
            del config['canonical_etypes']
            config['etypes'] = etypes
            config_f.seek(0)
            json.dump(config, config_f, indent=4)
            config_f.truncate()
        return config['etypes']

def _extract_etype_from_c_etype(c_etypes):
    etypes = {}
    for c_etype, eid in c_etypes.items():
        etype = c_etype.split(',')[1]
        etypes[etype] = eid
    return etypes

if __name__ == "__main__":
    test_get_canonical_etypes(10, 1000, 8)
