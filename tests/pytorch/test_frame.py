import torch as th
import numpy as np
from dgl.frame import Frame

N = 10
D = 32

def check_eq(a, b):
    assert a.shape == b.shape
    assert th.sum(a == b) == int(np.prod(list(a.shape)))

def create_test_data():
    c1 = th.randn(N, D)
    c2 = th.randn(N, D)
    c3 = th.randn(N, D)
    return {'a1' : c1, 'a2' : c2, 'a3' : c3}

def test_create():
    data = create_test_data()
    f1 = Frame()
    for k, v in data.items():
        f1.add_column(k, v)
    assert f1.schemes == set(data.keys())
    assert f1.num_columns == 3
    assert f1.num_rows == N
    f2 = Frame(data)
    assert f2.schemes == set(data.keys())
    assert f2.num_columns == 3
    assert f2.num_rows == N
    f1.clear()
    assert len(f1.schemes) == 0
    assert f1.num_rows == 0

def test_col_getter_setter():
    data = create_test_data()
    f = Frame(data)
    check_eq(f['a1'], data['a1'])
    f['a1'] = data['a2']
    check_eq(f['a2'], data['a2'])

def test_row_getter_setter():
    data = create_test_data()
    f = Frame(data)

    # getter
    # test non-duplicate keys
    rowid = th.tensor([0, 2])
    rows = f[rowid]
    for k, v in rows:
        assert v.shape == (len(rowid), D)
        check_eq(v, data[k][rowid])
    # test duplicate keys
    rowid = th.tensor([8, 2, 2, 1])
    rows = f[rowid]
    for k, v in rows:
        assert v.shape == (len(rowid), D)
        check_eq(v, data[k][rowid])

    # setter
    rowid = th.tensor([0, 2, 4])
    vals = {'a1' : th.zeros((len(rowid), D)),
            'a2' : th.zeros((len(rowid), D)),
            'a3' : th.zeros((len(rowid), D)),
            }
    f[rowid] = vals
    for k, v in f[rowid]:
        check_eq(v, th.zeros((len(rowid), D)))

def test_row_getter_setter_grad():
    #TODO
    pass

def test_append():
    data = create_test_data()
    f1 = Frame()
    f2 = Frame(data)
    f1.append(data)
    assert f1.num_rows == N
    f1.append(f2)
    assert f1.num_rows == 2 * N
    c1 = f1['a1']
    assert c1.shape == (2 * N, D)
    truth = th.cat([data['a1'], data['a1']])
    check_eq(truth, c1)

if __name__ == '__main__':
    test_create()
    test_col_getter_setter()
    test_append()
    test_row_getter_setter()
    test_row_getter_setter_grad()
