import torch as th
from torch.autograd import Variable
import numpy as np
from dgl.frame import Frame

N = 10
D = 32

def check_eq(a, b):
    assert a.shape == b.shape
    assert th.sum(a == b) == int(np.prod(list(a.shape)))

def create_test_data(grad=False):
    c1 = Variable(th.randn(N, D), requires_grad=grad)
    c2 = Variable(th.randn(N, D), requires_grad=grad)
    c3 = Variable(th.randn(N, D), requires_grad=grad)
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
    for k, v in rows.items():
        assert v.shape == (len(rowid), D)
        check_eq(v, data[k][rowid])
    # test duplicate keys
    rowid = th.tensor([8, 2, 2, 1])
    rows = f[rowid]
    for k, v in rows.items():
        assert v.shape == (len(rowid), D)
        check_eq(v, data[k][rowid])

    # setter
    rowid = th.tensor([0, 2, 4])
    vals = {'a1' : th.zeros((len(rowid), D)),
            'a2' : th.zeros((len(rowid), D)),
            'a3' : th.zeros((len(rowid), D)),
            }
    f[rowid] = vals
    for k, v in f[rowid].items():
        check_eq(v, th.zeros((len(rowid), D)))

def test_row_getter_setter_grad():
    data = create_test_data(grad=True)
    f = Frame(data)

    # getter
    c1 = f['a1']
    # test non-duplicate keys
    rowid = th.tensor([0, 2])
    rows = f[rowid]
    rows['a1'].backward(th.ones((len(rowid), D)))
    check_eq(c1.grad[:,0], th.tensor([1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))
    c1.grad.data.zero_()
    # test duplicate keys
    rowid = th.tensor([8, 2, 2, 1])
    rows = f[rowid]
    rows['a1'].backward(th.ones((len(rowid), D)))
    check_eq(c1.grad[:,0], th.tensor([0., 1., 2., 0., 0., 0., 0., 0., 1., 0.]))
    c1.grad.data.zero_()

    # setter
    c1 = f['a1']
    rowid = th.tensor([0, 2, 4])
    vals = {'a1' : Variable(th.zeros((len(rowid), D)), requires_grad=True),
            'a2' : Variable(th.zeros((len(rowid), D)), requires_grad=True),
            'a3' : Variable(th.zeros((len(rowid), D)), requires_grad=True),
            }
    f[rowid] = vals
    c11 = f['a1']
    c11.backward(th.ones((N, D)))
    check_eq(c1.grad[:,0], th.tensor([0., 1., 0., 1., 0., 1., 1., 1., 1., 1.]))
    check_eq(vals['a1'].grad, th.ones((len(rowid), D)))
    assert vals['a2'].grad is None

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
