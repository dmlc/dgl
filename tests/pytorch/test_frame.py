import torch as th
from torch.autograd import Variable
import numpy as np
from dgl.frame import Frame, FrameRef

N = 10
D = 5

def check_eq(a, b):
    return a.shape == b.shape and np.allclose(a.numpy(), b.numpy())

def check_fail(fn):
    try:
        fn()
        return False
    except:
        return True

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

def test_column1():
    # Test frame column getter/setter
    data = create_test_data()
    f = Frame(data)
    assert f.num_rows == N
    assert len(f) == 3
    assert check_eq(f['a1'], data['a1'])
    f['a1'] = data['a2']
    assert check_eq(f['a2'], data['a2'])
    # add a different length column should fail
    def failed_add_col():
        f['a4'] = th.zeros([N+1, D])
    assert check_fail(failed_add_col)
    # delete all the columns
    del f['a1']
    del f['a2']
    assert len(f) == 1
    del f['a3']
    assert f.num_rows == 0
    assert len(f) == 0
    # add a different length column should succeed
    f['a4'] = th.zeros([N+1, D])
    assert f.num_rows == N+1
    assert len(f) == 1

def test_column2():
    # Test frameref column getter/setter
    data = Frame(create_test_data())
    f = FrameRef(data, [3, 4, 5, 6, 7])
    assert f.num_rows == 5
    assert len(f) == 3
    assert check_eq(f['a1'], data['a1'][3:8])
    # set column should reflect on the referenced data
    f['a1'] = th.zeros([5, D])
    assert check_eq(data['a1'][3:8], th.zeros([5, D]))
    # add new column should be padded with zero
    f['a4'] = th.ones([5, D])
    assert len(data) == 4
    assert check_eq(data['a4'][0:3], th.zeros([3, D]))
    assert check_eq(data['a4'][3:8], th.ones([5, D]))
    assert check_eq(data['a4'][8:10], th.zeros([2, D]))

def test_append1():
    # test append API on Frame
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
    assert check_eq(truth, c1)

def test_append2():
    # test append on FrameRef
    data = Frame(create_test_data())
    f = FrameRef(data)
    assert f.is_contiguous()
    assert f.is_span_whole_column()
    assert f.num_rows == N
    # append on the underlying frame should not reflect on the ref
    data.append(data)
    assert f.is_contiguous()
    assert not f.is_span_whole_column()
    assert f.num_rows == N
    # append on the FrameRef should work
    f.append(data)
    assert not f.is_contiguous()
    assert not f.is_span_whole_column()
    assert f.num_rows == 3 * N
    new_idx = list(range(N)) + list(range(2*N, 4*N))
    assert check_eq(f.index_tensor(), th.tensor(new_idx))
    assert data.num_rows == 4 * N

def test_row1():
    # test row getter/setter
    data = create_test_data()
    f = FrameRef(Frame(data))

    # getter
    # test non-duplicate keys
    rowid = th.tensor([0, 2])
    rows = f[rowid]
    for k, v in rows.items():
        assert v.shape == (len(rowid), D)
        assert check_eq(v, data[k][rowid])
    # test duplicate keys
    rowid = th.tensor([8, 2, 2, 1])
    rows = f[rowid]
    for k, v in rows.items():
        assert v.shape == (len(rowid), D)
        assert check_eq(v, data[k][rowid])

    # setter
    rowid = th.tensor([0, 2, 4])
    vals = {'a1' : th.zeros((len(rowid), D)),
            'a2' : th.zeros((len(rowid), D)),
            'a3' : th.zeros((len(rowid), D)),
            }
    f[rowid] = vals
    for k, v in f[rowid].items():
        assert check_eq(v, th.zeros((len(rowid), D)))

def test_row2():
    # test row getter/setter autograd compatibility
    data = create_test_data(grad=True)
    f = FrameRef(Frame(data))

    # getter
    c1 = f['a1']
    # test non-duplicate keys
    rowid = th.tensor([0, 2])
    rows = f[rowid]
    rows['a1'].backward(th.ones((len(rowid), D)))
    assert check_eq(c1.grad[:,0], th.tensor([1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))
    c1.grad.data.zero_()
    # test duplicate keys
    rowid = th.tensor([8, 2, 2, 1])
    rows = f[rowid]
    rows['a1'].backward(th.ones((len(rowid), D)))
    assert check_eq(c1.grad[:,0], th.tensor([0., 1., 2., 0., 0., 0., 0., 0., 1., 0.]))
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
    assert check_eq(c1.grad[:,0], th.tensor([0., 1., 0., 1., 0., 1., 1., 1., 1., 1.]))
    assert check_eq(vals['a1'].grad, th.ones((len(rowid), D)))
    assert vals['a2'].grad is None

def test_row3():
    # test row delete
    data = Frame(create_test_data())
    f = FrameRef(data)
    assert f.is_contiguous()
    assert f.is_span_whole_column()
    assert f.num_rows == N
    del f[th.tensor([2, 3])]
    assert not f.is_contiguous()
    assert not f.is_span_whole_column()
    # delete is lazy: only reflect on the ref while the
    # underlying storage should not be touched
    assert f.num_rows == N - 2
    assert data.num_rows == N
    newidx = list(range(N))
    newidx.pop(2)
    newidx.pop(2)
    for k, v in f.items():
        assert check_eq(v, data[k][th.tensor(newidx)])

def test_sharing():
    data = Frame(create_test_data())
    f1 = FrameRef(data, index=[0, 1, 2, 3])
    f2 = FrameRef(data, index=[2, 3, 4, 5, 6])
    # test read
    for k, v in f1.items():
        assert check_eq(data[k][0:4], v)
    for k, v in f2.items():
        assert check_eq(data[k][2:7], v)
    f2_a1 = f2['a1']
    # test write
    # update own ref should not been seen by the other.
    f1[th.tensor([0, 1])] = {
            'a1' : th.zeros([2, D]),
            'a2' : th.zeros([2, D]),
            'a3' : th.zeros([2, D]),
            }
    assert check_eq(f2['a1'], f2_a1)
    # update shared space should been seen by the other.
    f1[th.tensor([2, 3])] = {
            'a1' : th.ones([2, D]),
            'a2' : th.ones([2, D]),
            'a3' : th.ones([2, D]),
            }
    f2_a1[0:2] = th.ones([2, D])
    assert check_eq(f2['a1'], f2_a1)

if __name__ == '__main__':
    test_create()
    test_column1()
    test_column2()
    test_append1()
    test_append2()
    test_row1()
    test_row2()
    test_row3()
    test_sharing()
