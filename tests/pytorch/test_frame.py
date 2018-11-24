import torch as th
from torch.autograd import Variable
import numpy as np
from dgl.frame import Frame, FrameRef
from dgl.utils import Index, toindex
import utils as U

N = 10
D = 5

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
        f1.update_column(k, v)
    print(f1.schemes)
    assert f1.keys() == set(data.keys())
    assert f1.num_columns == 3
    assert f1.num_rows == N
    f2 = Frame(data)
    assert f2.keys() == set(data.keys())
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
    assert U.allclose(f['a1'].data, data['a1'].data)
    f['a1'] = data['a2']
    assert U.allclose(f['a2'].data, data['a2'].data)
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
    assert U.allclose(f['a1'], data['a1'].data[3:8])
    # set column should reflect on the referenced data
    f['a1'] = th.zeros([5, D])
    assert U.allclose(data['a1'].data[3:8], th.zeros([5, D]))
    # add new partial column should fail with error initializer
    f.set_initializer(lambda shape, dtype : assert_(False))
    def failed_add_col():
        f['a4'] = th.ones([5, D])
    assert check_fail(failed_add_col)

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
    assert c1.data.shape == (2 * N, D)
    truth = th.cat([data['a1'], data['a1']])
    assert U.allclose(truth, c1.data)
    # append dict of different length columns should fail
    f3 = {'a1' : th.zeros((3, D)), 'a2' : th.zeros((3, D)), 'a3' : th.zeros((2, D))}
    def failed_append():
        f1.append(f3)
    assert check_fail(failed_append)

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
    assert th.all(f.index().tousertensor() == th.tensor(new_idx, dtype=th.int64))
    assert data.num_rows == 4 * N

def test_row1():
    # test row getter/setter
    data = create_test_data()
    f = FrameRef(Frame(data))

    # getter
    # test non-duplicate keys
    rowid = Index(th.tensor([0, 2]))
    rows = f[rowid]
    for k, v in rows.items():
        assert v.shape == (len(rowid), D)
        assert U.allclose(v, data[k][rowid])
    # test duplicate keys
    rowid = Index(th.tensor([8, 2, 2, 1]))
    rows = f[rowid]
    for k, v in rows.items():
        assert v.shape == (len(rowid), D)
        assert U.allclose(v, data[k][rowid])

    # setter
    rowid = Index(th.tensor([0, 2, 4]))
    vals = {'a1' : th.zeros((len(rowid), D)),
            'a2' : th.zeros((len(rowid), D)),
            'a3' : th.zeros((len(rowid), D)),
            }
    f[rowid] = vals
    for k, v in f[rowid].items():
        assert U.allclose(v, th.zeros((len(rowid), D)))

    # setting rows with new column should raise error with error initializer
    f.set_initializer(lambda shape, dtype : assert_(False))
    def failed_update_rows():
        vals['a4'] = th.ones((len(rowid), D))
        f[rowid] = vals
    assert check_fail(failed_update_rows)

def test_row2():
    # test row getter/setter autograd compatibility
    data = create_test_data(grad=True)
    f = FrameRef(Frame(data))

    # getter
    c1 = f['a1']
    # test non-duplicate keys
    rowid = Index(th.tensor([0, 2]))
    rows = f[rowid]
    rows['a1'].backward(th.ones((len(rowid), D)))
    assert U.allclose(c1.grad[:,0], th.tensor([1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))
    c1.grad.data.zero_()
    # test duplicate keys
    rowid = Index(th.tensor([8, 2, 2, 1]))
    rows = f[rowid]
    rows['a1'].backward(th.ones((len(rowid), D)))
    assert U.allclose(c1.grad[:,0], th.tensor([0., 1., 2., 0., 0., 0., 0., 0., 1., 0.]))
    c1.grad.data.zero_()

    # setter
    c1 = f['a1']
    rowid = Index(th.tensor([0, 2, 4]))
    vals = {'a1' : Variable(th.zeros((len(rowid), D)), requires_grad=True),
            'a2' : Variable(th.zeros((len(rowid), D)), requires_grad=True),
            'a3' : Variable(th.zeros((len(rowid), D)), requires_grad=True),
            }
    f[rowid] = vals
    c11 = f['a1']
    c11.backward(th.ones((N, D)))
    assert U.allclose(c1.grad[:,0], th.tensor([0., 1., 0., 1., 0., 1., 1., 1., 1., 1.]))
    assert U.allclose(vals['a1'].grad, th.ones((len(rowid), D)))
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
    newidx = toindex(newidx)
    for k, v in f.items():
        assert U.allclose(v, data[k][newidx])

def test_sharing():
    data = Frame(create_test_data())
    f1 = FrameRef(data, index=[0, 1, 2, 3])
    f2 = FrameRef(data, index=[2, 3, 4, 5, 6])
    # test read
    for k, v in f1.items():
        assert U.allclose(data[k].data[0:4], v)
    for k, v in f2.items():
        assert U.allclose(data[k].data[2:7], v)
    f2_a1 = f2['a1'].data
    # test write
    # update own ref should not been seen by the other.
    f1[Index(th.tensor([0, 1]))] = {
            'a1' : th.zeros([2, D]),
            'a2' : th.zeros([2, D]),
            'a3' : th.zeros([2, D]),
            }
    assert U.allclose(f2['a1'], f2_a1)
    # update shared space should been seen by the other.
    f1[Index(th.tensor([2, 3]))] = {
            'a1' : th.ones([2, D]),
            'a2' : th.ones([2, D]),
            'a3' : th.ones([2, D]),
            }
    f2_a1[0:2] = th.ones([2, D])
    assert U.allclose(f2['a1'], f2_a1)

def test_slicing():
    data = Frame(create_test_data(grad=True))
    f1 = FrameRef(data, index=slice(1, 5))
    f2 = FrameRef(data, index=slice(3, 8))
    # test read
    for k, v in f1.items():
        assert U.allclose(data[k].data[1:5], v)
    f2_a1 = f2['a1'].data
    # test write
    f1[Index(th.tensor([0, 1]))] = {
            'a1': th.zeros([2, D]),
            'a2': th.zeros([2, D]),
            'a3': th.zeros([2, D]),
            }
    assert U.allclose(f2['a1'], f2_a1)
    
    f1[Index(th.tensor([2, 3]))] = {
            'a1': th.ones([2, D]),
            'a2': th.ones([2, D]),
            'a3': th.ones([2, D]),
            }
    f2_a1[0:2] = 1
    assert U.allclose(f2['a1'], f2_a1)

    f1[2:4] = {
            'a1': th.zeros([2, D]),
            'a2': th.zeros([2, D]),
            'a3': th.zeros([2, D]),
            }
    f2_a1[0:2] = 0
    assert U.allclose(f2['a1'], f2_a1)

def test_add_rows():
    data = Frame()
    f1 = FrameRef(data)
    f1.add_rows(4)
    x = th.randn(1, 4)
    f1[Index(th.tensor([0]))] = {'x': x}
    ans = th.cat([x, th.zeros(3, 4)])
    assert U.allclose(f1['x'], ans)
    f1.add_rows(4)
    f1[4:8] = {'x': th.ones(4, 4), 'y': th.ones(4, 5)}
    ans = th.cat([ans, th.ones(4, 4)])
    assert U.allclose(f1['x'], ans)
    ans = th.cat([th.zeros(4, 5), th.ones(4, 5)])
    assert U.allclose(f1['y'], ans)

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
    test_slicing()
    test_add_rows()
