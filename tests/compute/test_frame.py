import numpy as np
from dgl.frame import Frame, FrameRef
from dgl.utils import Index, toindex
import backend as F

N = 10
D = 5

def check_fail(fn):
    try:
        fn()
        return False
    except:
        return True

def create_test_data(grad=False):
    c1 = F.randn((N, D))
    c2 = F.randn((N, D))
    c3 = F.randn((N, D))
    if grad:
        c1 = F.attach_grad(c1)
        c2 = F.attach_grad(c2)
        c3 = F.attach_grad(c3)
    return {'a1' : c1, 'a2' : c2, 'a3' : c3}

def test_create():
    data = create_test_data()
    f1 = Frame(num_rows=N)
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
    assert F.allclose(f['a1'].data, data['a1'])
    f['a1'] = data['a2']
    assert F.allclose(f['a2'].data, data['a2'])
    # add a different length column should fail
    def failed_add_col():
        f['a4'] = F.zeros([N+1, D])
    assert check_fail(failed_add_col)
    # delete all the columns
    del f['a1']
    del f['a2']
    assert len(f) == 1
    del f['a3']
    assert len(f) == 0

def test_column2():
    # Test frameref column getter/setter
    data = Frame(create_test_data())
    f = FrameRef(data, toindex([3, 4, 5, 6, 7]))
    assert f.num_rows == 5
    assert len(f) == 3
    assert F.allclose(f['a1'], F.narrow_row(data['a1'].data, 3, 8))
    # set column should reflect on the referenced data
    f['a1'] = F.zeros([5, D])
    assert F.allclose(F.narrow_row(data['a1'].data, 3, 8), F.zeros([5, D]))
    # add new partial column should fail with error initializer
    f.set_initializer(lambda shape, dtype : assert_(False))
    def failed_add_col():
        f['a4'] = F.ones([5, D])
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
    assert tuple(F.shape(c1.data)) == (2 * N, D)
    truth = F.cat([data['a1'], data['a1']], 0)
    assert F.allclose(truth, c1.data)
    # append dict of different length columns should fail
    f3 = {'a1' : F.zeros((3, D)), 'a2' : F.zeros((3, D)), 'a3' : F.zeros((2, D))}
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
    assert F.array_equal(f._index.tousertensor(), F.tensor(new_idx, dtype=F.int64))
    assert data.num_rows == 4 * N

def test_append3():
    # test append on empty frame
    f = Frame(num_rows=5)
    data = {'h' : F.ones((3, 2))}
    f.append(data)
    assert f.num_rows == 8
    ans = F.cat([F.zeros((5, 2)), F.ones((3, 2))], 0)
    assert F.allclose(f['h'].data, ans)
    # test append with new column
    data = {'h' : 2 * F.ones((3, 2)), 'w' : 2 * F.ones((3, 2))}
    f.append(data)
    assert f.num_rows == 11
    ans1 = F.cat([ans, 2 * F.ones((3, 2))], 0)
    ans2 = F.cat([F.zeros((8, 2)), 2 * F.ones((3, 2))], 0)
    assert F.allclose(f['h'].data, ans1)
    assert F.allclose(f['w'].data, ans2)

def test_row1():
    # test row getter/setter
    data = create_test_data()
    f = FrameRef(Frame(data))

    # getter
    # test non-duplicate keys
    rowid = Index(F.tensor([0, 2]))
    rows = f[rowid]
    for k, v in rows.items():
        assert tuple(F.shape(v)) == (len(rowid), D)
        assert F.allclose(v, F.gather_row(data[k], F.tensor(rowid.tousertensor())))
    # test duplicate keys
    rowid = Index(F.tensor([8, 2, 2, 1]))
    rows = f[rowid]
    for k, v in rows.items():
        assert tuple(F.shape(v)) == (len(rowid), D)
        assert F.allclose(v, F.gather_row(data[k], F.tensor(rowid.tousertensor())))

    # setter
    rowid = Index(F.tensor([0, 2, 4]))
    vals = {'a1' : F.zeros((len(rowid), D)),
            'a2' : F.zeros((len(rowid), D)),
            'a3' : F.zeros((len(rowid), D)),
            }
    f[rowid] = vals
    for k, v in f[rowid].items():
        assert F.allclose(v, F.zeros((len(rowid), D)))

    # setting rows with new column should raise error with error initializer
    f.set_initializer(lambda shape, dtype : assert_(False))
    def failed_update_rows():
        vals['a4'] = F.ones((len(rowid), D))
        f[rowid] = vals
    assert check_fail(failed_update_rows)

def test_row2():
    # test row getter/setter autograd compatibility
    data = create_test_data(grad=True)
    f = FrameRef(Frame(data))

    with F.record_grad():
        # getter
        c1 = f['a1']
        # test non-duplicate keys
        rowid = Index(F.tensor([0, 2]))
        rows = f[rowid]
        y = rows['a1']
    F.backward(y, F.ones((len(rowid), D)))
    assert F.allclose(F.grad(c1)[:,0], F.tensor([1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]))

    f['a1'] = F.attach_grad(f['a1'])
    with F.record_grad():
        c1 = f['a1']
        # test duplicate keys
        rowid = Index(F.tensor([8, 2, 2, 1]))
        rows = f[rowid]
        y = rows['a1']
    F.backward(y, F.ones((len(rowid), D)))
    assert F.allclose(F.grad(c1)[:,0], F.tensor([0., 1., 2., 0., 0., 0., 0., 0., 1., 0.]))

    f['a1'] = F.attach_grad(f['a1'])
    with F.record_grad():
        # setter
        c1 = f['a1']
        rowid = Index(F.tensor([0, 2, 4]))
        vals = {'a1' : F.attach_grad(F.zeros((len(rowid), D))),
                'a2' : F.attach_grad(F.zeros((len(rowid), D))),
                'a3' : F.attach_grad(F.zeros((len(rowid), D))),
                }
        f[rowid] = vals
        c11 = f['a1']
    F.backward(c11, F.ones((N, D)))
    assert F.allclose(F.grad(c1)[:,0], F.tensor([0., 1., 0., 1., 0., 1., 1., 1., 1., 1.]))
    assert F.allclose(F.grad(vals['a1']), F.ones((len(rowid), D)))
    assert F.is_no_grad(vals['a2'])

def test_row3():
    # test row delete
    data = Frame(create_test_data())
    f = FrameRef(data)
    assert f.is_contiguous()
    assert f.is_span_whole_column()
    assert f.num_rows == N
    del f[toindex(F.tensor([2, 3]))]
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
        assert F.allclose(v, data[k][newidx])

def test_row4():
    # test updating row with empty frame but has preset num_rows
    f = FrameRef(Frame(num_rows=5))
    rowid = Index(F.tensor([0, 2, 4]))
    f[rowid] = {'h' : F.ones((3, 2))}
    ans = F.zeros((5, 2))
    ans[F.tensor([0, 2, 4])] = F.ones((3, 2))
    assert F.allclose(f['h'], ans)

def test_sharing():
    data = Frame(create_test_data())
    f1 = FrameRef(data, index=toindex([0, 1, 2, 3]))
    f2 = FrameRef(data, index=toindex([2, 3, 4, 5, 6]))
    # test read
    for k, v in f1.items():
        assert F.allclose(F.narrow_row(data[k].data, 0, 4), v)
    for k, v in f2.items():
        assert F.allclose(F.narrow_row(data[k].data, 2, 7), v)
    f2_a1 = f2['a1']
    # test write
    # update own ref should not been seen by the other.
    f1[Index(F.tensor([0, 1]))] = {
            'a1' : F.zeros([2, D]),
            'a2' : F.zeros([2, D]),
            'a3' : F.zeros([2, D]),
            }
    assert F.allclose(f2['a1'], f2_a1)
    # update shared space should been seen by the other.
    f1[Index(F.tensor([2, 3]))] = {
            'a1' : F.ones([2, D]),
            'a2' : F.ones([2, D]),
            'a3' : F.ones([2, D]),
            }
    F.narrow_row_set(f2_a1, 0, 2, F.ones([2, D]))
    assert F.allclose(f2['a1'], f2_a1)

def test_slicing():
    data = Frame(create_test_data(grad=True))
    f1 = FrameRef(data, index=toindex(slice(1, 5)))
    f2 = FrameRef(data, index=toindex(slice(3, 8)))
    # test read
    for k, v in f1.items():
        assert F.allclose(F.narrow_row(data[k].data, 1, 5), v)
    f2_a1 = f2['a1']    # is a tensor
    # test write
    f1[Index(F.tensor([0, 1]))] = {
            'a1': F.zeros([2, D]),
            'a2': F.zeros([2, D]),
            'a3': F.zeros([2, D]),
            }
    assert F.allclose(f2['a1'], f2_a1)
    
    f1[Index(F.tensor([2, 3]))] = {
            'a1': F.ones([2, D]),
            'a2': F.ones([2, D]),
            'a3': F.ones([2, D]),
            }
    F.narrow_row_set(f2_a1, 0, 2, 1)
    assert F.allclose(f2['a1'], f2_a1)

    f1[toindex(slice(2, 4))] = {
            'a1': F.zeros([2, D]),
            'a2': F.zeros([2, D]),
            'a3': F.zeros([2, D]),
            }
    F.narrow_row_set(f2_a1, 0, 2, 0)
    assert F.allclose(f2['a1'], f2_a1)

def test_add_rows():
    data = Frame()
    f1 = FrameRef(data)
    f1.add_rows(4)
    x = F.randn((1, 4))
    f1[Index(F.tensor([0]))] = {'x': x}
    ans = F.cat([x, F.zeros((3, 4))], 0)
    assert F.allclose(f1['x'], ans)
    f1.add_rows(4)
    f1[toindex(slice(4, 8))] = {'x': F.ones((4, 4)), 'y': F.ones((4, 5))}
    ans = F.cat([ans, F.ones((4, 4))], 0)
    assert F.allclose(f1['x'], ans)
    ans = F.cat([F.zeros((4, 5)), F.ones((4, 5))], 0)
    assert F.allclose(f1['y'], ans)

def test_inplace():
    f = FrameRef(Frame(create_test_data()))
    print(f.schemes)
    a1addr = id(f['a1'])
    a2addr = id(f['a2'])
    a3addr = id(f['a3'])

    # column updates are always out-of-place
    f['a1'] = F.ones((N, D))
    newa1addr = id(f['a1'])
    assert a1addr != newa1addr
    a1addr = newa1addr
    # full row update that becomes column update
    f[toindex(slice(0, N))] = {'a1' : F.ones((N, D))}
    assert id(f['a1']) != a1addr

    # row update (outplace) w/ slice
    f[toindex(slice(1, 4))] = {'a2' : F.ones((3, D))}
    newa2addr = id(f['a2'])
    assert a2addr != newa2addr
    a2addr = newa2addr
    # row update (outplace) w/ list
    f[toindex([1, 3, 5])] = {'a2' : F.ones((3, D))}
    newa2addr = id(f['a2'])
    assert a2addr != newa2addr
    a2addr = newa2addr

    # row update (inplace) w/ slice
    f.update_data(toindex(slice(1, 4)), {'a2' : F.ones((3, D))}, True)
    newa2addr = id(f['a2'])
    assert a2addr == newa2addr
    # row update (inplace) w/ list
    f.update_data(toindex([1, 3, 5]), {'a2' : F.ones((3, D))}, True)
    newa2addr = id(f['a2'])
    assert a2addr == newa2addr

if __name__ == '__main__':
    test_create()
    test_column1()
    test_column2()
    test_append1()
    test_append2()
    test_append3()
    test_row1()
    test_row2()
    test_row3()
    test_row4()
    test_sharing()
    test_slicing()
    test_add_rows()
    test_inplace()
