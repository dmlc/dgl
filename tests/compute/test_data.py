import dgl.data as data
import unittest, pytest
import numpy as np

def test_minigc():
    ds = data.MiniGCDataset(16, 10, 20)
    g, l = list(zip(*ds))
    print(g, l)

def test_data_hash():
    class HashTestDataset(data.DGLDataset):
        def __init__(self, hash_key=()):
            super(HashTestDataset, self).__init__('hashtest', hash_key=hash_key)
        def _load(self):
            pass

    a = HashTestDataset((True, 0, '1', (1,2,3)))
    b = HashTestDataset((True, 0, '1', (1,2,3)))
    c = HashTestDataset((True, 0, '1', (1,2,4)))
    assert a.hash == b.hash
    assert a.hash != c.hash

def test_row_normalize():
    features = np.array([[1., 1., 1.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([1./3., 1./3., 1./3.]), row_norm_feat)

    features = np.array([[1.], [1.], [1.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1.], [1.], [1.]]), row_norm_feat)

    features = np.array([[1., 0., 0.],[0., 1., 1.],[0., 0., 0.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[0., 0.5, 0.5],[0., 0., 0.]]),
                       row_norm_feat)

    # input (2, 3)
    features = np.array([[1., 0., 0.],[2., 1., 1.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[0.5, 0.25, 0.25]]),
                       row_norm_feat)

    # input (3, 2)
    features = np.array([[1., 0.],[1., 1.],[0., 0.]])
    row_norm_feat = data.utils.row_normalize(features)
    assert np.allclose(np.array([[1., 0.],[0.5, 0.5],[0., 0.]]),
                       row_norm_feat)

def test_col_normalize():
    features = np.array([[1., 1., 1.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[1., 1., 1.]]), col_norm_feat)

    features = np.array([[1.], [1.], [1.]])
    row_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[1./3.],[1./3.], [1./3.]]), row_norm_feat)

    features = np.array([[1., 0., 0.],[1., 1., 0.],[0., 0., 0.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[0.5, 0., 0.],[0.5, 1.0, 0.],[0., 0., 0.]]),
                       col_norm_feat)

    # input (2. 3)
    features = np.array([[1., 0., 0.],[1., 1., 0.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[0.5, 0., 0.],[0.5, 1.0, 0.]]),
                       col_norm_feat)

    # input (3. 2)
    features = np.array([[1., 0.],[1., 1.],[2., 0.]])
    col_norm_feat = data.utils.col_normalize(features)
    assert np.allclose(np.array([[0.25, 0.],[0.25, 1.0],[0.5, 0.]]),
                       col_norm_feat)

def test_float_row_normalize():
    features = np.array([[1.],[2.],[-3.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1.],[1.],[-1.]]), row_norm_feat)

    features = np.array([[1., 2., -3.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1./6., 2./6., -3./6.]]), row_norm_feat)

    features = np.array([[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[0.5, 0.25, 0.25],[1./6., 2./6., -3./6.]]),
                       row_norm_feat)

     # input (2 3)
    features = np.array([[1., 0., 0.],[-2., 1., 1.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1., 0., 0.],[-0.5, 0.25, 0.25]]),
                       row_norm_feat)

     # input (3, 2)
    features = np.array([[1., 0.],[-2., 1.],[1., 2.]])
    row_norm_feat = data.utils.float_row_l1_normalize(features)
    assert np.allclose(np.array([[1., 0.],[-2./3., 1./3.],[1./3., 2./3.]]),
                       row_norm_feat)

def test_float_col_normalize():
    features = np.array([[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[1., 1., -1.]]), col_norm_feat)

    features = np.array([[1.], [2.], [-3.]])
    row_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[1./6.],[2./6.], [-3./6.]]), row_norm_feat)

    features = np.array([[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[0.25, 0., 0.],[0.5, 1./3., 0.25],[0.25, 2./3., -0.75]]),
                       col_norm_feat)

    # input (2. 3)
    features = np.array([[1., 0., 0.],[2., 1., -1.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[1./3., 0., 0.],[2./3., 1.0, -1.]]),
                       col_norm_feat)

    # input (3. 2)
    features = np.array([[1., 0.],[2., 1.],[1., -2.]])
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[0.25, 0.],[0.5, 1./3.],[0.25, -2./3.]]),
                       col_norm_feat)

if __name__ == '__main__':
    #test_minigc()
    #test_data_hash()

    test_row_normalize()
    test_col_normalize()
    test_float_row_normalize()
    test_float_col_normalize()