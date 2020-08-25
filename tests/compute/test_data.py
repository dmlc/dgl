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
    col_norm_feat = data.utils.float_col_l1_normalize(features)
    assert np.allclose(np.array([[1./6.],[2./6.], [-3./6.]]), col_norm_feat)

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

def test_float_col_maxmin_normalize():
    features = np.array([[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 0., 0.]]), col_norm_feat)

    features = np.array([[1.], [2.], [-3.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[4./5.],[5./5.], [0.]]), col_norm_feat)

    features = np.array([[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 0., 3./4.],[1., 0.5, 1.],[0., 1., 0.]]),
                       col_norm_feat)

    # input (2. 3)
    features = np.array([[1., 0., 0.],[2., 1., -1.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 0., 1.],[1., 1., 0.]]),
                       col_norm_feat)

    # input (3. 2)
    features = np.array([[1., 0.],[2., 1.],[4., -2.]])
    col_norm_feat = data.utils.float_col_maxmin_normalize(features)
    assert np.allclose(np.array([[0., 2./3.],[1./3., 1.],[1., 0.]]),
                       col_norm_feat)

@unittest.skip("spacy language test is too heavy")
def test_embed_word2vec():
    import spacy

    inputs = ['hello', 'world']
    languages = ['en_core_web_lg', 'fr_core_news_lg']
    nlps = [spacy.load(languages[0])]

    feats = data.utils.embed_word2vec(inputs[0], nlps)
    doc = nlps[0](inputs[0])
    assert np.allclose(doc.vector, feats)

    nlps.append(spacy.load(languages[1]))
    for input in inputs:
        feats = data.utils.embed_word2vec(input, nlps)
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        assert np.allclose(np.concatenate((doc0.vector, doc1.vector)),
                           feats)

@unittest.skip("spacy language test is too heavy")
def test_parse_lang_feat():
    import spacy

    inputs = ['hello', 'world']
    languages = ['en_core_web_lg', 'fr_core_news_lg']
    nlps = [spacy.load(languages[0]), spacy.load(languages[1])]
    feats = data.utils.parse_lang_feat(inputs, nlps)

    res_feats = []
    for input in inputs:
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        res_feats.append(np.concatenate((doc0.vector, doc1.vector)))
    res_feats = np.stack(res_feats)
    assert np.allclose(feats, res_feats)

    inputs = ["1", "2", "3", "4", "1", "2", "3", "4", "5", "6", "7", "8"]
    feats = data.utils.parse_lang_feat(inputs, nlps)

    res_feats = []
    for input in inputs:
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        res_feats.append(np.concatenate((doc0.vector, doc1.vector)))
    res_feats = np.stack(res_feats)
    assert np.allclose(feats, res_feats)

    inputs = ["1", "2", "3", "4", "1", "2", "3", "4", "5", "6", "7", "8"]
    feats = data.utils.parse_word2vec_feature(inputs, languages)

    res_feats = []
    for input in inputs:
        doc0 = nlps[0](input)
        doc1 = nlps[1](input)
        res_feats.append(np.concatenate((doc0.vector, doc1.vector)))
    res_feats = np.stack(res_feats)
    assert np.allclose(feats, res_feats)

@unittest.skip("LabelBinarizer and MultiLabelBinarizer is not included in CI env")
def test_parse_category_feat():
    # single-hot
    inputs = ['A', 'B']
    feats = data.utils.parse_category_single_feat(inputs)
    assert np.allclose(np.array([[1.,0.],[0.,1.]]), feats)

    inputs = ['A', 'B', 'C', 'A']
    feats = data.utils.parse_category_single_feat(inputs)
    assert np.allclose(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]), feats)
    # col norm
    feats = data.utils.parse_category_single_feat(inputs, norm='col')
    assert np.allclose(np.array([[.5,0.,0.],[0.,1.,0.],[0.,0.,1.],[.5,0.,0.]]), feats)

    # multi-hot
    inputs = [['A'], ['B']]
    feats = data.utils.parse_category_multi_feat(inputs)
    assert np.allclose(np.array([[1.,0.],[0.,1.]]), feats)

    inputs = [['A', 'B', 'C',], ['A', 'B'], ['C'], ['A']]
    feats = data.utils.parse_category_multi_feat(inputs)
    assert np.allclose(np.array([[1.,1.,1.],[1.,1.,0.],[0.,0.,1.],[1.,0.,0.]]), feats)
    # row norm
    feats = data.utils.parse_category_multi_feat(inputs, norm='row')
    assert np.allclose(np.array([[1./3.,1./3.,1./3.],[.5,.5,0.],[0.,0.,1.],[1.,0.,0.]]), feats)
    # col norm
    feats = data.utils.parse_category_multi_feat(inputs, norm='col')
    assert np.allclose(np.array([[1./3.,0.5,0.5],[1./3.,0.5,0.],[0.,0.,0.5],[1./3.,0.,0.]]), feats)

def test_parse_numerical_feat():
    inputs = [[1., 2., -3.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[1., 1., -1.]]), col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 0., 0.]]), col_norm_feat)

    inputs = [[1.], [2.], [-3.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[1./6.],[2./6.], [-3./6.]]), col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[4./5.],[5./5.], [0.]]), col_norm_feat)

    inputs = [[1., 0., 0.],[2., 1., 1.],[1., 2., -3.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[0.25, 0., 0.],[0.5, 1./3., 0.25],[0.25, 2./3., -0.75]]),
                       col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 0., 3./4.],[1., 0.5, 1.],[0., 1., 0.]]),
                       col_norm_feat)

    # input (2. 3)
    inputs = [[1., 0., 0.],[2., 1., -1.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[1./3., 0., 0.],[2./3., 1.0, -1.]]),
                       col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 0., 1.],[1., 1., 0.]]),
                       col_norm_feat)

    # input (3. 2)
    inputs = [[1., 0.],[2., 1.],[1., -2.]]
    feat = data.utils.parse_numerical_feat(inputs)
    assert np.allclose(inputs, feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='standard')
    assert np.allclose(np.array([[0.25, 0.],[0.5, 1./3.],[0.25, -2./3.]]),
                       col_norm_feat)
    col_norm_feat = data.utils.parse_numerical_feat(inputs, norm='min-max')
    assert np.allclose(np.array([[0., 2./3.],[1., 1.],[0., 0.]]),
                       col_norm_feat)

def test_parse_numerical_multihot_feat():
    inputs = [0., 15., 20., 10.1, 25., 40.]
    low = 10.
    high = 30.
    bucket_cnt = 2 #10~20, 20~30
    window_size = 0.
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size)
    assert np.allclose(np.array([[1., 0.], [1., 0.], [0., 1.], [1., 0.], [0., 1.], [0., 1.]]), feat)

    inputs = [0., 5., 15., 20., 10.1, 25., 30.1, 40.]
    low = 10.
    high = 30.
    bucket_cnt = 4 #10~15,15~20,20~25,25~30
    window_size = 10.
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size)
    assert np.allclose(np.array([[1., 0., 0., 0],
                                 [1., 0., 0., 0],
                                 [1., 1., 1., 0.],
                                 [0., 1., 1., 1.],
                                 [1., 1., 0., 0.],
                                 [0., 0., 1., 1.],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., 1.]]), feat)

    # col norm
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size, norm='col')
    assert np.allclose(np.array([[1./4., 0.,    0.,    0],
                                 [1./4., 0.,    0.,    0],
                                 [1./4., 1./3., 1./3., 0.],
                                 [0.,    1./3., 1./3., 1./4.],
                                 [1./4., 1./3., 0.,    0.],
                                 [0.,    0.,    1./3., 1./4.],
                                 [0.,    0.,    0.,    1./4.],
                                 [0.,    0.,    0.,    1./4.]]), feat)

    # row norm
    feat = data.utils.parse_numerical_multihot_feat(inputs, low, high, bucket_cnt, window_size, norm='row')
    assert np.allclose(np.array([[1., 0., 0., 0],
                                 [1., 0., 0., 0],
                                 [1./3., 1./3., 1./3., 0.],
                                 [0., 1./3., 1./3., 1./3.],
                                 [1./2., 1./2., 0., 0.],
                                 [0., 0., 1./2., 1./2.],
                                 [0., 0., 0., 1.],
                                 [0., 0., 0., 1.]]), feat)

if __name__ == '__main__':
    #test_minigc()
    #test_data_hash()

    test_row_normalize()
    test_col_normalize()
    test_float_row_normalize()
    test_float_col_normalize()
    test_float_col_maxmin_normalize()
    #test_embed_word2vec()

    #test_parse_lang_feat()
    #test_parse_category_feat()
    test_parse_numerical_feat()
    test_parse_numerical_multihot_feat()