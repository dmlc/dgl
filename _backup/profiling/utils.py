import torch as th


def sparse_sp2th(matrix):
    coo = matrix.tocoo()
    rows = th.from_numpy(coo.row).long().view(1, -1)
    cols = th.from_numpy(coo.col).long().view(1, -1)
    data = th.from_numpy(coo.data).float()
    return th.sparse.FloatTensor(th.cat((rows, cols), 0), data, coo.shape)
