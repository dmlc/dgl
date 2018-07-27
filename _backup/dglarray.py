import dgl.backend as F

class DGLArray:
    def __init__(self):
        pass

    def __getitem__(self, x):
        raise NotImplementedError()

class DGLDenseArray(DGLArray):
    def __init__(self):
        pass

class DGLSparseArray(DGLArray):
    def __init__(self, data, ):
        raise NotImplementedError()
