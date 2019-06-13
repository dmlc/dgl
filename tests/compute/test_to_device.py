import dgl
import backend as F

def test_to_device():
    g = dgl.DGLGraph()
    g.add_nodes(5, {'h' : F.ones((5, 2))})
    g.add_edges([0, 1], [1, 2], {'m' : F.ones((2, 2))})
    if F.is_cuda_available():
        g.to(F.cuda())


if __name__ == '__main__':
    test_to_device()
