from typing import List, Tuple
from dgl.heterograph import DGLHeteroGraph
from torch._C import TupleType
import dgl
from dgl.data import CoraGraphDataset
import dgl.function as fn
import torch
g = CoraGraphDataset()[0]
torch.classes.load_library("/home/ubuntu/dev/torchdgl/build/libTorchDGLGraph.so")
TorchDGLGraph = torch.classes.my_classes.TorchDGLGraph
TorchDGLMetaGraph = torch.classes.my_classes.TorchDGLMetaGraph
from dgl.sparse import _gspmm

# def update_all(g: TorchDGLGraph, feat: torch.Tensor):
#     h2 = _gspmm(g, 'copy_lhs', "sum", feat, None)
#     return h2

# update_all(g._graph, g.ndata['feat'])
# import torch
# ss = torch.jit.script(update_all)
try:
    from typing_extensions import Final
except:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final
USE_TORCH=True

def t1(t: torch.Tensor):
    return torch.sum(t)

def poly_t1(t: torch.Tensor):
    print(t)
    return torch.tensor([1])

def poly_jit(b, t):
    # b: bool = USE_TORCH
    if b:
        return t1(t)
    else:
        return poly_t1(t)

func = torch.jit.script(poly_jit)