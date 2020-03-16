"""DataLoader utils."""
import dgl
from mxnet import nd
from gluoncv.data.batchify import Pad

def dgl_mp_batchify_fn(data):
    if isinstance(data[0], tuple):
        data = zip(*data)
        return [dgl_mp_batchify_fn(i) for i in data]
    
    for dt in data:
        if dt is not None:
            if isinstance(dt, dgl.DGLGraph):
                return [d for d in data if isinstance(d, dgl.DGLGraph)]
            elif isinstance(dt, nd.NDArray):
                pad = Pad(axis=(1, 2), num_shards=1, ret_length=False)
                data_list = [dt for dt in data if dt is not None]
                return pad(data_list)
