import dgl
from ...dataloading import NodeDataLoader
from ..multi_gpu_tensor import MultiGPUTensor

def _load_tensor(tensor, device, comm, part):
    loaded_tensor = None
    if comm:
        loaded_tensor = MultiGPUTensor(shape=tensor.shape, dtype=tensor.dtype,
            device=device, comm=comm, partition=part)
        loaded_tensor.all_set_global(tensor)
    else:
        loaded_tensor = tensor.to(device)
    return loaded_tensor

def _gather_row(tensor, index):
    if isinstance(tensor, MultiGPUTensor):
        return tensor.all_gather_row(index)
    else:
        return tensor[index]

class _NodeDataIterator:
    def __init__(self, it, n_feat, node_feat, node_label):
        self._it = it
        self._n_feat = n_feat
        self._node_feat = node_feat
        self._node_label = node_label

    # Make this an iterator for PyTorch Lightning compatibility
    def __iter__(self):
        return self

    def __next__(self):
        input_nodes, output_nodes, blocks = next(self._it)

        # re-attach node features
        for block in blocks:
            for ntype in block.ntypes:
                for k, v in self._n_feat.items():
                    block.ndata[k][ntype] = _gather_row(v, block.ndata[dgl.NID]['_N'])

        result = [input_nodes, output_nodes, blocks]
        if self._node_feat is not None:
            input_feats = _gather_row(self._node_feat, input_nodes)
            result.append(input_feats)

        if self._node_label is not None:
            output_labels = _gather_row(self._node_label, output_nodes)
            result.append(output_labels)

        return result
            
class MultiGPUNodeDataLoader(NodeDataLoader):
    def __init__(self, g, nids, block_sampler, device, comm, partition=None, use_ddp=True,
                 node_feat=None, node_label=None, **kwargs):
        assert comm is None or use_ddp, "'use_ddp' must be true when using NCCL."

        # we need to remove all of the features
        n_feat = {k: g.ndata.pop(k) for k in list(g.ndata.keys())}

        super(MultiGPUNodeDataLoader, self).__init__(
            g=g, nids=nids, block_sampler=block_sampler, device=device,
            use_ddp=use_ddp, **kwargs)

        # move features to GPU 
        self._n_feat = {}
        for k, v in n_feat.items():
            self._n_feat[k] = _load_tensor(v, device, comm, partition)

        self._node_feat = None
        if node_feat is not None:
            self._node_feat = _load_tensor(node_feat, device, comm, partition)
        
        self._node_label = None
        if node_label is not None:
            self._node_label = _load_tensor(node_label, device, comm, partition)

    def __iter__(self):
        it = super(MultiGPUNodeDataLoader, self).__iter__()
        return _NodeDataIterator(it, self._n_feat, self._node_feat,
                                 self._node_label)
            
