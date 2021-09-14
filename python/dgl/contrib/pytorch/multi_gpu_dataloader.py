##
#   Copyright 2021 Contributors 
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import dgl
import torch as th
from ...dataloading import NodeDataLoader
from ..multi_gpu_datastore import MultiGPUDataStore
from typing import Mapping

def _load_tensor(tensor, device, comm, part):
    loaded_tensor = None
    if comm:
        loaded_tensor = MultiGPUDataStore(shape=tensor.shape, dtype=tensor.dtype,
            device=device, comm=comm, partition=part)
        loaded_tensor.all_set_global(tensor)
    else:
        loaded_tensor = tensor.to(device)
    return loaded_tensor

def _gather_row(tensor, index):
    if isinstance(tensor, MultiGPUDataStore):
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
            for k, v in self._n_feat.items():
                for ntype, data in v.items():
                    index = block.ndata[dgl.NID][ntype]
                    block.ndata[k][ntype] = _gather_row(data, index)

        result = [input_nodes, output_nodes, blocks]
        if self._node_feat is not None:
            input_feats = _gather_row(self._node_feat, input_nodes)
            result.append(input_feats)

        if self._node_label is not None:
            output_labels = _gather_row(self._node_label, output_nodes)
            result.append(output_labels)

        return result
            
class MultiGPUNodeDataLoader(NodeDataLoader):
    """PyTorch dataloader for batch-iterating over a set of nodes, generating the list
    of message flow graphs (MFGs) as computation dependency of the said minibatch.
    The feature data of the graph is stored partitioned in GPU memory.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    block_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.
    device : device context
        The device to store the node data on, and the device
        of the generated MFGs in each iteration, which should be a
        PyTorch device object (e.g., ``torch.device``) specifying a GPU.
    comm : dgl.cuda.nccl.Communicator, optional
        The communicator to use to exchange features for each mini-batch.
        Must not be None when multiple GPUs are used.
    partition : dgl.partition.NDArrayPartition, optional
        The partition specifying to which device each node's data belongs.
        Must not be None when multiple GPUs are used.
    use_ddp : boolean, optional
        If True, tells the DataLoader to split the training set for each
        participating process appropriately using
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Note that :func:`~dgl.dataloading.NodeDataLoader.set_epoch` must be called
        at the beginning of every epoch if :attr:`use_ddp` is True.

        Overrides the :attr:`sampler` argument of :class:`torch.utils.data.DataLoader`.
    node_feat : Tensor or dict[ntype, Tensor], optional
        The node features to distribute separate from the graph for each
        mini-batch. If specified, the sliced tensor corresponding to the
        input nodes of the minibatch will be returned as the 4th item from the
        iterator.
    node_label : Tensor or dict[ntype, Tensor], optional
        The node labels to distribute for each
        mini-batch. If specified, the sliced tensor corresponding to the
        seeds of the minibatch will be returned as the last item from the
        iterator.
    kwargs : dict
        Arguments being passed to :py:class:`dgl.dataloader.NodeDataLoader`.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.contrib.MultiGPUNodeDataLoader(
    ...     g, train_nid, sampler,
    ...     device=dev_id, comm=nccl_comm, node_feat=nfeat,
    ...     node_label=labels, batch_size=1024, shuffle=True,
    ...     drop_last=False)
    >>> for input_nodes, output_nodes, blocks, batch_feats, batch_labels in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks, batch_feats, batch_labels)

    In this example, `nccl_comm` is a :py:class:`dgl.cuda.nccl.Communicator`
    that has previously been setup, and `dev_id` is the GPU being used by the
    local process.
    """
    def __init__(self, g, nids, block_sampler, device, comm=None, partition=None, use_ddp=True,
                 node_feat=None, node_label=None, **kwargs):
        assert comm is None or use_ddp, "'use_ddp' must be true when using NCCL."
        assert device != th.device("cpu"), "The device must be a GPU."

        # we need to remove all of the features
        n_feat = {}
        for feat_name in list(g.ndata.keys()):
            if isinstance(g.ndata[feat_name], Mapping):
                n_feat[feat_name] = {k: g.ndata[feat_name][k] for k in g.ntypes}
            else:
                # make homogenous graphs still be mapped by node type, in order
                # to simplify the rest of the code using this structure
                n_feat[feat_name] = {'_N': g.ndata[feat_name]}
            g.ndata.pop(feat_name)

        super(MultiGPUNodeDataLoader, self).__init__(
            g=g, nids=nids, block_sampler=block_sampler, device=device,
            use_ddp=use_ddp, **kwargs)

        # move features to GPU 
        self._n_feat = {}
        for k, v in n_feat.items():
            self._n_feat[k] = {}
            for ntype in v.keys():
                self._n_feat[k][ntype] = \
                    _load_tensor(v[ntype], device, comm, partition)

        self._node_feat = None
        if node_feat is not None:
            self._node_feat = _load_tensor(node_feat, device, comm, partition)
        
        self._node_label = None
        if node_label is not None:
            self._node_label = _load_tensor(node_label, device, comm, partition)

    def __iter__(self):
        """Return the iterator of the data loader."""
        it = super(MultiGPUNodeDataLoader, self).__iter__()
        return _NodeDataIterator(it, self._n_feat, self._node_feat,
                                 self._node_label)
            
