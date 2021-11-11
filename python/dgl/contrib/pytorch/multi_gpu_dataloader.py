##
#   Copyright (c) 2021, NVIDIA CORPORATION.
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
from dgl import utils
from dgl.frame import Frame
import torch as th
from ...dataloading import NodeDataLoader
from ...partition import NDArrayPartition, create_edge_partition_from_nodes
from ...cuda import nccl
from ..multi_gpu_datastore import MultiGPUDataStore
from typing import Mapping

def _load_tensor(tensor, device, comm, part):
    loaded_tensor = MultiGPUDataStore(shape=tensor.shape, dtype=tensor.dtype,
        device=device, comm=comm, partition=part)
    loaded_tensor.all_set_global(tensor)
    return loaded_tensor

def _gather_row(tensor, index):
    return tensor.all_gather_row(index)

class _NodeDataIterator:
    def __init__(self, it, n_feat, e_feat, node_feat, node_label):
        self._it = it
        # {ntype: {feature: tensor}}
        self._n_feat = n_feat
        # {etype: {feature: tensor}}
        self._e_feat = e_feat
        self._node_feat = node_feat
        self._node_label = node_label

    # Make this an iterator for PyTorch Lightning compatibility
    def __iter__(self):
        return self

    def __next__(self):
        input_nodes, output_nodes, blocks = next(self._it)

        # re-attach node features
        for block in blocks:
            node_frames = []
            # have to handle src and dst nodes separately in a block
            for ntype in block.srctypes:
                index = block.srcnodes[ntype].data[dgl.NID]
                frame = Frame(num_rows=len(index))
                for k, v in self._n_feat[ntype].items():
                    data = _gather_row(v, index)
                    frame.update_column(k, data)
                node_frames.append(frame)
            for ntype in block.dsttypes:
                index = block.dstnodes[ntype].data[dgl.NID]
                frame = Frame(num_rows=len(index))
                for k, v in self._n_feat[ntype].items():
                    data = _gather_row(v, index)
                    frame.update_column(k, data)
                node_frames.append(frame)

            edge_frames = []
            for etype in block.canonical_etypes:
                index = block.edges[etype].data[dgl.EID]
                if isinstance(index, Mapping):
                    index = index[block.to_canonical_etype(etype)]
                frame = Frame(num_rows=len(index))
                for k, v in self._e_feat[etype].items():
                    data = _gather_row(v, index)
                    frame.update_column(k, data)
                edge_frames.append(frame)

            utils.set_new_frames(block, node_frames=node_frames,
                edge_frames=edge_frames)

        result = [input_nodes, output_nodes, blocks]
        if self._node_feat is not None:
            if isinstance(self._node_feat, Mapping):
                input_feats = {}
                for k,v in self._node_feat.items():
                    input_feats[k] = _gather_row(v, input_nodes[k])
            else:
                input_feats = _gather_row(self._node_feat, input_nodes)
            result.append(input_feats)

        if self._node_label is not None:
            output_labels = _gather_row(self._node_label, output_nodes)
            result.append(output_labels)

        return result

class MultiGPUNodeDataLoader(NodeDataLoader):
    """Specialized pytorch dataloader for iterating over a set of nodes,
    generating the message flow graphs (MFGs) and collecting feature and label
    data for the minibatch.

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
    partition : dgl.partition.NDArrayPartition or dict[ntype,
            dgl.partition.NDArrayPartition], optional
        The partition specifying to which device each node's data belongs.
        If not specified, the indices will be striped evenly across the
        GPUs.
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
        input nodes of the mini-batch will be returned as the 4th item from the
        iterator.
    node_label : Tensor or dict[ntype, Tensor], optional
        The node labels to distribute for each
        mini-batch. If specified, the sliced tensor corresponding to the
        seeds of the mini-batch will be returned as the last item from the
        iterator.
    kwargs : dict
        Arguments being passed to :py:class:`dgl.dataloader.NodeDataLoader`.
        If `num_workers` is specified, it must be 0.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = dgl.contrib.MultiGPUNodeDataLoader(
    ...     g, train_nid, sampler,
    ...     device=dev_id, node_feat=nfeat,
    ...     node_label=labels, batch_size=1024, shuffle=True,
    ...     drop_last=False)
    >>> for input_nodes, output_nodes, blocks, batch_feats, batch_labels in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks, batch_feats, batch_labels)

    """
    def __init__(self, g, nids, block_sampler, device, partition=None, use_ddp=True,
                 node_feat=None, node_label=None, num_workers=0, **kwargs):
        assert device != th.device("cpu"), "The device must be a GPU."
        assert num_workers == 0, "MultiGPUNodeDataLoader only works with " \
            "0 workers."

        # create the nccl communicator (if we have multiple processes)
        comm = None
        if th.distributed.is_initialized():
            rank = th.distributed.get_rank()
            world_size = th.distributed.get_world_size()

            objs = [None]
            nccl_id = None
            if rank == 0:
                nccl_id = nccl.UniqueId()
                objs[0] = str(nccl_id)
            th.distributed.broadcast_object_list(objs)
            if rank != 0:
                nccl_id = nccl.UniqueId(objs[0])
            comm = nccl.Communicator(world_size, rank, nccl_id)
            assert use_ddp, "'use_ddp' must be true when using NCCL."
        else:
            comm = nccl.Communicator(1, 0, nccl.UniqueId())

        if partition is None:
            partition = {}
            if len(g.ntypes) > 1:
                for ntype in g.ntypes:
                    partition[ntype] = NDArrayPartition(
                        g.number_of_nodes(ntype),
                        comm.size(),
                        mode='remainder')
            else:
                # homogenous graph
                partition[g.ntypes[0]] = NDArrayPartition(
                    g.number_of_nodes(),
                    comm.size(),
                    mode='remainder')
        elif not isinstance(partition, Mapping):
            assert len(g.ntypes) == 1, "For multiple ntypes, `parition` must " \
                "be a mapping of ntypes to NDArrayPartitions."
            partition = {g.ntypes[0]: partition}

        edge_partition = create_edge_partition_from_nodes( \
            partition, g)

        # save all node features to GPU
        self._n_feat = {}
        for i, ntype in enumerate(g.ntypes):
            feats = {}
            for feat_name in list(g._node_frames[i].keys()):
                if isinstance(g.ndata[feat_name], Mapping):
                    data = g.ndata[feat_name][ntype]
                else:
                    data = g.ndata[feat_name]
                feats[feat_name] = _load_tensor(data, device, comm,
                                                partition[ntype])
            self._n_feat[ntype] = feats

        # remove all node features
        for i, ntype in enumerate(g.ntypes):
            for feat_name in list(g._node_frames[i].keys()):
                g._node_frames[i].pop(feat_name)

        # save all edge features to GPU
        self._e_feat = {}
        for i, etype in enumerate(g.canonical_etypes):
            feats = {}
            for feat_name in list(g._edge_frames[i].keys()):
                if isinstance(g.edata[feat_name], Mapping):
                    data = g.edata[feat_name][etype]
                else:
                    data = g.edata[feat_name]
                feats[feat_name] = _load_tensor(data, device, comm,
                                                edge_partition[etype])
            self._e_feat[etype] = feats

        # remove all edge features
        for i, etype in enumerate(g.etypes):
            for feat_name in list(g._edge_frames[i].keys()):
                g._edge_frames[i].pop(feat_name)

        super(MultiGPUNodeDataLoader, self).__init__(
            g=g, nids=nids, block_sampler=block_sampler, device=device,
            use_ddp=use_ddp, num_workers=num_workers, **kwargs)

        self._node_feat = None
        if node_feat is not None:
            if isinstance(node_feat, Mapping):
                self._node_feat = {}
                for k,v in node_feat.items():
                    self._node_feat[k] = _load_tensor(v, device, comm,
                                                      partition[k])
            else:
                assert len(g.ntypes) == 1, "For multiple ntypes, `node_feat` " \
                    "must be a mapping of ntypes to tensors."
                self._node_feat = _load_tensor(node_feat, device, comm,
                    partition[g.ntypes[0]])

        self._node_label = None
        if node_label is not None:
            if isinstance(node_label, Mapping):
                self._node_label = {}
                for k,v in node_label.items():
                    self._node_label[k] = _load_tensor(v, device, comm,
                                                      partition[k])
            else:
                assert len(g.ntypes) == 1, "For multiple ntypes, `node_label` " \
                    "must be a mapping of ntypes to tensors."
                self._node_label = _load_tensor(node_label, device, comm,
                    partition[g.ntypes[0]])

    def __iter__(self):
        """Return the iterator of the data loader."""
        it = super(MultiGPUNodeDataLoader, self).__iter__()
        return _NodeDataIterator(it, self._n_feat, self._e_feat,
                                 self._node_feat, self._node_label)

