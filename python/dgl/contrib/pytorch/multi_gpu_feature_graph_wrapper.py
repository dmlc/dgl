##
#   Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
""" The MultiGPUDataStore class. """

import dgl
from dgl import utils
from dgl.frame import Frame
import torch as th
from ...dataloading import NodeDataLoader
from ...partition import NDArrayPartition, create_edge_partition_from_nodes
from ...cuda import nccl
from ...sampling import neighbor
from ...base import EID
from ..multi_gpu_feature_storage import MultiGPUFeatureStorage
from ... import backend as F
from typing import Mapping

def _load_tensor(tensor, device, comm, part):
    loaded_tensor = MultiGPUFeatureStorage(
        shape=tensor.shape, dtype=tensor.dtype,
        device=device, comm=comm, partition=part)
    loaded_tensor.all_set_global(tensor)
    return loaded_tensor

def _gather_row(tensor, index):
    return tensor.all_gather_row(index)

class MultiGPUFeatureGraphWrapper:
    """.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    device : device context
        The device to store the node data on, and the device
        of the generated MFGs in each iteration, which should be a
        PyTorch device object (e.g., ``torch.device``) specifying a GPU.
    partition : dgl.partition.NDArrayPartition or dict[ntype,
            dgl.partition.NDArrayPartition], optional
        The partition specifying to which device each node's data belongs.
        If not specified, the indices will be striped evenly across the
        GPUs.

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
    def __init__(self, g, device, partition=None, **kwargs):
        assert device != th.device("cpu"), "The device must be a GPU."

        self._device = device
        self._g = g

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

    def sample_neighbors(self, nodes, fanout, edge_dir='in', prob=None,
                         replace=False, output_device=None, exclude_edges=None):

        if output_device is None:
            output_device = self._device
        assert output_device == self._device, \
            "The output device of the sampler must be the same device as " \
            "the GPU feature store: {} vs. {}".format(
                output_device, self._device)
        
        frontier = neighbor.sample_neighbors(
            g=self._g, nodes=nodes, fanout=fanout, edge_dir=edge_dir,
            prob=prob, replace=replace, copy_ndata=False, copy_edata=False,
            exclude_edges=exclude_edges, output_device=output_device)

        # manually copy node features
        node_frames = []
        for ntype in frontier.ntypes:
            index = frontier.nodes(ntype).to(self._device)
            frame = Frame(num_rows=len(index))
            for k, v in self._n_feat[ntype].items():
                data = _gather_row(v, index).to(output_device)
                frame.update_column(k, data)
            node_frames.append(frame)
        
        # manually copy edge features
        edge_frames = []
        for etype in frontier.canonical_etypes:
            index = frontier.edges(form='eid', etype=etype).to(self._device)
            if isinstance(index, Mapping):
                index = index[frontier.to_canonical_etype(etype)]
            frame = Frame(num_rows=len(index))
            for k, v in self._e_feat[etype].items():
                data = _gather_row(v, index).to(output_device)
                frame.update_column(k, data)

            # add eids if they do not exist
            if EID not in frame:
                frame.update_column(EID, index.to(output_device))
            edge_frames.append(frame)

        utils.set_new_frames(frontier, node_frames=node_frames,
            edge_frames=edge_frames)

        return frontier

    def get_node_storage(self, key, ntype=None):
        if ntype == None:
            assert len(self._n_feat) == 1, "ntype must be specified for " \
                                           "graphs with more than one ntype."
            ntype = self._n_feat.keys()[0]
        return self._n_feat[ntype][key]
            
    def get_edge_storage(self, key, etype=None):
        if etype == None:
            assert len(self._e_feat) == 1, "etype must be specified for " \
                                           "graphs with more than one etype."
            etype = self._e_feat.keys()[0]
        return self._e_feat[etype][key]
 
    def __getattr__(self, key):
        if key in ['ntypes', 'etypes', 'canonical_etypes',
                   'subgraph', 'edge_subgraph', 'find_edges', 'num_nodes']:
            # Delegate to the wrapped GraphStorage instance.
            return getattr(self._g, key)
        else:
            return super().__getattr__(key)

