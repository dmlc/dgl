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
""" The MultiGPUFeatureGraphWrapper class. """

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


class _MultiGPUFrame(object):
    def __init__(self):
        self._column_names = []

    def subframe(self, rowids):
        frame = Frame(num_rows = len(rowids))
        for name in self._column_names:
            col = self._get_storage(name).fetch(rowids, F.context(rowids))
            frame.update_column(name, col)
        return frame

    def add_column(self, name):
        self._column_names.append(name)


class _MultiGPUNodeFrame(_MultiGPUFrame):
    def __init__(self, ntype, data):
        super().__init__()
        self._ntype = ntype
        self._data = data

    def _get_storage(self, name):
        return self._data.get_node_storage(name, self._ntype)


class _MultiGPUEdgeFrame(_MultiGPUFrame):
    def __init__(self, etype, data):
        super().__init__()
        self._etype = etype
        self._data = data

    def _get_storage(self, name):
        return self._data.get_edge_storage(name, self._etype)


class MultiGPUFeatureGraphWrapper(object):
    """This class wraps a DGLGraph object and enables neighbor sampling where
    the features are stored split across the GPUs.

    NOTE: This class is currently experimental, and it's interface and
    functionality are subject to change in future versions of DGL.

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

    >>> sampler = MultiLayerNeighborSampler([15, 10, 5])
    >>> dataloader = NodeDataLoader(
    ...     MultiGPUFeatureGraphWrapper(g, dev_id), train_nid, sampler,
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
        self._n_frames = []
        for i, ntype in enumerate(g.ntypes):
            feats = {}
            self._n_frames.append(_MultiGPUNodeFrame(ntype, self))
            for feat_name in list(g._node_frames[i].keys()):
                if isinstance(g.ndata[feat_name], Mapping):
                    data = g.ndata[feat_name][ntype]
                else:
                    data = g.ndata[feat_name]
                feats[feat_name] = _load_tensor(data, device, comm,
                                                partition[ntype])
                self._n_frames[i].add_column(feat_name)
            self._n_feat[ntype] = feats

        # save all edge features to GPU
        self._e_feat = {}
        self._e_frames = []
        for i, etype in enumerate(g.canonical_etypes):
            feats = {}
            self._e_frames.append(_MultiGPUEdgeFrame(etype, self))
            for feat_name in list(g._edge_frames[i].keys()):
                if isinstance(g.edata[feat_name], Mapping):
                    data = g.edata[feat_name][etype]
                else:
                    data = g.edata[feat_name]
                feats[feat_name] = _load_tensor(data, device, comm,
                                                edge_partition[etype])
                self._e_frames[i].add_column(feat_name)
            self._e_feat[etype] = feats

    def sample_neighbors(self, nodes, fanout, edge_dir='in', prob=None,
                         replace=False, output_device=None, exclude_edges=None):
        """Sample neighboring edges of the given nodes and return the
        induced subgraph.

        Parameters
        ----------
        nodes : tensor or dict
            The nodes to sample from.
        fanout : int or dict[etype, int]
            The number of edges to sample per node. For graphs with multiple
            edge types, a dictionary can be supplied with a different fanout
            per type. A value of -1 will result in all edges being selected.
        edge_dir : str, optional
            The direction of edges to sample. ``in`` for inbound edges and
            ``out`` for outbound edges.
        prob : str, optional
            The feature name used a probability for each edge. Feature must be
            a single dimension of float type, and all value must be
            non-negative.
        replace : bool, optional
            Whether to sample with replacement.
        output_device : Framework-specific device context object, optional
            The output device. Default is the same as this objet.
        exclude_edges: tensor or dict
            Edges to exclude during neihgbor sampling.

        Returns
        -------
        DGLGraph
            A sampled subgraph containing only the sampled neighboring edges.
        """
        if output_device is None:
            output_device = self._device
        assert output_device == self._device, \
            "The output device of the sampler must be the same device as " \
            "the GPU feature store: {} vs. {}".format(
                output_device, self._device)

        # _dist_training must be true to get the eid's properly attached
        # without copying features from the CPU
        frontier = neighbor.sample_neighbors(
            g=self._g, nodes=nodes, fanout=fanout, edge_dir=edge_dir,
            prob=prob, replace=replace, copy_ndata=False, copy_edata=False,
            exclude_edges=exclude_edges, output_device=output_device,
            _dist_training=True)

        induced_edges = {etype: frontier.edges[etype].data[EID] \
            for etype in frontier.canonical_etypes}

        # set node frames
        utils.set_new_frames(frontier, node_frames=self._n_frames)

        # need to insert eid into edge frames
        e_frames = []
        for i, etype in enumerate(frontier.canonical_etypes):
            e_frames.append(self._e_frames[i].subframe(induced_edges[etype]))
            e_frames[i][EID] = induced_edges[etype]

        # set edge frames
        utils.set_new_frames(frontier, edge_frames=e_frames)

        return frontier

    def get_node_storage(self, key, ntype=None):
        """Get the node feature storage for the given features 'key' and
        'ntype'.

        Paramters
        ---------
        key : String
            The name of the feature to get.
        ntype : String, Tuple
            The edge type to get the feature of.

        Returns
        -------
        FeatureStorage
            The feature storage of the given feature.
        """
        if ntype == None:
            assert len(self._n_feat) == 1, "ntype must be specified for " \
                                           "graphs with more than one ntype."
            ntype = self._n_feat.keys()[0]
        return self._n_feat[ntype][key]

    def get_edge_storage(self, key, etype=None):
        """Get the edge feature storage for the given features 'key' and
        'etype'.

        Paramters
        ---------
        key : String
            The name of the feature to get.
        etype : String, Tuple
            The edge type to get the feature of.

        Returns
        -------
        FeatureStorage
            The feature storage of the given feature.
        """
        if etype == None:
            assert len(self._e_feat) == 1, "etype must be specified for " \
                                           "graphs with more than one etype."
            etype = self._e_feat.keys()[0]
        return self._e_feat[etype][key]

    @property
    def ntypes(self):
        return self._g.ntypes

    @property
    def etypes(self):
        return self._g.etypes

    @property
    def canonical_etypes(self):
        return self._g.canonical_etypes

    def num_nodes(self, ntype=None):
        return self._g.num_nodes(ntype=ntype)

    @property
    def nodes(self):
        return self._g.nodes
