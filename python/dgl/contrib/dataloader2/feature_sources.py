#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import torch
import numpy
from collections.abc import Mapping
from ... import backend as F
from ...storages import FeatureStorage, TensorStorage
from ...base import DGLError
from ...utils import gather_pinned_tensor_rows
from .dataloader2 import NODES_TAG, INPUT_NODES_TAG, OUTPUT_NODES_TAG, EDGES_TAG

def feat_as_mapping(feat, types):
    if feat is None:
        return {}
    if isinstance(feat, Mapping):
        return feat

    if len(types) != 1:
        raise ValueError("feat must be mapping for heterogenous graphs.")
    return {types[0]: feat}


class FeatureSource:
    def get_featured_entities(self):
        """
        Get tree of features available from this feature source.

        Returns
        -------
        dict of lists
            The tree has entity classes at the top level, then edge/node
            types. Valid entity classes are:
            * NODES_TAG: All nodes in all graphs.
            * INPUT_NODES_TAG: Source nodes in the first graph of an MFG.
            * OUTPUT_NODES_TAG: Destination nodes in the last graph of an MFG.
            * 'edges': All edges in all graphs.

            For example a typical feature descriptor for a graph convolution
            could look like:
            {
                'nodes:input': ['user', 'item']
                'nodes:output': ['item']
                'edges': ['buys']
            }
            This indicates that only input nodes, output nodes, and edges have
            valid features. The dataloader does not need to request features
            for other ids.
        """
        return {}


    def fetch_features(self, req, output_device):
        """
        Get all of the features corresponding to the given dictionary.

        Parameters
        ----------
        req : dict of dict of tensors
            This should be a dictionary with two levels, the first being the
            graph component (nodes or edges), and the second being
            type (ntype or etype). The values in the second level dictionary
            should be 1-d tensors of the requested IDs. Edge types should be
            specified by their canonical names.

            For example, the request may look like:
            ```
            {
                'nodes': {
                    'user': torch.tensor([0, 1]),
                    'item': torch.tensor([2, 1])
                },
                'edges': {
                    ('user', 'reviews', 'item'): torch.tensor([3]),
                    ('user', 'purchases', 'item'): torch.tensor([1, 3, 4])
                }
            }
            ```

        output_device : torch.device
            The device the returned tensors should be on.

        Returns
        -------
        dict of dicts of dicts of tensors
            The response to the request should match its structure, but in
            place of the requested ids, there should be a dictionary of feature
            names to feature tensors. For example, the response to the above
            request may look like:

            ```
            {
                'nodes': {
                    'user': {
                        'feat': torch.tensor([[-0.5, 0.3, 0.0],
                                              [0.1, 0.1, -0.9])
                    },
                    'item': {
                        'feat': torch.tensor([[0.1, 0.2, 0.4],
                                              [0.2, -0.1, 0.3])
                        'label': torch.tensor([0, 1])
                    }
                },
                'edges': {
                    ('user', 'reviews', 'item'): {
                        'feat': torch.tensor([0.3, 0.0, -0.1, 0.4])
                    }
                }
            }
            ```
            When node/edge types are requested for which there are no features,
            they should not be included in the response tensor. In the above
            case the 'purchases' edge type has no feature associated with it.
        """
        pass


class FeatureRequestHandler:
    def __init__(self):
        self._storages = {}
        self._features = {}


    def set_storage(self, graph_object, object_type, feat_name, storage):
        if F.is_tensor(storage):
            storage = TensorStorage(storage)

        if graph_object not in self._features:
            self._features[graph_object] = {}
        if object_type not in self._features[graph_object]:
            self._features[graph_object][object_type] = []
        if feat_name in self._features[graph_object][object_type]:
            raise DGLError("Duplicate feature name for {} of type {} " \
                "with name {}".format(graph_object, object_type, feat_name))
        self._features[graph_object][object_type].append(feat_name)

        if graph_object == NODES_TAG or \
                graph_object == INPUT_NODES_TAG or \
                graph_object == OUTPUT_NODES_TAG:
            base_graph_object = NODES_TAG
        else:
            base_graph_object = graph_object
        if base_graph_object not in self._storages:
            self._storages[base_graph_object] = {}
        if object_type not in self._storages[base_graph_object]:
            self._storages[base_graph_object][object_type] = {}
        if feat_name in self._storages[base_graph_object][object_type]:
            raise DGLError("Duplicate feature storage for {} of type {} " \
                "with name {}".format(base_graph_object, object_type, feat_name))
        self._storages[base_graph_object][object_type][feat_name] = storage


    def get_featured_entities(self):
        return self._features


    def fetch_features(self, req, output_device):
        resp = {}
        for graph_object, tree in req.items():
            if graph_object not in self._features:
                continue
            resp[graph_object] = {}
            if graph_object == NODES_TAG or \
                    graph_object == INPUT_NODES_TAG or \
                    graph_object == OUTPUT_NODES_TAG:
                base_graph_object = NODES_TAG
                for ntype, ids in tree.items():
                    if ntype not in self._features[graph_object]:
                        # this ntype has no features
                        continue
                    resp[graph_object][ntype] = {}
                    base_storage = self._storages[base_graph_object]
                    feat_names = self._features[graph_object][ntype]
                    for feat_name in feat_names:
                        tensor = base_storage[ntype][feat_name].fetch( \
                            ids, output_device)
                        resp[graph_object][ntype][feat_name] = tensor
            elif graph_object == EDGES_TAG:
                for etype, ids in tree.items():
                    if etype not in self._features[graph_object]:
                        # this etype has no features
                        continue
                    resp[graph_object][etype] = {}
                    feat_names = self._features[graph_object][etype]
                    for feat_name in feat_names:
                        tensor = self._storages[EDGES_TAG][ntype][feat_name].fetch( \
                            ids, output_device)
                        resp[graph_object][etype][feat_name] = tensor
            else:
                raise DGLError("Unknown component '{}'".format(comp))
        return resp


def _feat_mapping_of_mapping(feat_tree, types):
    if feat_tree is None:
        return {}

    if len(types) != 1:
        raise ValueError("feat must be mapping for heterogenous graphs.")

    has_type = False
    for k, v in feat_tree.items():
        if isinstance(v, Mapping):
            has_type = True
        else:
            assert has_type == False
    if len(feat_tree) > 0 and not has_type:
        feat_tree = {types[0]: feat_tree}
    return feat_tree


class TensorFeatureSource:
    def __init__(self, node_feats=None, edge_feats=None, input_feats=None,
            output_feats=None):
        self._handler = FeatureRequestHandler()

        for ntype, tree in _feat_mapping_of_mapping(node_feats, ['_N']).items():
            for feat_name, tensor in tree.items():
                self._handler.set_storage(NODES_TAG, ntype, feat_name, tensor)

        for etype, tree in _feat_mapping_of_mapping( \
                edge_feats, [('_N', '_E', '_N')]).items():
            for feat_name, tensor in tree.items():
                self._handler.set_storage(EDGES_TAG, etype, feat_name, tensor)

        for ntype, tree in _feat_mapping_of_mapping(input_feats, ['_N']).items():
            for feat_name, tensor in tree.items():
                self._handler.set_storage(INPUT_NODES_TAG, ntype, feat_name, tensor)

        for ntype, tree in _feat_mapping_of_mapping(output_feats, ['_N']).items():
            for feat_name, tensor in tree.items():
                self._handler.set_storage(OUTPUT_NODES_TAG, ntype, feat_name, tensor)

        self.get_featured_entities = self._handler.get_featured_entities


    def fetch_features(self, req, output_device):
        # convert from a string if need be
        output_device = torch.device(output_device)

        return self._handler.fetch_features(req, output_device)


class GraphFeatureSource:
    def __init__(self, g, node_feats=None, edge_feats=None, input_feats=None,
            output_feats=None):

        self._node_feats = feat_as_mapping(node_feats, g.ntypes)
        self._edge_feats = feat_as_mapping(edge_feats, g.canonical_etypes)
        self._input_feats = feat_as_mapping(input_feats, g.ntypes)
        self._output_feats = feat_as_mapping(output_feats, g.ntypes)

        self._source = TensorFeatureSource(
            node_feats={ntype: \
                {feat_name: g.nodes[ntype].data[feat_name] \
                    for feat_name in self._node_feats[ntype]} \
                for ntype in self._node_feats.keys() \
            }, \
            edge_feats={etype: \
                {feat_name: g.edges[etype].data[feat_name] \
                    for feat_name in self._edge_feats[etype]} \
                for etype in self._edge_feats.keys() \
            }, \
            input_feats={ntype: \
                {feat_name: g.nodes[ntype].data[feat_name] \
                    for feat_name in self._input_feats[ntype]} \
                for ntype in self._input_feats.keys() \
            }, \
            output_feats={ntype: \
                {feat_name: g.nodes[ntype].data[feat_name] \
                    for feat_name in self._output_feats[ntype]} \
                for ntype in self._output_feats.keys() \
            } \
        )
        self.get_featured_entities = self._source.get_featured_entities
        self.fetch_features = self._source.fetch_features


class _RedisStorage(FeatureStorage):
    def __init__(self, source, table_name):
        self._source = source
        self._name = table_name

    def fetch(self, indices, device):
        db = self._source._db
        table_types = self._source._table_types

        # to have any kind of performance, we would need to go from redis to
        # dlpack in C++ rather than python.
        table_keys = [idx.numpy().tobytes() for idx in indices.cpu()]
        table_values = db.hmget(self._name, table_keys)
        tensor = torch.stack([ \
            torch.from_numpy(numpy.frombuffer(row, \
                dtype=table_types[self._name])) \
            for row in table_values])
        if tensor.shape[1] == 1:
            tensor = tensor.squeeze(-1)
        return tensor.to(device)



class RedisFeatureSource:
    @staticmethod
    def _get_table_name(item, item_type, feat_name):
        if item == EDGES_TAG:
            item_type = item_type[0] + "_" + item_type[1] + "_" + item_type[2]
        table_name = item + ":" + item_type + ":" + feat_name
        return table_name

    def __init__(self, client, node_feats=None, edge_feats=None, input_feats=None,
            output_feats=None):
        self._db = client
        self._table_types = {}
        self._handler = FeatureRequestHandler()

        for ntype, feat_names in feat_as_mapping(node_feats, ['_N']).items():
            for feat_name in feat_names:
                storage = _RedisStorage(self, self._get_table_name( \
                    NODES_TAG, ntype, feat_name))
                self._handler.set_storage(NODES_TAG, ntype, feat_name, storage)

        for etype, feat_names in feat_as_mapping( \
                edge_feats, [('_N', '_E', '_N')]).items():
            for feat_name in feat_names:
                storage = _RedisStorage(self, self._get_table_name( \
                    EDGES_TAG, etype, feat_name))
                self._handler.set_storage(EDGES_TAG, etype, feat_name, storage)

        for ntype, feat_names in feat_as_mapping(input_feats, ['_N']).items():
            for feat_name in feat_names:
                storage = _RedisStorage(self, self._get_table_name( \
                    NODES_TAG, ntype, feat_name))
                self._handler.set_storage(INPUT_NODES_TAG, ntype, feat_name, storage)

        for ntype, feat_names in feat_as_mapping(output_feats, ['_N']).items():
            for feat_name in feat_names:
                storage = _RedisStorage(self, self._get_table_name( \
                    NODES_TAG, ntype, feat_name))
                self._handler.set_storage(OUTPUT_NODES_TAG, ntype, feat_name, storage)

        self.get_featured_entities = self._handler.get_featured_entities
        self.fetch_features = self._handler.fetch_features


    def set_table(self, item, item_type, feat_name, tensor):
        # to have any kind of performance, we would need to go from dlpack to
        # redis in C++ rather than python.
        table_name = self._get_table_name(item, item_type, feat_name)

        np_tensor = tensor.numpy()

        self._table_types[table_name] = np_tensor.dtype

        # insert smaller batches at a time
        chunk_size = 4096
        for chunk in range(0, len(tensor), chunk_size):
            chunk_end = min(chunk+chunk_size, len(tensor))
            batch = {
                idx.numpy().tobytes(): np_tensor[idx].tobytes() \
                for idx in torch.arange(chunk, chunk_end) \
            }
            self._db.hmset(table_name, batch)
