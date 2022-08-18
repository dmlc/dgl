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

import torch
from collections.abc import Mapping
from .. import backend as F
from ..base import DGLError
from ..utils import gather_pinned_tensor_rows

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
            * 'nodes': All nodes in all graphs.
            * 'nodes:input': Source nodes in the first graph of an MFG.
            * 'nodes:output': Destination nodes in the last graph of an MFG.
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
        self._node_feats =  _feat_mapping_of_mapping(node_feats, ['_N'])
        self._edge_feats =  _feat_mapping_of_mapping(edge_feats, \
                                                     [('_N', '_E', '_N')])
        self._input_feats =  _feat_mapping_of_mapping(input_feats, ['_N'])
        self._output_feats =  _feat_mapping_of_mapping(output_feats, ['_N'])


    def get_featured_entities(self):
        feats = {}
        if len(self._node_feats) > 0:
            feats["nodes"] = list(self._node_feats.keys())
        if len(self._input_feats) > 0:
            feats["nodes:input"] = list(self._input_feats.keys())
        if len(self._output_feats) > 0:
            feats["nodes:output"] = list(self._output_feats.keys())
        if len(self._edge_feats) > 0:
            feats["edges"] = list(self._edge_feats.keys())
        return feats

    def fetch_features(self, req, output_device):
        # convert from a string if need be
        output_device = torch.device(output_device)

        resp = {}
        for comp, tree in req.items():
            if comp == 'nodes' or comp == 'nodes:input' or comp == 'nodes:output':
                resp[comp] = {}
                for ntype, ids in tree.items():
                    resp[comp][ntype] = {}
                    if comp == 'nodes':
                        feats = self._node_feats
                    elif comp == 'nodes:input':
                        feats = self._input_feats
                    else:
                        assert comp == 'nodes:output'
                        feats = self._output_feats
                    if ntype not in feats:
                        # this ntype has no features
                        continue
                    feat_names = feats[ntype].keys()
                    for feat_name in feat_names:
                        tensor = feats[ntype][feat_name]
                        if output_device.type == 'cuda' and F.is_pinned(tensor):
                            resp[comp][ntype][feat_name] = \
                                gather_pinned_tensor_rows(tensor, ids)
                        else:
                            resp[comp][ntype][feat_name] = \
                                tensor[ids].to(output_device)
            elif comp == 'edges':
                resp[comp] = {}
                for etype, ids in tree.items():
                    resp[comp][etype] = {}
                    if etype in self._edge_feats:
                        feat_names = self._edge_feats[etype].keys()
                    else:
                        # this etype has no features
                        continue
                    for feat_name in feat_names:
                        tensor = self._edge_feats[etype][feat_name]
                        if output_device.type == 'cuda' and F.is_pinned(tensor):
                            resp[comp][etype][feat_name] = \
                                gather_pinned_tensor_rows(tensor, ids)
                        else:
                            resp[comp][etype][feat_name] = \
                                tensor[ids].to(output_device)
            else:
                raise DGLError("Unknown component '{}'".format(comp))

        return resp


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


