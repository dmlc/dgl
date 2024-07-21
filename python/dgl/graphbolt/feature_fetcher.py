"""Feature fetchers"""

from functools import partial
from typing import Dict

import torch

from torch.utils.data import functional_datapipe

from .base import etype_tuple_to_str

from .minibatch_transformer import MiniBatchTransformer


__all__ = [
    "FeatureFetcher",
]


@functional_datapipe("fetch_feature")
class FeatureFetcher(MiniBatchTransformer):
    """A feature fetcher used to fetch features for node/edge in graphbolt.

    Functional name: :obj:`fetch_feature`.

    Parameters
    ----------
    datapipe : DataPipe
        The datapipe.
    feature_store : FeatureStore
        A storage for features, support read and update.
    node_feature_keys : List[str] or Dict[str, List[str]]
        Node features keys indicates the node features need to be read.
        - If `node_features` is a list: It means the graph is homogeneous
        graph, and the 'str' inside are feature names.
        - If `node_features` is a dictionary: The keys should be node type
        and the values are lists of feature names.
    edge_feature_keys : List[str] or Dict[str, List[str]]
        Edge features name indicates the edge features need to be read.
        - If `edge_features` is a list: It means the graph is homogeneous
        graph, and the 'str' inside are feature names.
        - If `edge_features` is a dictionary: The keys are edge types,
        following the format 'str:str:str', and the values are lists of
        feature names.
    """

    def __init__(
        self,
        datapipe,
        feature_store,
        node_feature_keys=None,
        edge_feature_keys=None,
        overlap_fetch=False,
    ):
        self.feature_store = feature_store
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys
        max_val = 0
        if overlap_fetch:
            if isinstance(node_feature_keys, Dict):
                node_feature_key_list = [
                    ("node", type_name, feature_name)
                    for type_name, feature_names in node_feature_keys.items()
                    for feature_name in feature_names
                ]
            elif node_feature_keys is not None:
                node_feature_key_list = [
                    ("node", None, feature_name)
                    for feature_name in node_feature_keys
                ]
            else:
                node_feature_key_list = []
            if isinstance(edge_feature_keys, Dict):
                edge_feature_key_list = [
                    ("edge", type_name, feature_name)
                    for type_name, feature_names in edge_feature_keys.items()
                    for feature_name in feature_names
                ]
            elif edge_feature_keys is not None:
                edge_feature_key_list = [
                    ("edge", None, feature_name)
                    for feature_name in edge_feature_keys
                ]
            else:
                edge_feature_key_list = []
            for feature_key_list in [
                node_feature_key_list,
                edge_feature_key_list,
            ]:
                for feature_key in feature_key_list:
                    for device_str in ["cpu", "cuda"]:
                        try:
                            max_val = max(
                                feature_store[
                                    feature_key
                                ].read_async_num_stages(
                                    torch.device(device_str)
                                ),
                                max_val,
                            )
                        except:
                            pass
        datapipe = datapipe.transform(self._read)
        for i in range(max_val, 0, -1):
            datapipe = datapipe.transform(
                partial(self._execute_stage, i)
            ).buffer(1)
        super().__init__(
            datapipe, self._identity if max_val == 0 else self._final_stage
        )
        # A positive value indicates that the overlap optimization is enabled.
        self.max_num_stages = max_val

    @staticmethod
    def _execute_stage(current_stage, data):
        all_features = [data.node_features] + [
            data.edge_features[i] for i in range(data.num_layers())
        ]
        for features in all_features:
            for key in features:
                handle, stage = features[key]
                assert current_stage >= stage
                if current_stage == stage:
                    value = next(handle)
                    features[key] = (handle, stage - 1) if stage > 1 else value
        return data

    @staticmethod
    def _final_stage(data):
        all_features = [data.node_features] + [
            data.edge_features[i] for i in range(data.num_layers())
        ]
        for features in all_features:
            for key in features:
                features[key] = features[key].wait()
        return data

    @staticmethod
    def _identity(data):
        return data

    def _read(self, data):
        """
        Fill in the node/edge features field in data.

        Parameters
        ----------
        data : MiniBatch
            An instance of :class:`MiniBatch`. Even if 'node_feature' or
            'edge_feature' is already filled, it will be overwritten for
            overlapping features.

        Returns
        -------
        MiniBatch
            An instance of :class:`MiniBatch` filled with required features.
        """
        node_features = {}
        num_layers = data.num_layers()
        edge_features = [{} for _ in range(num_layers)]
        is_heterogeneous = isinstance(
            self.node_feature_keys, Dict
        ) or isinstance(self.edge_feature_keys, Dict)
        # Read Node features.
        input_nodes = data.node_ids()

        if self.node_feature_keys and input_nodes is not None:
            if is_heterogeneous:
                for type_name, nodes in input_nodes.items():
                    if type_name not in self.node_feature_keys or nodes is None:
                        continue
                    for feature_name in self.node_feature_keys[type_name]:
                        if self.max_num_stages > 0:
                            feature = self.feature_store[
                                ("node", type_name, feature_name)
                            ]
                            result = (
                                feature.read_async(nodes),
                                feature.read_async_num_stages(nodes.device),
                            )
                        else:
                            result = self.feature_store.read(
                                "node",
                                type_name,
                                feature_name,
                                nodes,
                            )
                        node_features[(type_name, feature_name)] = result
            else:
                for feature_name in self.node_feature_keys:
                    if self.max_num_stages > 0:
                        feature = self.feature_store[
                            ("node", None, feature_name)
                        ]
                        result = (
                            feature.read_async(input_nodes),
                            feature.read_async_num_stages(input_nodes.device),
                        )
                    else:
                        result = self.feature_store.read(
                            "node",
                            None,
                            feature_name,
                            input_nodes,
                        )
                    node_features[feature_name] = result
        # Read Edge features.
        if self.edge_feature_keys and num_layers > 0:
            for i in range(num_layers):
                original_edge_ids = data.edge_ids(i)
                if original_edge_ids is None:
                    continue
                if is_heterogeneous:
                    # Convert edge type to string.
                    original_edge_ids = {
                        (
                            etype_tuple_to_str(key)
                            if isinstance(key, tuple)
                            else key
                        ): value
                        for key, value in original_edge_ids.items()
                    }
                    for type_name, edges in original_edge_ids.items():
                        if (
                            type_name not in self.edge_feature_keys
                            or edges is None
                        ):
                            continue
                        for feature_name in self.edge_feature_keys[type_name]:
                            if self.max_num_stages > 0:
                                feature = self.feature_store[
                                    ("edge", type_name, feature_name)
                                ]
                                result = (
                                    feature.read_async(edges),
                                    feature.read_async_num_stages(edges.device),
                                )
                            else:
                                result = self.feature_store.read(
                                    "edge", type_name, feature_name, edges
                                )
                            edge_features[i][(type_name, feature_name)] = result
                else:
                    for feature_name in self.edge_feature_keys:
                        if self.max_num_stages > 0:
                            feature = self.feature_store[
                                ("edge", None, feature_name)
                            ]
                            result = (
                                feature.read_async(original_edge_ids),
                                feature.read_async_num_stages(
                                    original_edge_ids.device
                                ),
                            )
                        else:
                            result = self.feature_store.read(
                                "edge",
                                None,
                                feature_name,
                                original_edge_ids,
                            )
                        edge_features[i][feature_name] = result
        data.set_node_features(node_features)
        data.set_edge_features(edge_features)
        return data
