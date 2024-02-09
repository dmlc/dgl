"""Feature fetchers"""

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
    ):
        super().__init__(datapipe, self._read)
        self.feature_store = feature_store
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys
        self.stream = None

    def _read_data(self, data, stream):
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

        def record_stream(tensor):
            if stream is not None and tensor.is_cuda:
                tensor.record_stream(stream)
            return tensor

        if self.node_feature_keys and input_nodes is not None:
            if is_heterogeneous:
                for type_name, feature_names in self.node_feature_keys.items():
                    nodes = input_nodes[type_name]
                    if nodes is None:
                        continue
                    if nodes.is_cuda:
                        nodes.record_stream(torch.cuda.current_stream())
                    for feature_name in feature_names:
                        node_features[
                            (type_name, feature_name)
                        ] = record_stream(
                            self.feature_store.read(
                                "node",
                                type_name,
                                feature_name,
                                nodes,
                            )
                        )
            else:
                if input_nodes.is_cuda:
                    input_nodes.record_stream(torch.cuda.current_stream())
                for feature_name in self.node_feature_keys:
                    node_features[feature_name] = record_stream(
                        self.feature_store.read(
                            "node",
                            None,
                            feature_name,
                            input_nodes,
                        )
                    )
        # Read Edge features.
        if self.edge_feature_keys and num_layers > 0:
            for i in range(num_layers):
                original_edge_ids = data.edge_ids(i)
                if original_edge_ids is None:
                    continue
                if is_heterogeneous:
                    # Convert edge type to string.
                    original_edge_ids = {
                        etype_tuple_to_str(key)
                        if isinstance(key, tuple)
                        else key: value
                        for key, value in original_edge_ids.items()
                    }
                    for (
                        type_name,
                        feature_names,
                    ) in self.edge_feature_keys.items():
                        edges = original_edge_ids.get(type_name, None)
                        if edges is None:
                            continue
                        if edges.is_cuda:
                            edges.record_stream(torch.cuda.current_stream())
                        for feature_name in feature_names:
                            edge_features[i][
                                (type_name, feature_name)
                            ] = record_stream(
                                self.feature_store.read(
                                    "edge", type_name, feature_name, edges
                                )
                            )
                else:
                    if original_edge_ids.is_cuda:
                        original_edge_ids.record_stream(
                            torch.cuda.current_stream()
                        )
                    for feature_name in self.edge_feature_keys:
                        edge_features[i][feature_name] = record_stream(
                            self.feature_store.read(
                                "edge",
                                None,
                                feature_name,
                                original_edge_ids,
                            )
                        )
        data.set_node_features(node_features)
        data.set_edge_features(edge_features)
        return data

    def _read(self, data):
        current_stream = None
        if self.stream is not None:
            current_stream = torch.cuda.current_stream()
            self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            data = self._read_data(data, current_stream)
            if self.stream is not None:
                data.wait = torch.cuda.current_stream().record_event().wait
            return data
