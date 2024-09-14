"""Feature fetchers"""

from functools import partial
from typing import Dict

import torch

from torch.utils.data import functional_datapipe

from .base import etype_tuple_to_str
from .impl.cooperative_conv import CooperativeConvFunction

from .minibatch_transformer import MiniBatchTransformer


__all__ = [
    "FeatureFetcher",
    "FeatureFetcherStartMarker",
]


def get_feature_key_list(feature_keys, domain):
    """Processes node_feature_keys and extracts their feature keys to a list."""
    if isinstance(feature_keys, Dict):
        return [
            (domain, type_name, feature_name)
            for type_name, feature_names in feature_keys.items()
            for feature_name in feature_names
        ]
    elif feature_keys is not None:
        return [(domain, None, feature_name) for feature_name in feature_keys]
    else:
        return []


@functional_datapipe("mark_feature_fetcher_start")
class FeatureFetcherStartMarker(MiniBatchTransformer):
    """Used to mark the start of a FeatureFetcher and is a no-op. All the
    datapipes created during a FeatureFetcher instantiation are guarenteed to be
    contained between FeatureFetcherStartMarker and FeatureFetcher instances in
    the datapipe graph.
    """

    def __init__(self, datapipe):
        super().__init__(datapipe, self._identity)


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
    overlap_fetch : bool, optional
        If True, the feature fetcher will overlap the UVA feature fetcher
        operations with the rest of operations by using an alternative CUDA
        stream or utilizing asynchronous operations. Default is True.
    cooperative: bool, optional
        Boolean indicating whether Cooperative Minibatching, which was initially
        proposed in
        `Deep Graph Library PR#4337<https://github.com/dmlc/dgl/pull/4337>`__
        and was later first fully described in
        `Cooperative Minibatching in Graph Neural Networks
        <https://arxiv.org/abs/2310.12403>`__. Cooperation between the GPUs
        eliminates duplicate work performed across the GPUs due to the
        overlapping sampled k-hop neighborhoods of seed nodes when performing
        GNN minibatching.
    """

    def __init__(
        self,
        datapipe,
        feature_store,
        node_feature_keys=None,
        edge_feature_keys=None,
        overlap_fetch=True,
        cooperative=False,
    ):
        datapipe = datapipe.mark_feature_fetcher_start()
        self.feature_store = feature_store
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys
        max_val = 0
        if overlap_fetch:
            for feature_key_list in [
                get_feature_key_list(node_feature_keys, "node"),
                get_feature_key_list(edge_feature_keys, "edge"),
            ]:
                for feature_key in feature_key_list:
                    if feature_key not in feature_store:
                        continue
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
                        except AssertionError:
                            pass
        datapipe = datapipe.transform(self._read)
        for i in range(max_val, 0, -1):
            datapipe = datapipe.transform(
                partial(self._execute_stage, i)
            ).buffer(1)
        if max_val > 0:
            datapipe = datapipe.transform(self._final_stage)
        if cooperative:
            datapipe = datapipe.transform(self._cooperative_exchange)
            datapipe = datapipe.buffer()
        super().__init__(datapipe)
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
                    features[key] = (handle if stage > 1 else value, stage - 1)
        return data

    @staticmethod
    def _final_stage(data):
        all_features = [data.node_features] + [
            data.edge_features[i] for i in range(data.num_layers())
        ]
        for features in all_features:
            for key in features:
                value, stage = features[key]
                assert stage == 0
                features[key] = value.wait()
        return data

    def _cooperative_exchange(self, data):
        subgraph = data.sampled_subgraphs[0]
        is_heterogeneous = isinstance(
            self.node_feature_keys, Dict
        ) or isinstance(self.edge_feature_keys, Dict)
        if is_heterogeneous:
            node_features = {key: {} for key, _ in data.node_features.keys()}
            for (key, ntype), feature in data.node_features.items():
                node_features[key][ntype] = feature
            for key, feature in node_features.items():
                new_feature = CooperativeConvFunction.apply(subgraph, feature)
                for ntype, tensor in new_feature.items():
                    data.node_features[(key, ntype)] = tensor
        else:
            for key in data.node_features:
                feature = data.node_features[key]
                new_feature = CooperativeConvFunction.apply(subgraph, feature)
                data.node_features[key] = new_feature
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

        def read_helper(feature_key, index):
            if self.max_num_stages > 0:
                feature = self.feature_store[feature_key]
                num_stages = feature.read_async_num_stages(index.device)
                if num_stages > 0:
                    return (feature.read_async(index), num_stages)
                else:  # Asynchronicity is not needed, compute in _final_stage.

                    class _Waiter:
                        def __init__(self, feature, index):
                            self.feature = feature
                            self.index = index

                        def wait(self):
                            """Returns the stored value when invoked."""
                            result = self.feature.read(self.index)
                            # Ensure there is no memory leak.
                            self.feature = self.index = None
                            return result

                    return (_Waiter(feature, index), 0)
            else:
                domain, type_name, feature_name = feature_key
                return self.feature_store.read(
                    domain, type_name, feature_name, index
                )

        if self.node_feature_keys and input_nodes is not None:
            if is_heterogeneous:
                for type_name, nodes in input_nodes.items():
                    if type_name not in self.node_feature_keys or nodes is None:
                        continue
                    for feature_name in self.node_feature_keys[type_name]:
                        node_features[(type_name, feature_name)] = read_helper(
                            ("node", type_name, feature_name), nodes
                        )
            else:
                for feature_name in self.node_feature_keys:
                    node_features[feature_name] = read_helper(
                        ("node", None, feature_name), input_nodes
                    )
        # Read Edge features.
        if self.edge_feature_keys and num_layers > 0:
            for i in range(num_layers):
                original_edge_ids = data.edge_ids(i)
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
                            edge_features[i][
                                (type_name, feature_name)
                            ] = read_helper(
                                ("edge", type_name, feature_name), edges
                            )
                else:
                    for feature_name in self.edge_feature_keys:
                        edge_features[i][feature_name] = read_helper(
                            ("edge", None, feature_name), original_edge_ids
                        )
        data.set_node_features(node_features)
        data.set_edge_features(edge_features)
        return data
