"""Feature fetchers"""

from typing import Dict

from torch.utils.data import functional_datapipe

from torchdata.datapipes.iter import Mapper


@functional_datapipe("fetch_feature")
class FeatureFetcher(Mapper):
    """A feature fetcher used to fetch features for node/edge in graphbolt."""

    def __init__(
        self,
        datapipe,
        feature_store,
        node_feature_keys=None,
        edge_feature_keys=None,
    ):
        """
        Initlization for a feature fetcher.

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
        super().__init__(datapipe, self._read)
        self.feature_store = feature_store
        self.node_feature_keys = node_feature_keys
        self.edge_feature_keys = edge_feature_keys

    def _read(self, data):
        """
        Fill in the node/edge features field in data.

        Parameters
        ----------
        data : DataBlock
            An instance of the 'DataBlock' class. Even if 'node_feature' or
            'edge_feature' is already filled, it will be overwritten for
            overlapping features.

        Returns
        -------
        DataBlock
            An instance of 'DataBlock' filled with required features.
        """
        data.node_features = {}
        num_layer = len(data.sampled_subgraphs) if data.sampled_subgraphs else 0
        data.edge_features = [{} for _ in range(num_layer)]
        is_heterogeneous = isinstance(
            self.node_feature_keys, Dict
        ) or isinstance(self.edge_feature_keys, Dict)
        # Read Node features.
        if self.node_feature_keys and data.input_nodes is not None:
            if is_heterogeneous:
                for type_name, feature_names in self.node_feature_keys.items():
                    nodes = data.input_nodes[type_name]
                    if nodes is None:
                        continue
                    for feature_name in feature_names:
                        data.node_features[
                            (type_name, feature_name)
                        ] = self.feature_store.read(
                            "node",
                            type_name,
                            feature_name,
                            nodes,
                        )
            else:
                for feature_name in self.node_feature_keys:
                    data.node_features[feature_name] = self.feature_store.read(
                        "node",
                        None,
                        feature_name,
                        data.input_nodes,
                    )
        # Read Edge features.
        if self.edge_feature_keys and data.sampled_subgraphs:
            for i, subgraph in enumerate(data.sampled_subgraphs):
                if subgraph.reverse_edge_ids is None:
                    continue
                if is_heterogeneous:
                    for (
                        type_name,
                        feature_names,
                    ) in self.edge_feature_keys.items():
                        edges = subgraph.reverse_edge_ids.get(type_name, None)
                        if edges is None:
                            continue
                        for feature_name in feature_names:
                            data.edge_features[i][
                                (type_name, feature_name)
                            ] = self.feature_store.read(
                                "edge", type_name, feature_name, edges
                            )
                else:
                    for feature_name in self.edge_feature_keys:
                        data.edge_features[i][
                            feature_name
                        ] = self.feature_store.read(
                            "edge",
                            None,
                            feature_name,
                            subgraph.reverse_edge_ids,
                        )
        return data
