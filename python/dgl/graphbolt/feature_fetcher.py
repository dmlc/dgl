"""Feature fetchers"""

from torchdata.datapipes.iter import Mapper


class FeatureFetcher(Mapper):
    """A feature fetcher used to fetch features for node/edge in graphbolt."""

    def __init__(self, datapipe, feature_store, feature_keys):
        """
        Initlization for a feature fetcher.

        Parameters
        ----------
        datapipe : DataPipe
            The datapipe.
        feature_store : FeatureStore
            A storage for features, support read and update.
        feature_keys:   (str, str, str)
            Features need to be read, with each feature being uniquely identified
            by a triplet '(domain, type_name, feature_name)'.
        """
        super().__init__(datapipe, self._read)
        self.feature_store = feature_store
        self.feature_keys = feature_keys

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
        for key in self.feature_keys:
            domain, type_name, name = key
            if domain == "node":
                data.node_feature[(type_name, name)] = self.feature_store.read(
                    key, data.input_nodes[type_name]
                )
            elif domain == "edge":
                for i, sub_graph in enumerate(data.sampled_subgraphs):
                    edge_ids = sub_graph.reverse_edge_ids
                    data.edge_feature[i][
                        (type_name, name)
                    ] = self.feature_store.read(key, edge_ids)
