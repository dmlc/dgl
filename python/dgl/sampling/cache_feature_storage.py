""""Cached Feature Storage"""
class CacheFeatureStorage:
    """Storage for node/edge feature data.

    Cache part of features in specified GPU to accelerate fetching.
    By default, features of nodes with high degree is cached.
    """

    def __init__(self, source_feature_storage, graph, device, cache_rate):
        """Initializes cache by selecting top-degree nodes

        Parameters
        ----------
        source_feature_storage : FeatureStorage
            feature storage for the whole graph. missing feature comes from here
        graph : DGLGraph
            the graph that the feature is associated to
        device : Framework-specific device context object
            The context to store cache. currently must be gpu
        cache_rate : float
            Rate of node features to be cached.
        """
        # * sort nodes according to its degree(in-degree by default)
        # * build a |V| array indicating node's offset in cache
        # * slice top |V|*cache_rate nodes' feature to device
        pass

    def fetch(self, ids, device):
        """Fetch the features of the given node/edge IDs.

        Parameters
        ----------
        ids : Tensor
            Node or edge IDs.

        Returns
        -------
        Tensor
            Feature data stored in PyTorch Tensor.
        """
        # * split ids into 2 arrays of miss & hit nodes
        # * fetch miss feature from source_feature_storage
        # * combine miss feature with hit feature from cache

        # Default implementation directly passes to internal feature storage
        return self.source_feature_storage.fetch(ids, device)
    def async_fetch(self, ids, device):
        """Initiate asynchronous fetching of the features of the given node/edge IDs to
        the given device.

        To get the actual tensor, use the ``await`` expression:

        >>> future = feature_storage.async_fetch(ids, device)   # initiate transfer
        >>> tensor = await future      # get result

        Parameters
        ----------
        ids : Tensor
            Node or edge IDs.
        device : Device
            Device context.

        Returns
        -------
        Awaitable
           A future that ensures the transfer is done and returns the feature tensor
        """
        pass
