"""Feature store for GraphBolt."""

from typing import NamedTuple

import torch

__all__ = ["Feature", "FeatureStore", "FeatureKey"]


class FeatureKey(NamedTuple):
    """A named tuple class to represent feature keys in FeatureStore classes.
    The fields are domain, type and name all of which take string values.
    """

    domain: str
    type: str
    name: int


class Feature:
    r"""A wrapper of feature data for access."""

    def __init__(self):
        pass

    def read(self, ids: torch.Tensor = None):
        """Read from the feature.

        Parameters
        ----------
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.
        Returns
        -------
        torch.Tensor
            The read feature.
        """
        raise NotImplementedError

    def read_async(self, ids: torch.Tensor):
        """Read the feature by index asynchronously.
        Parameters
        ----------
        ids : torch.Tensor
            The index of the feature. Only the specified indices of the
            feature are read.
        Returns
        -------
        A generator object.
            The returned generator object returns a future on
            `read_async_num_stages(ids.device)`th invocation. The return result
            can be accessed by calling `.wait()`. on the returned future object.
            It is undefined behavior to call `.wait()` more than once.
        Example Usage
        --------
        >>> import dgl.graphbolt as gb
        >>> feature = gb.Feature(...)
        >>> ids = torch.tensor([0, 2])
        >>> async_handle = feature.read_async(ids)
        >>> for _ in range(feature.read_async_num_stages(ids.device)):
        ...     future = next(async_handle)
        >>> result = future.wait()  # result contains the read values.
        """
        raise NotImplementedError

    def read_async_num_stages(self, ids_device: torch.device):
        """The number of stages of the read_async operation. See read_async
        function for directions on its use.
        Parameters
        ----------
        ids_device : torch.device
            The device of the ids parameter passed into read_async.
        Returns
        -------
        int
            The number of stages of the read_async operation.
        """
        raise NotImplementedError

    def size(self):
        """Get the size of the feature.

        Returns
        -------
        torch.Size
            The size of the feature.
        """
        raise NotImplementedError

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
        """Update the feature.

        Parameters
        ----------
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The indices of the feature to update. If specified, only the
            specified indices of the feature will be updated. For the feature,
            the `ids[i]` row is updated to `value[i]`. So the indices and value
            must have the same length. If None, the entire feature will be
            updated.
        """
        raise NotImplementedError

    def metadata(self):
        """Get the metadata of the feature.

        Returns
        -------
        Dict
            The metadata of the feature.
        """
        return {}


class FeatureStore:
    r"""A store to manage multiple features for access."""

    def __init__(self):
        pass

    def __getitem__(self, feature_key: FeatureKey) -> Feature:
        """Access the underlying `Feature` with its (domain, type, name) as
        the feature_key.
        """
        raise NotImplementedError

    def __setitem__(self, feature_key: FeatureKey, feature: Feature):
        """Set the underlying `Feature` with its (domain, type, name) as
        the feature_key and feature as the value.
        """
        raise NotImplementedError

    def __contains__(self, feature_key: FeatureKey) -> bool:
        """Checks whether the provided (domain, type, name) as the feature_key
        is container in the FeatureStore."""
        raise NotImplementedError

    def read(
        self,
        domain: str,
        type_name: str,
        feature_name: str,
        ids: torch.Tensor = None,
    ):
        """Read from the feature store.

        Parameters
        ----------
        domain : str
            The domain of the feature such as "node", "edge" or "graph".
        type_name : str
            The node or edge type name.
        feature_name : str
            The feature name.
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.

        Returns
        -------
        torch.Tensor
            The read feature.
        """
        return self.__getitem__((domain, type_name, feature_name)).read(ids)

    def size(
        self,
        domain: str,
        type_name: str,
        feature_name: str,
    ):
        """Get the size of the specified feature in the feature store.

        Parameters
        ----------
        domain : str
            The domain of the feature such as "node", "edge" or "graph".
        type_name : str
            The node or edge type name.
        feature_name : str
            The feature name.
        Returns
        -------
        torch.Size
            The size of the specified feature in the feature store.
        """
        return self.__getitem__((domain, type_name, feature_name)).size()

    def metadata(
        self,
        domain: str,
        type_name: str,
        feature_name: str,
    ):
        """Get the metadata of the specified feature in the feature store.

        Parameters
        ----------
        domain : str
            The domain of the feature such as "node", "edge" or "graph".
        type_name : str
            The node or edge type name.
        feature_name : str
            The feature name.
        Returns
        -------
        Dict
            The metadata of the feature.
        """
        return self.__getitem__((domain, type_name, feature_name)).metadata()

    def update(
        self,
        domain: str,
        type_name: str,
        feature_name: str,
        value: torch.Tensor,
        ids: torch.Tensor = None,
    ):
        """Update the feature store.

        Parameters
        ----------
        domain : str
            The domain of the feature such as "node", "edge" or "graph".
        type_name : str
            The node or edge type name.
        feature_name : str
            The feature name.
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The indices of the feature to update. If specified, only the
            specified indices of the feature will be updated. For the feature,
            the `ids[i]` row is updated to `value[i]`. So the indices and value
            must have the same length. If None, the entire feature will be
            updated.
        """
        self.__getitem__((domain, type_name, feature_name)).update(value, ids)

    def keys(self):
        """Get the keys of the features.

        Returns
        -------
        List[tuple]
            The keys of the features. The tuples are in `(domain, type_name,
            feat_name)` format.
        """
        raise NotImplementedError
