"""Feature store for GraphBolt."""
import torch


class FeatureStore:
    r"""Base class for feature store."""

    def __init__(self):
        pass

    def read_feature(self, key: str, ids: torch.Tensor = None):
        """Read a feature from the feature store.

        Parameters
        ----------
        key : str
            The key that uniquely identifies the feature in the feature store.
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.

        Returns
        -------
        torch.Tensor
            The read feature.
        """
        raise NotImplementedError

    def add_feature(self, key: str, value: torch.Tensor):
        """Add a new feature to the feature store. If the feature already
        exists, it will be overwritten.

        Parameters
        ----------
        key : str
            The key that uniquely identifies the feature in the feature store.
        value : torch.Tensor
            The value of the new feature.
        """
        raise NotImplementedError

    def update_feature(
        self, key: str, value: torch.Tensor, ids: torch.Tensor = None
    ):
        """Update a feature in the feature store.

        This function is used to update a feature in the feature store. The
        feature is identified by a unique key, and its value is specified using
        a tensor.

        Parameters
        ----------
        key : str
            The key that uniquely identifies the feature in the feature store.
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature will be updated. If None, the entire feature will be
            updated.
        """
        raise NotImplementedError


class InMemoryFeatureStore(FeatureStore):
    r"""In-memory key-value feature store, where the key is a string and value
    is Pytorch tensor."""

    def __init__(self, feature_dict: dict = None):
        """Initialize an in-memory feature store.

        Parameters
        ----------
        feature_dict : dict, optional
            A dictionary of tensors, where the key is the name of a feature and
            the value is the tensor. If None, creates an empty feature store.

        Examples
        --------
        >>> import torch
        >>> feature_dict = {
        ...     "user": torch.arange(0, 5),
        ...     "item": torch.arange(0, 6),
        ... }
        >>> feature_store = InMemoryFeatureStore(feature_dict)
        >>> feature_store.get_items("user", torch.tensor([0, 1, 2]))
        tensor([0, 1, 2])
        >>> feature_store.get_items("item", torch.tensor([0, 1, 2]))
        tensor([0, 1, 2])
        >>> feature_store.set_items("user", torch.tensor([0, 1, 2]),
        ... torch.ones(3))
        >>> feature_store.get_items("user", torch.tensor([0, 1, 2]))
        tensor([1., 1., 1.])
        """
        super(InMemoryFeatureStore, self).__init__()
        for k, v in feature_dict.items():
            assert isinstance(
                k, str
            ), f"Key in InMemoryFeatureStore must be str, but got {k}."
            assert isinstance(v, torch.Tensor), (
                f"Value in InMemoryFeatureStore must be torch.Tensor,"
                f"but got {v}."
            )

        self._feature_dict = feature_dict

    def read_feature(self, key: str, ids: torch.Tensor = None):
        """Read a feature from the feature store by index.

        Parameters
        ----------
        key : str
            The key of the feature.
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.

        Returns
        -------
        torch.Tensor
            The read feature.
        """
        assert (
            key in self._feature_dict
        ), f"key {key} not in {self._feature_dict.keys()}"
        if ids is None:
            return self._feature_dict[key]
        return self._feature_dict[key][ids]

    def add_feature(self, key: str, value: torch.Tensor):
        """Add a new feature to the feature store. If the feature already
        exists, it will be overwritten.

        Parameters
        ----------
        key : str
            The key that uniquely identifies the feature in the feature store.
        value : torch.Tensor
            The value of the new feature.
        """
        self._feature_dict[key] = value

    def update_feature(
        self, key: str, value: torch.Tensor, ids: torch.Tensor = None
    ):
        """Update a feature in the feature store.

        This function is used to update a feature in the feature store. The
        feature is identified by a unique key, and its value is specified using
        a tensor.

        Parameters
        ----------
        key : str
            The key that uniquely identifies the feature in the feature store.
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature will be updated. If None, the entire feature will be
            updated.
        """
        assert (
            key in self._feature_dict
        ), f"key {key} not in {self._feature_dict.keys()}"
        self._feature_dict[key][ids] = value
