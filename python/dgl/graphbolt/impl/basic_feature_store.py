"""Basic feature store for GraphBolt."""

from typing import Dict, Tuple

from ..feature_store import Feature, FeatureKey, FeatureStore

__all__ = ["BasicFeatureStore"]


class BasicFeatureStore(FeatureStore):
    r"""A basic feature store to manage multiple features for access."""

    def __init__(self, features: Dict[Tuple[str, str, str], Feature]):
        r"""Initiate a basic feature store.


        Parameters
        ----------
        features : Dict[Tuple[str, str, str], Feature]
            The dict of features served by the feature store, in which the key
            is tuple of (domain, type_name, feature_name).

        Returns
        -------
        The feature stores.
        """
        super().__init__()
        self._features = features

    def __getitem__(self, feature_key: FeatureKey) -> Feature:
        """Access the underlying `Feature` with its (domain, type, name) as
        the feature_key.
        """
        return self._features[feature_key]

    def __setitem__(self, feature_key: FeatureKey, feature: Feature):
        """Set the underlying `Feature` with its (domain, type, name) as
        the feature_key and feature as the value.
        """
        self._features[feature_key] = feature

    def __contains__(self, feature_key: FeatureKey) -> bool:
        """Checks whether the provided (domain, type, name) as the feature_key
        is container in the BasicFeatureStore."""
        return feature_key in self._features

    def __len__(self):
        """Return the number of features."""
        return len(self._features)

    def keys(self):
        """Get the keys of the features.

        Returns
        -------
        List[tuple]
            The keys of the features. The tuples are in `(domain, type_name,
            feat_name)` format.
        """
        return list(self._features.keys())
