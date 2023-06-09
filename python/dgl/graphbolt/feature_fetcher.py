from ..base import NID
from .feature_store import FeatureStore


class FeatureFetcher(object):
    def __init__(self, feature_store: FeatureStore):
        self._feature_store = feature_store

    def fetch(self, blocks):
        return self._feature_store.get_items(blocks[0].srcdata[NID])
