import torch


class FeatureStore(object):
    def get_items(self, ids):
        raise NotImplementedError

    def set_items(self, ids, vals):
        raise NotImplementedError


class InMemoryFeatureStore(object):
    def __init__(self, tensor: torch.Tensor):
        super(InMemoryFeatureStore, self).__init__()
        self._tensor = tensor

    def get_items(self, ids):
        return self._tensor[ids]

    def set_items(self, ids, vals):
        self._tensor[ids] = vals
