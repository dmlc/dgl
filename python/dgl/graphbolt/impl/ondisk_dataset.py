"""GraphBolt OnDiskDataset."""

from typing import List

from ..dataset import Dataset
from ..feature_store import FeatureStore
from ..itemset import ItemSet, ItemSetDict
from ..utils import read_data, tensor_to_tuple
from .ondisk_metadata import OnDiskMetaData, OnDiskTVTSet

__all__ = ["OnDiskDataset"]


class OnDiskDataset(Dataset):
    """An on-disk dataset.

    An on-disk dataset is a dataset which reads graph topology, feature data
    and TVT set from disk. Due to limited resources, the data which are too
    large to fit into RAM will remain on disk while others reside in RAM once
    ``OnDiskDataset`` is initialized. This behavior could be controled by user
    via ``in_memory`` field in YAML file.

    A full example of YAML file is as follows:

    .. code-block:: yaml

        train_sets:
          - - type_name: paper # could be null for homogeneous graph.
              format: numpy
              in_memory: true # If not specified, default to true.
              path: set/paper-train.npy
        validation_sets:
          - - type_name: paper
              format: numpy
              in_memory: true
              path: set/paper-validation.npy
        test_sets:
          - - type_name: paper
              format: numpy
              in_memory: true
              path: set/paper-test.npy

    Parameters
    ----------
    path: str
        The YAML file path.
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            self._meta = OnDiskMetaData.parse_raw(f.read(), proto="yaml")
        self._train_sets = self._init_tvt_sets(self._meta.train_sets)
        self._validation_sets = self._init_tvt_sets(self._meta.validation_sets)
        self._test_sets = self._init_tvt_sets(self._meta.test_sets)

    def train_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the training set."""
        return self._train_sets

    def validation_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the validation set."""
        return self._validation_sets

    def test_sets(self) -> List[ItemSet] or List[ItemSetDict]:
        """Return the test set."""
        return self._test_sets

    def graph(self) -> object:
        """Return the graph."""
        raise NotImplementedError

    def feature(self) -> FeatureStore:
        """Return the feature."""
        raise NotImplementedError

    def _init_tvt_sets(
        self, tvt_sets: List[List[OnDiskTVTSet]]
    ) -> List[ItemSet] or List[ItemSetDict]:
        """Initialize the TVT sets."""
        if (tvt_sets is None) or (len(tvt_sets) == 0):
            return None
        ret = []
        for tvt_set in tvt_sets:
            if (tvt_set is None) or (len(tvt_set) == 0):
                ret.append(None)
            if tvt_set[0].type_name is None:
                assert (
                    len(tvt_set) == 1
                ), "Only one TVT set is allowed if type_name is not specified."
                data = read_data(
                    tvt_set[0].path, tvt_set[0].format, tvt_set[0].in_memory
                )
                ret.append(ItemSet(tensor_to_tuple(data)))
            else:
                data = {}
                for tvt in tvt_set:
                    data[tvt.type_name] = ItemSet(
                        tensor_to_tuple(
                            read_data(tvt.path, tvt.format, tvt.in_memory)
                        )
                    )
                ret.append(ItemSetDict(data))
        return ret
