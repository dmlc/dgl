"""Minibatch Sampler"""

from typing import Mapping, Optional

from torch.utils.data import default_collate
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from .itemset import *
from ..batch import batch as dgl_batch
from ..heterograph import DGLGraph

__all__ = ["MinibatchSampler"]


def _colloate(batch):
    """Collate batch."""
    data = next(iter(batch))
    if isinstance(data, DGLGraph):
        return dgl_batch(batch)
    elif isinstance(data, Mapping):
        raise NotImplementedError
    return default_collate(batch)


class MinibatchSampler(IterDataPipe):
    """Minibatch Sampler.

    Creates mini-batches of data which could be node/edge IDs, node pairs with
    or without labels, head/tail/negative_tails, DGLGraphs and heterogeneous
    counterparts.

    Note: This class `MinibatchSampler` is not decorated with
    `torchdata.datapipes.functional_datapipe` on purpose. This indicates it
    does not support function-like call. But any iterable datapipes from
    `torchdata` can be further appended.

    Parameters
    ----------
    item_set : ItemSet or DictItemSet
        Data to be sampled for mini-batches.
    batch_size : int
        The size of each batch.
    drop_last : bool
        Option to drop the last batch if it's not full.
    shuffle : bool
        Option to shuffle before sample.

    Examples
    --------
    1. Node/edge IDs.
    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> item_set = gb.ItemSet(torch.arange(0, 10))
    >>> minibatch_sampler = gb.MinibatchSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> list(minibatch_sampler)
    [tensor([1, 2, 5, 7]), tensor([3, 0, 9, 4]), tensor([6, 8])]

    2. Node pairs.
    >>> item_set = gb.ItemSet((torch.arange(0, 10), torch.arange(10, 20)))
    >>> minibatch_sampler = gb.MinibatchSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> list(minibatch_sampler)
    [[tensor([9, 8, 3, 1]), tensor([19, 18, 13, 11])], [tensor([2, 5, 7, 4]),
    tensor([12, 15, 17, 14])], [tensor([0, 6]), tensor([10, 16])]

    3. Node pairs and labels.

    4. Head, tail and negative tails

    5. DGLGraphs.

    6. Append other datapipes.
    >>> item_set = gb.ItemSet(torch.arange(0, 10))
    >>> data_pipe = gb.MinibatchSampler(item_set, 4)
    >>> data_pipe = data_pipe.enumerate()
    >>> list(data_pipe)
    [(0, tensor([0, 1, 2, 3])), (1, tensor([4, 5, 6, 7])), (2, tensor([8, 9]))]
    """

    def __init__(
        self,
        item_set: ItemSet or DictItemSet,
        batch_size: int,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
    ):
        super().__init__()
        self._item_set = item_set
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._shuffle = shuffle

    def __iter__(self):
        data_pipe = IterableWrapper(self._item_set)
        if self._shuffle:
            data_pipe = data_pipe.shuffle()
        data_pipe = data_pipe.batch(
            batch_size=self._batch_size,
            drop_last=self._drop_last,
        )
        data_pipe = data_pipe.collate(collate_fn=_colloate)
        return iter(data_pipe)
