"""Item Sampler"""

from collections.abc import Mapping
from functools import partial
from typing import Callable, Iterator, Optional

from torch.utils.data import default_collate
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from ..base import dgl_warning

from ..batch import batch as dgl_batch
from ..heterograph import DGLGraph
from .itemset import ItemSet, ItemSetDict
from .minibatch import MiniBatch

__all__ = ["ItemSampler", "minibatcher_default"]


def minibatcher_default(batch, names):
    """Default minibatcher.

    The default minibatcher maps a list of items to a `MiniBatch` with the
    same names as the items. The names of items are supposed to be provided
    and align with the data attributes of `MiniBatch`. If any unknown item name
    is provided, exception will be raised. If the names of items are not
    provided, the item list is returned as is and a warning will be raised.

    Parameters
    ----------
    batch : list
        List of items.
    names : Tuple[str] or None
        Names of items in `batch` with same length. The order should align
        with `batch`.

    Returns
    -------
    MiniBatch
        A minibatch.
    """
    if names is None:
        dgl_warning(
            "Failed to map item list to `MiniBatch` as the names of items are "
            "not provided. Please provide a customized `MiniBatcher`. "
            "The item list is returned as is."
        )
        return batch
    if len(names) == 1:
        # Handle the case of single item: batch = tensor([0, 1, 2, 3]), names =
        # ("seed_nodes",) as `zip(batch, names)` will iterate over the tensor
        # instead of the batch.
        init_data = {names[0]: batch}
    else:
        if isinstance(batch, Mapping):
            init_data = {
                name: {k: v[i] for k, v in batch.items()}
                for i, name in enumerate(names)
            }
        else:
            init_data = {name: item for item, name in zip(batch, names)}
    minibatch = MiniBatch()
    for name, item in init_data.items():
        if not hasattr(minibatch, name):
            dgl_warning(
                f"Unknown item name '{name}' is detected and added into "
                "`MiniBatch`. You probably need to provide a customized "
                "`MiniBatcher`."
            )
        setattr(minibatch, name, item)
    return minibatch


class ItemSampler(IterDataPipe):
    """Item Sampler.

    Creates item subset of data which could be node IDs, node pairs with or
    without labels, node pairs with negative sources/destinations, DGLGraphs
    and heterogeneous counterparts.

    Note: This class `ItemSampler` is not decorated with
    `torchdata.datapipes.functional_datapipe` on purpose. This indicates it
    does not support function-like call. But any iterable datapipes from
    `torchdata` can be further appended.

    Parameters
    ----------
    item_set : ItemSet or ItemSetDict
        Data to be sampled.
    batch_size : int
        The size of each batch.
    minibatcher : Optional[Callable]
        A callable that takes in a list of items and returns a `MiniBatch`.
    drop_last : bool
        Option to drop the last batch if it's not full.
    shuffle : bool
        Option to shuffle before sample.

    Examples
    --------
    1. Node IDs.
    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> item_set = gb.ItemSet(torch.arange(0, 10), names="seed_nodes")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=tensor([9, 0, 7, 2]), node_pairs=None, labels=None,
        negative_srcs=None, negative_dsts=None, sampled_subgraphs=None,
        input_nodes=None, node_features=None, edge_features=None,
        compacted_node_pairs=None, compacted_negative_srcs=None,
        compacted_negative_dsts=None)

    2. Node pairs.
    >>> item_set = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2),
    ...     names="node_pairs")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None, node_pairs=tensor([[16, 17],
        [ 4,  5],
        [ 6,  7],
        [10, 11]]), labels=None, negative_srcs=None, negative_dsts=None,
        sampled_subgraphs=None, input_nodes=None, node_features=None,
        edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    3. Node pairs and labels.
    >>> item_set = gb.ItemSet(
    ...     (torch.arange(0, 20).reshape(-1, 2), torch.arange(10, 15)),
    ...     names=("node_pairs", "labels")
    ... )
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None, node_pairs=tensor([[8, 9],
        [4, 5],
        [0, 1],
        [6, 7]]), labels=tensor([14, 12, 10, 13]), negative_srcs=None,
        negative_dsts=None, sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    4. Node pairs and negative destinations.
    >>> node_pairs = torch.arange(0, 20).reshape(-1, 2)
    >>> negative_dsts = torch.arange(10, 30).reshape(-1, 2)
    >>> item_set = gb.ItemSet((node_pairs, negative_dsts), names=("node_pairs",
    ...     "negative_dsts"))
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None, node_pairs=tensor([[10, 11],
        [ 6,  7],
        [ 2,  3],
        [ 8,  9]]), labels=None, negative_srcs=None,
        negative_dsts=tensor([[20, 21],
        [16, 17],
        [12, 13],
        [18, 19]]), sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    5. DGLGraphs.
    >>> import dgl
    >>> graphs = [ dgl.rand_graph(10, 20) for _ in range(5) ]
    >>> item_set = gb.ItemSet(graphs)
    >>> item_sampler = gb.ItemSampler(item_set, 3)
    >>> list(item_sampler)
    [Graph(num_nodes=30, num_edges=60,
      ndata_schemes={}
      edata_schemes={}),
     Graph(num_nodes=20, num_edges=40,
      ndata_schemes={}
      edata_schemes={})]

    6. Further process batches with other datapipes such as
    `torchdata.datapipes.iter.Mapper`.
    >>> item_set = gb.ItemSet(torch.arange(0, 10))
    >>> data_pipe = gb.ItemSampler(item_set, 4)
    >>> def add_one(batch):
    ...     return batch + 1
    >>> data_pipe = data_pipe.map(add_one)
    >>> list(data_pipe)
    [tensor([1, 2, 3, 4]), tensor([5, 6, 7, 8]), tensor([ 9, 10])]

    7. Heterogeneous node IDs.
    >>> ids = {
    ...     "user": gb.ItemSet(torch.arange(0, 5), names="seed_nodes"),
    ...     "item": gb.ItemSet(torch.arange(0, 6), names="seed_nodes"),
    ... }
    >>> item_set = gb.ItemSetDict(ids)
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes={'user': tensor([0, 1, 2, 3])}, node_pairs=None,
    labels=None, negative_srcs=None, negative_dsts=None, sampled_subgraphs=None,
    input_nodes=None, node_features=None, edge_features=None,
    compacted_node_pairs=None, compacted_negative_srcs=None,
    compacted_negative_dsts=None)

    8. Heterogeneous node pairs.
    >>> node_pairs_like = torch.arange(0, 10).reshape(-1, 2)
    >>> node_pairs_follow = torch.arange(10, 20).reshape(-1, 2)
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet(
    ...         node_pairs_like, names="node_pairs"),
    ...     "user:follow:user": gb.ItemSet(
    ...         node_pairs_follow, names="node_pairs"),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None, node_pairs={'user:like:item': tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])}, labels=None, negative_srcs=None, negative_dsts=None,
        sampled_subgraphs=None, input_nodes=None, node_features=None,
        edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    9. Heterogeneous node pairs and labels.
    >>> node_pairs_like = torch.arange(0, 10).reshape(-1, 2)
    >>> labels_like = torch.arange(0, 10)
    >>> node_pairs_follow = torch.arange(10, 20).reshape(-1, 2)
    >>> labels_follow = torch.arange(10, 20)
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet((node_pairs_like, labels_like),
    ...         names=("node_pairs", "labels")),
    ...     "user:follow:user": gb.ItemSet((node_pairs_follow, labels_follow),
    ...         names=("node_pairs", "labels")),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None, node_pairs={'user:like:item': tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])}, labels={'user:like:item': tensor([0, 1, 2, 3])},
        negative_srcs=None, negative_dsts=None, sampled_subgraphs=None,
        input_nodes=None, node_features=None, edge_features=None,
        compacted_node_pairs=None, compacted_negative_srcs=None,
        compacted_negative_dsts=None)

    10. Heterogeneous node pairs and negative destinations.
    >>> node_pairs_like = torch.arange(0, 10).reshape(-1, 2)
    >>> negative_dsts_like = torch.arange(10, 20).reshape(-1, 2)
    >>> node_pairs_follow = torch.arange(20, 30).reshape(-1, 2)
    >>> negative_dsts_follow = torch.arange(30, 40).reshape(-1, 2)
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet((node_pairs_like, negative_dsts_like),
    ...         names=("node_pairs", "negative_dsts")),
    ...     "user:follow:user": gb.ItemSet((node_pairs_follow,
    ...         negative_dsts_follow), names=("node_pairs", "negative_dsts")),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None, node_pairs={'user:like:item': tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7]])}, labels=None, negative_srcs=None,
        negative_dsts={'user:like:item': tensor([[10, 11],
        [12, 13],
        [14, 15],
        [16, 17]])}, sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)
    """

    def __init__(
        self,
        item_set: ItemSet or ItemSetDict,
        batch_size: int,
        minibatcher: Optional[Callable] = minibatcher_default,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self._item_set = item_set
        self._batch_size = batch_size
        self._minibatcher = minibatcher
        self._drop_last = drop_last
        self._shuffle = shuffle

    def __iter__(self) -> Iterator:
        data_pipe = IterableWrapper(self._item_set)
        # Shuffle before batch.
        if self._shuffle:
            # `torchdata.datapipes.iter.Shuffler` works with stream too.
            # To ensure randomness, make sure the buffer size is at least 10
            # times the batch size.
            buffer_size = max(10000, 10 * self._batch_size)
            data_pipe = data_pipe.shuffle(buffer_size=buffer_size)

        # Batch.
        data_pipe = data_pipe.batch(
            batch_size=self._batch_size,
            drop_last=self._drop_last,
        )

        # Collate.
        def _collate(batch):
            data = next(iter(batch))
            if isinstance(data, DGLGraph):
                return dgl_batch(batch)
            elif isinstance(data, Mapping):
                assert len(data) == 1, "Only one type of data is allowed."
                # Collect all the keys.
                keys = {key for item in batch for key in item.keys()}
                # Collate each key.
                return {
                    key: default_collate(
                        [item[key] for item in batch if key in item]
                    )
                    for key in keys
                }
            return default_collate(batch)

        data_pipe = data_pipe.collate(collate_fn=partial(_collate))

        # Map to minibatch.
        data_pipe = data_pipe.map(
            partial(self._minibatcher, names=self._item_set.names)
        )

        return iter(data_pipe)
