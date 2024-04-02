"""Item Sampler"""

from collections.abc import Mapping
from functools import partial
from typing import Callable, Iterator, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from ..base import dgl_warning

from ..batch import batch as dgl_batch
from ..heterograph import DGLGraph
from .internal import calculate_range
from .itemset import ItemSet, ItemSetDict
from .minibatch import MiniBatch

__all__ = ["ItemSampler", "DistributedItemSampler", "minibatcher_default"]


def minibatcher_default(batch, names):
    """Default minibatcher which maps a list of items to a `MiniBatch` with the
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
    # TODO(#7254): Hacks for original `seed_nodes` and `node_pairs`, which need
    # to be cleaned up later.
    if "node_pairs" in names:
        pos_seeds = init_data["node_pairs"]
        # Build negative graph.
        if "negative_srcs" in names and "negative_dsts" in names:
            neg_srcs = init_data["negative_srcs"]
            neg_dsts = init_data["negative_dsts"]
            (
                init_data["seeds"],
                init_data["labels"],
                init_data["indexes"],
            ) = _construct_seeds(
                pos_seeds, neg_srcs=neg_srcs, neg_dsts=neg_dsts
            )
        elif "negative_srcs" in names:
            neg_srcs = init_data["negative_srcs"]
            (
                init_data["seeds"],
                init_data["labels"],
                init_data["indexes"],
            ) = _construct_seeds(pos_seeds, neg_srcs=neg_srcs)
        elif "negative_dsts" in names:
            neg_dsts = init_data["negative_dsts"]
            (
                init_data["seeds"],
                init_data["labels"],
                init_data["indexes"],
            ) = _construct_seeds(pos_seeds, neg_dsts=neg_dsts)
        else:
            init_data["seeds"] = pos_seeds
    for name, item in init_data.items():
        if not hasattr(minibatch, name):
            dgl_warning(
                f"Unknown item name '{name}' is detected and added into "
                "`MiniBatch`. You probably need to provide a customized "
                "`MiniBatcher`."
            )
        # TODO(#7254): Hacks for original `seed_nodes` and `node_pairs`, which
        # need to be cleaned up later.
        if name == "seed_nodes":
            name = "seeds"
        if name in ("node_pairs", "negative_srcs", "negative_dsts"):
            continue
        setattr(minibatch, name, item)
    return minibatch


class ItemShufflerAndBatcher:
    """A shuffler to shuffle items and create batches.

    This class is used internally by :class:`ItemSampler` to shuffle items and
    create batches. It is not supposed to be used directly. The intention of
    this class is to avoid time-consuming iteration over :class:`ItemSet`. As
    an optimization, it slices from the :class:`ItemSet` via indexing first,
    then shuffle and create batches.

    Parameters
    ----------
    item_set : ItemSet
        Data to be iterated.
    shuffle : bool
        Option to shuffle before batching.
    batch_size : int
        The size of each batch.
    drop_last : bool
        Option to drop the last batch if it's not full.
    buffer_size : int
        The size of the buffer to store items sliced from the :class:`ItemSet`
        or :class:`ItemSetDict`.
    distributed : bool
        Option to apply on :class:`DistributedItemSampler`.
    drop_uneven_inputs : bool
        Option to make sure the numbers of batches for each replica are the
        same. Applies only when `distributed` is True.
    world_size : int
        The number of model replicas that will be created during Distributed
        Data Parallel (DDP) training. It should be the same as the real world
        size, otherwise it could cause errors. Applies only when `distributed`
        is True.
    rank : int
        The rank of the current replica. Applies only when `distributed` is
        True.
    rng : np.random.Generator
        The random number generator to use for shuffling.
    """

    def __init__(
        self,
        item_set: ItemSet,
        shuffle: bool,
        batch_size: int,
        drop_last: bool,
        buffer_size: int,
        distributed: Optional[bool] = False,
        drop_uneven_inputs: Optional[bool] = False,
        world_size: Optional[int] = 1,
        rank: Optional[int] = 0,
        rng: Optional[np.random.Generator] = None,
    ):
        self._item_set = item_set
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._buffer_size = buffer_size
        # Round up the buffer size to the nearest multiple of batch size.
        self._buffer_size = (
            (self._buffer_size + batch_size - 1) // batch_size * batch_size
        )
        self._distributed = distributed
        self._drop_uneven_inputs = drop_uneven_inputs
        self._num_replicas = world_size
        self._rank = rank
        self._rng = rng

    def _collate_batch(self, buffer, indices, offsets=None):
        """Collate a batch from the buffer. For internal use only."""
        if isinstance(buffer, torch.Tensor):
            # For item set that's initialized with integer or single tensor,
            # `buffer` is a tensor.
            return torch.index_select(buffer, dim=0, index=indices)
        elif isinstance(buffer, list) and isinstance(buffer[0], DGLGraph):
            # For item set that's initialized with a list of
            # DGLGraphs, `buffer` is a list of DGLGraphs.
            return dgl_batch([buffer[idx] for idx in indices])
        elif isinstance(buffer, tuple):
            # For item set that's initialized with a tuple of items,
            # `buffer` is a tuple of tensors.
            return tuple(item[indices] for item in buffer)
        elif isinstance(buffer, Mapping):
            # For item set that's initialized with a dict of items,
            # `buffer` is a dict of tensors/lists/tuples.
            keys = list(buffer.keys())
            key_indices = torch.searchsorted(offsets, indices, right=True) - 1
            batch = {}
            for j, key in enumerate(keys):
                mask = (key_indices == j).nonzero().squeeze(1)
                if len(mask) == 0:
                    continue
                batch[key] = self._collate_batch(
                    buffer[key], indices[mask] - offsets[j]
                )
            return batch
        raise TypeError(f"Unsupported buffer type {type(buffer).__name__}.")

    def _calculate_offsets(self, buffer):
        """Calculate offsets for each item in buffer. For internal use only."""
        if not isinstance(buffer, Mapping):
            return None
        offsets = [0]
        for value in buffer.values():
            if isinstance(value, torch.Tensor):
                offsets.append(offsets[-1] + len(value))
            elif isinstance(value, tuple):
                offsets.append(offsets[-1] + len(value[0]))
            else:
                raise TypeError(
                    f"Unsupported buffer type {type(value).__name__}."
                )
        return torch.tensor(offsets)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
        buffer = None
        total = len(self._item_set)
        start_offset, assigned_count, output_count = calculate_range(
            self._distributed,
            total,
            self._num_replicas,
            self._rank,
            num_workers,
            worker_id,
            self._batch_size,
            self._drop_last,
            self._drop_uneven_inputs,
        )
        start = 0
        while start < assigned_count:
            end = min(start + self._buffer_size, assigned_count)
            buffer = self._item_set[start_offset + start : start_offset + end]
            indices = torch.arange(end - start)
            if self._shuffle:
                self._rng.shuffle(indices.numpy())
            offsets = self._calculate_offsets(buffer)
            for i in range(0, len(indices), self._batch_size):
                if output_count <= 0:
                    break
                batch_indices = indices[
                    i : i + min(self._batch_size, output_count)
                ]
                output_count -= self._batch_size
                yield self._collate_batch(buffer, batch_indices, offsets)
            buffer = None
            start = end


class ItemSampler(IterDataPipe):
    """A sampler to iterate over input items and create subsets.

    Input items could be node IDs, node pairs with or without labels, node
    pairs with negative sources/destinations, DGLGraphs and heterogeneous
    counterparts.

    Note: This class `ItemSampler` is not decorated with
    `torchdata.datapipes.functional_datapipe` on purpose. This indicates it
    does not support function-like call. But any iterable datapipes from
    `torchdata` can be further appended.

    Parameters
    ----------
    item_set : Union[ItemSet, ItemSetDict]
        Data to be sampled.
    batch_size : int
        The size of each batch.
    minibatcher : Optional[Callable]
        A callable that takes in a list of items and returns a `MiniBatch`.
    drop_last : bool
        Option to drop the last batch if it's not full.
    shuffle : bool
        Option to shuffle before sample.
    use_indexing : bool
        Option to use indexing to slice items from the item set. This is an
        optimization to avoid time-consuming iteration over the item set. If
        the item set does not support indexing, this option will be disabled
        automatically. If the item set supports indexing but the user wants to
        disable it, this option can be set to False. By default, it is set to
        True.
    buffer_size : int
        The size of the buffer to store items sliced from the :class:`ItemSet`
        or :class:`ItemSetDict`. By default, it is set to -1, which means the
        buffer size will be set as the total number of items in the item set if
        indexing is supported. If indexing is not supported, it is set to 10 *
        batch size. If the item set is too large, it is recommended to set a
        smaller buffer size to avoid out of memory error. As items are shuffled
        within each buffer, a smaller buffer size may incur less randomness and
        such less randomness can further affect the training performance such as
        convergence speed and accuracy. Therefore, it is recommended to set a
        larger buffer size if possible.

    Examples
    --------
    1. Node IDs.

    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> item_set = gb.ItemSet(torch.arange(0, 10), names="seed_nodes")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=tensor([0, 1, 2, 3]), node_pairs=None, labels=None,
        negative_srcs=None, negative_dsts=None, sampled_subgraphs=None,
        input_nodes=None, node_features=None, edge_features=None,
        compacted_node_pairs=None, compacted_negative_srcs=None,
        compacted_negative_dsts=None)

    2. Node pairs.

    >>> item_set = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2),
    ...     names="node_pairs")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs=(tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7])),
        labels=None, negative_srcs=None, negative_dsts=None,
        sampled_subgraphs=None, input_nodes=None, node_features=None,
        edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    3. Node pairs and labels.

    >>> item_set = gb.ItemSet(
    ...     (torch.arange(0, 20).reshape(-1, 2), torch.arange(10, 20)),
    ...     names=("node_pairs", "labels")
    ... )
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs=(tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7])),
        labels=tensor([10, 11, 12, 13]), negative_srcs=None,
        negative_dsts=None, sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    4. Node pairs and negative destinations.

    >>> node_pairs = torch.arange(0, 20).reshape(-1, 2)
    >>> negative_dsts = torch.arange(10, 30).reshape(-1, 2)
    >>> item_set = gb.ItemSet((node_pairs, negative_dsts), names=("node_pairs",
    ...     "negative_dsts"))
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs=(tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7])),
        labels=None, negative_srcs=None,
        negative_dsts=tensor([[10, 11],
        [12, 13],
        [14, 15],
        [16, 17]]), sampled_subgraphs=None, input_nodes=None,
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
    :class:`torchdata.datapipes.iter.Mapper`.

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
    MiniBatch(seed_nodes=None,
        node_pairs={'user:like:item':
            (tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7]))},
        labels=None, negative_srcs=None, negative_dsts=None,
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
    MiniBatch(seed_nodes=None,
        node_pairs={'user:like:item':
            (tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7]))},
        labels={'user:like:item': tensor([0, 1, 2, 3])},
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
    MiniBatch(seed_nodes=None,
        node_pairs={'user:like:item':
            (tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7]))},
        labels=None, negative_srcs=None,
        negative_dsts={'user:like:item': tensor([[10, 11],
        [12, 13],
        [14, 15],
        [16, 17]])}, sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)
    """

    def __init__(
        self,
        item_set: Union[ItemSet, ItemSetDict],
        batch_size: int,
        minibatcher: Optional[Callable] = minibatcher_default,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
        # [TODO][Rui] For now, it's a temporary knob to disable indexing. In
        # the future, we will enable indexing for all the item sets.
        use_indexing: Optional[bool] = True,
        buffer_size: Optional[int] = -1,
    ) -> None:
        super().__init__()
        self._names = item_set.names
        # Check if the item set supports indexing.
        indexable = True
        try:
            item_set[0]
        except TypeError:
            indexable = False
        self._use_indexing = use_indexing and indexable
        self._item_set = (
            item_set if self._use_indexing else IterableWrapper(item_set)
        )
        if buffer_size == -1:
            if indexable:
                # Set the buffer size to the total number of items in the item
                # set if indexing is supported and the buffer size is not
                # specified.
                buffer_size = len(self._item_set)
            else:
                # Set the buffer size to 10 * batch size if indexing is not
                # supported and the buffer size is not specified.
                buffer_size = 10 * batch_size
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._minibatcher = minibatcher
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._distributed = False
        self._drop_uneven_inputs = False
        self._world_size = None
        self._rank = None
        self._rng = np.random.default_rng()

    def _organize_items(self, data_pipe) -> None:
        # Shuffle before batch.
        if self._shuffle:
            data_pipe = data_pipe.shuffle(buffer_size=self._buffer_size)

        # Batch.
        data_pipe = data_pipe.batch(
            batch_size=self._batch_size,
            drop_last=self._drop_last,
        )

        return data_pipe

    @staticmethod
    def _collate(batch):
        """Collate items into a batch. For internal use only."""
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

    def __iter__(self) -> Iterator:
        if self._use_indexing:
            seed = self._rng.integers(0, np.iinfo(np.int32).max)
            data_pipe = IterableWrapper(
                ItemShufflerAndBatcher(
                    self._item_set,
                    self._shuffle,
                    self._batch_size,
                    self._drop_last,
                    self._buffer_size,
                    distributed=self._distributed,
                    drop_uneven_inputs=self._drop_uneven_inputs,
                    world_size=self._world_size,
                    rank=self._rank,
                    rng=np.random.default_rng(seed),
                )
            )
        else:
            # Organize items.
            data_pipe = self._organize_items(self._item_set)

            # Collate.
            data_pipe = data_pipe.collate(collate_fn=self._collate)

        # Map to minibatch.
        data_pipe = data_pipe.map(partial(self._minibatcher, names=self._names))

        return iter(data_pipe)


class DistributedItemSampler(ItemSampler):
    """A sampler to iterate over input items and create subsets distributedly.

    This sampler creates a distributed subset of items from the given data set,
    which can be used for training with PyTorch's Distributed Data Parallel
    (DDP). The items can be node IDs, node pairs with or without labels, node
    pairs with negative sources/destinations, DGLGraphs, or heterogeneous
    counterparts. The original item set is split such that each replica
    (process) receives an exclusive subset.

    Note: The items will be first split onto each replica, then get shuffled
    (if needed) and batched. Therefore, each replica will always get a same set
    of items.

    Note: This class `DistributedItemSampler` is not decorated with
    `torchdata.datapipes.functional_datapipe` on purpose. This indicates it
    does not support function-like call. But any iterable datapipes from
    `torchdata` can be further appended.

    Parameters
    ----------
    item_set : Union[ItemSet, ItemSetDict]
        Data to be sampled.
    batch_size : int
        The size of each batch.
    minibatcher : Optional[Callable]
        A callable that takes in a list of items and returns a `MiniBatch`.
    drop_last : bool
        Option to drop the last batch if it's not full.
    shuffle : bool
        Option to shuffle before sample.
    num_replicas: int
        The number of model replicas that will be created during Distributed
        Data Parallel (DDP) training. It should be the same as the real world
        size, otherwise it could cause errors. By default, it is retrieved from
        the current distributed group.
    drop_uneven_inputs : bool
        Option to make sure the numbers of batches for each replica are the
        same. If some of the replicas have more batches than the others, the
        redundant batches of those replicas will be dropped. If the drop_last
        parameter is also set to True, the last batch will be dropped before the
        redundant batches are dropped.
        Note: When using Distributed Data Parallel (DDP) training, the program
        may hang or error if the a replica has fewer inputs. It is recommended
        to use the Join Context Manager provided by PyTorch to solve this
        problem. Please refer to
        https://pytorch.org/tutorials/advanced/generic_join.html. However, this
        option can be used if the Join Context Manager is not helpful for any
        reason.
    buffer_size : int
        The size of the buffer to store items sliced from the :class:`ItemSet`
        or :class:`ItemSetDict`. By default, it is set to -1, which means the
        buffer size will be set as the total number of items in the item set.
        If the item set is too large, it is recommended to set a smaller buffer
        size to avoid out of memory error. As items are shuffled within each
        buffer, a smaller buffer size may incur less randomness and such less
        randomness can further affect the training performance such as
        convergence speed and accuracy. Therefore, it is recommended to set a
        larger buffer size if possible.

    Examples
    --------
    0. Preparation: DistributedItemSampler needs multi-processing environment to
    work. You need to spawn subprocesses and initialize processing group before
    executing following examples. Due to randomness, the output is not always
    the same as listed below.

    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> item_set = gb.ItemSet(torch.arange(15))
    >>> num_replicas = 4
    >>> batch_size = 2
    >>> mp.spawn(...)

    1. shuffle = False, drop_last = False, drop_uneven_inputs = False.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=False,
    >>>     drop_uneven_inputs=False
    >>> )
    >>> data_loader = gb.DataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 1]), tensor([2, 3])]
    Replica#1: [tensor([4, 5]), tensor([6, 7])]
    Replica#2: [tensor([8, 9]), tensor([10, 11])]
    Replica#3: [tensor([12, 13]), tensor([14])]

    2. shuffle = False, drop_last = True, drop_uneven_inputs = False.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=True,
    >>>     drop_uneven_inputs=False
    >>> )
    >>> data_loader = gb.DataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 1]), tensor([2, 3])]
    Replica#1: [tensor([4, 5]), tensor([6, 7])]
    Replica#2: [tensor([8, 9]), tensor([10, 11])]
    Replica#3: [tensor([12, 13])]

    3. shuffle = False, drop_last = False, drop_uneven_inputs = True.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=False,
    >>>     drop_uneven_inputs=True
    >>> )
    >>> data_loader = gb.DataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 1]), tensor([2, 3])]
    Replica#1: [tensor([4, 5]), tensor([6, 7])]
    Replica#2: [tensor([8, 9]), tensor([10, 11])]
    Replica#3: [tensor([12, 13]), tensor([14])]

    4. shuffle = False, drop_last = True, drop_uneven_inputs = True.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=True,
    >>>     drop_uneven_inputs=True
    >>> )
    >>> data_loader = gb.DataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 1])]
    Replica#1: [tensor([4, 5])]
    Replica#2: [tensor([8, 9])]
    Replica#3: [tensor([12, 13])]

    5. shuffle = True, drop_last = True, drop_uneven_inputs = False.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=True, drop_last=True,
    >>>     drop_uneven_inputs=False
    >>> )
    >>> data_loader = gb.DataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    (One possible output:)
    Replica#0: [tensor([3, 2]), tensor([0, 1])]
    Replica#1: [tensor([6, 5]), tensor([7, 4])]
    Replica#2: [tensor([8, 10])]
    Replica#3: [tensor([14, 12])]

    6. shuffle = True, drop_last = True, drop_uneven_inputs = True.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=True, drop_last=True,
    >>>     drop_uneven_inputs=True
    >>> )
    >>> data_loader = gb.DataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    (One possible output:)
    Replica#0: [tensor([1, 3])]
    Replica#1: [tensor([7, 5])]
    Replica#2: [tensor([11, 9])]
    Replica#3: [tensor([13, 14])]
    """

    def __init__(
        self,
        item_set: Union[ItemSet, ItemSetDict],
        batch_size: int,
        minibatcher: Optional[Callable] = minibatcher_default,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
        drop_uneven_inputs: Optional[bool] = False,
        buffer_size: Optional[int] = -1,
    ) -> None:
        super().__init__(
            item_set,
            batch_size,
            minibatcher,
            drop_last,
            shuffle,
            use_indexing=True,
            buffer_size=buffer_size,
        )
        self._distributed = True
        self._drop_uneven_inputs = drop_uneven_inputs
        if not dist.is_available():
            raise RuntimeError(
                "Distributed item sampler requires distributed package."
            )
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()


def _construct_seeds(pos_seeds, neg_srcs=None, neg_dsts=None):
    # For homogeneous graph.
    if isinstance(pos_seeds, torch.Tensor):
        negative_ratio = neg_srcs.size(1) if neg_srcs else neg_dsts.size(1)
        neg_srcs = (
            neg_srcs
            if neg_srcs is not None
            else pos_seeds[:, 0].repeat_interleave(negative_ratio)
        ).view(-1)
        neg_dsts = (
            neg_dsts
            if neg_dsts is not None
            else pos_seeds[:, 1].repeat_interleave(negative_ratio)
        ).view(-1)
        neg_seeds = torch.cat((neg_srcs, neg_dsts)).view(2, -1).T
        seeds = torch.cat((pos_seeds, neg_seeds))
        pos_seeds_num = pos_seeds.size(0)
        labels = torch.empty(seeds.size(0), device=pos_seeds.device)
        labels[:pos_seeds_num] = 1
        labels[pos_seeds_num:] = 0
        pos_indexes = torch.arange(
            0,
            pos_seeds_num,
            device=pos_seeds.device,
        )
        neg_indexes = pos_indexes.repeat_interleave(negative_ratio)
        indexes = torch.cat((pos_indexes, neg_indexes))
    # For heterogeneous graph.
    else:
        negative_ratio = (
            list(neg_srcs.values())[0].size(1)
            if neg_srcs
            else list(neg_dsts.values())[0].size(1)
        )
        seeds = {}
        labels = {}
        indexes = {}
        for etype in pos_seeds:
            neg_src = (
                neg_srcs[etype]
                if neg_srcs is not None
                else pos_seeds[etype][:, 0].repeat_interleave(negative_ratio)
            ).view(-1)
            neg_dst = (
                neg_dsts[etype]
                if neg_dsts is not None
                else pos_seeds[etype][:, 1].repeat_interleave(negative_ratio)
            ).view(-1)
            seeds[etype] = torch.cat(
                (
                    pos_seeds[etype],
                    torch.cat(
                        (
                            neg_src,
                            neg_dst,
                        )
                    )
                    .view(2, -1)
                    .T,
                )
            )
            pos_seeds_num = pos_seeds[etype].size(0)
            labels[etype] = torch.empty(
                seeds[etype].size(0), device=pos_seeds[etype].device
            )
            labels[etype][:pos_seeds_num] = 1
            labels[etype][pos_seeds_num:] = 0
            pos_indexes = torch.arange(
                0,
                pos_seeds_num,
                device=pos_seeds[etype].device,
            )
            neg_indexes = pos_indexes.repeat_interleave(negative_ratio)
            indexes[etype] = torch.cat((pos_indexes, neg_indexes))
    return seeds, labels, indexes
