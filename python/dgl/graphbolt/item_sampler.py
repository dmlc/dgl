"""Item Sampler"""

from collections.abc import Mapping
from typing import Callable, Iterator, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterDataPipe

from .internal import calculate_range
from .internal_utils import gb_warning
from .itemset import HeteroItemSet, ItemSet
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
        gb_warning(
            "Failed to map item list to `MiniBatch` as the names of items are "
            "not provided. Please provide a customized `MiniBatcher`. "
            "The item list is returned as is."
        )
        return batch
    if len(names) == 1:
        # Handle the case of single item: batch = tensor([0, 1, 2, 3]), names =
        # ("seeds",) as `zip(batch, names)` will iterate over the tensor
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
            gb_warning(
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


class ItemSampler(IterDataPipe):
    """A sampler to iterate over input items and create minibatches.

    Input items could be node IDs, node pairs with or without labels, node
    pairs with negative sources/destinations.

    Note: This class `ItemSampler` is not decorated with
    `torch.utils.data.functional_datapipe` on purpose. This indicates it
    does not support function-like call. But any iterable datapipes from
    `torch.utils.data.datapipes` can be further appended.

    Parameters
    ----------
    item_set : Union[ItemSet, HeteroItemSet]
        Data to be sampled.
    batch_size : int
        The size of each batch.
    minibatcher : Optional[Callable]
        A callable that takes in a list of items and returns a `MiniBatch`.
    drop_last : bool
        Option to drop the last batch if it's not full.
    shuffle : bool
        Option to shuffle before sample.
    seed: int
        The seed for reproducible stochastic shuffling. If None, a random seed
        will be generated.

    Examples
    --------
    1. Node IDs.

    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> item_set = gb.ItemSet(torch.arange(0, 10), names="seeds")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seeds=tensor([0, 1, 2, 3]), sampled_subgraphs=None,
        node_features=None, labels=None, input_nodes=None,
        indexes=None, edge_features=None, compacted_seeds=None,
        blocks=None,)

    2. Node pairs.

    >>> item_set = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2),
    ...     names="seeds")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seeds=tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
        sampled_subgraphs=None, node_features=None, labels=None,
        input_nodes=None, indexes=None, edge_features=None,
        compacted_seeds=None, blocks=None,)

    3. Node pairs and labels.

    >>> item_set = gb.ItemSet(
    ...     (torch.arange(0, 20).reshape(-1, 2), torch.arange(10, 20)),
    ...     names=("seeds", "labels")
    ... )
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seeds=tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
        sampled_subgraphs=None, node_features=None,
        labels=tensor([10, 11, 12, 13]), input_nodes=None,
        indexes=None, edge_features=None, compacted_seeds=None,
        blocks=None,)

    4. Node pairs, labels and indexes.

    >>> seeds = torch.arange(0, 20).reshape(-1, 2)
    >>> labels = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> indexes = torch.tensor([0, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    >>> item_set = gb.ItemSet((seeds, labels, indexes), names=("seeds",
    ...     "labels", "indexes"))
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seeds=tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
        sampled_subgraphs=None, node_features=None,
        labels=tensor([1, 1, 0, 0]), input_nodes=None,
        indexes=tensor([0, 1, 0, 0]), edge_features=None,
        compacted_seeds=None, blocks=None,)

    5. Further process batches with other datapipes such as
    :class:`torch.utils.data.datapipes.iter.Mapper`.

    >>> item_set = gb.ItemSet(torch.arange(0, 10))
    >>> data_pipe = gb.ItemSampler(item_set, 4)
    >>> def add_one(batch):
    ...     return batch + 1
    >>> data_pipe = data_pipe.map(add_one)
    >>> list(data_pipe)
    [tensor([1, 2, 3, 4]), tensor([5, 6, 7, 8]), tensor([ 9, 10])]

    6. Heterogeneous node IDs.

    >>> ids = {
    ...     "user": gb.ItemSet(torch.arange(0, 5), names="seeds"),
    ...     "item": gb.ItemSet(torch.arange(0, 6), names="seeds"),
    ... }
    >>> item_set = gb.HeteroItemSet(ids)
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seeds={'user': tensor([0, 1, 2, 3])}, sampled_subgraphs=None,
        node_features=None, labels=None, input_nodes=None, indexes=None,
        edge_features=None, compacted_seeds=None, blocks=None,)

    7. Heterogeneous node pairs.

    >>> seeds_like = torch.arange(0, 10).reshape(-1, 2)
    >>> seeds_follow = torch.arange(10, 20).reshape(-1, 2)
    >>> item_set = gb.HeteroItemSet({
    ...     "user:like:item": gb.ItemSet(
    ...         seeds_like, names="seeds"),
    ...     "user:follow:user": gb.ItemSet(
    ...         seeds_follow, names="seeds"),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seeds={'user:like:item':
        tensor([[0, 1], [2, 3], [4, 5], [6, 7]])}, sampled_subgraphs=None,
        node_features=None, labels=None, input_nodes=None, indexes=None,
        edge_features=None, compacted_seeds=None, blocks=None,)

    8. Heterogeneous node pairs and labels.

    >>> seeds_like = torch.arange(0, 10).reshape(-1, 2)
    >>> labels_like = torch.arange(0, 5)
    >>> seeds_follow = torch.arange(10, 20).reshape(-1, 2)
    >>> labels_follow = torch.arange(5, 10)
    >>> item_set = gb.HeteroItemSet({
    ...     "user:like:item": gb.ItemSet((seeds_like, labels_like),
    ...         names=("seeds", "labels")),
    ...     "user:follow:user": gb.ItemSet((seeds_follow, labels_follow),
    ...         names=("seeds", "labels")),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seeds={'user:like:item':
        tensor([[0, 1], [2, 3], [4, 5], [6, 7]])}, sampled_subgraphs=None,
        node_features=None, labels={'user:like:item': tensor([0, 1, 2, 3])},
        input_nodes=None, indexes=None, edge_features=None,
        compacted_seeds=None, blocks=None,)

    9. Heterogeneous node pairs, labels and indexes.

    >>> seeds_like = torch.arange(0, 10).reshape(-1, 2)
    >>> labels_like = torch.tensor([1, 1, 0, 0, 0])
    >>> indexes_like = torch.tensor([0, 1, 0, 0, 1])
    >>> seeds_follow = torch.arange(20, 30).reshape(-1, 2)
    >>> labels_follow = torch.tensor([1, 1, 0, 0, 0])
    >>> indexes_follow = torch.tensor([0, 1, 0, 0, 1])
    >>> item_set = gb.HeteroItemSet({
    ...     "user:like:item": gb.ItemSet((seeds_like, labels_like,
    ...         indexes_like), names=("seeds", "labels", "indexes")),
    ...     "user:follow:user": gb.ItemSet((seeds_follow,labels_follow,
    ...         indexes_follow), names=("seeds", "labels", "indexes")),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seeds={'user:like:item':
        tensor([[0, 1], [2, 3], [4, 5], [6, 7]])}, sampled_subgraphs=None,
        node_features=None, labels={'user:like:item': tensor([1, 1, 0, 0])},
        input_nodes=None, indexes={'user:like:item': tensor([0, 1, 0, 0])},
        edge_features=None, compacted_seeds=None, blocks=None,)
    """

    def __init__(
        self,
        item_set: Union[ItemSet, HeteroItemSet],
        batch_size: int,
        minibatcher: Optional[Callable] = minibatcher_default,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._item_set = item_set
        self._names = item_set.names
        self._batch_size = batch_size
        self._minibatcher = minibatcher
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._distributed = False
        self._drop_uneven_inputs = False
        self._world_size = None
        self._rank = None
        # For the sake of reproducibility, the seed should be allowed to be
        # manually set by the user.
        if seed is None:
            self._seed = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            self._seed = seed
        # The attribute `self._epoch` is added to make shuffling work properly
        # across multiple epochs. Otherwise, the same ordering will always be
        # used in every epoch.
        self._epoch = 0

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0
        total = len(self._item_set)
        start_offset, assigned_count, output_count = calculate_range(
            self._distributed,
            total,
            self._world_size,
            self._rank,
            num_workers,
            worker_id,
            self._batch_size,
            self._drop_last,
            self._drop_uneven_inputs,
        )
        if self._shuffle:
            g = torch.Generator().manual_seed(self._seed + self._epoch)
            permutation = torch.randperm(total, generator=g)
            indices = permutation[start_offset : start_offset + assigned_count]
        else:
            indices = torch.arange(start_offset, start_offset + assigned_count)
        for i in range(0, assigned_count, self._batch_size):
            if output_count <= 0:
                break
            yield self._minibatcher(
                self._item_set[
                    indices[i : i + min(self._batch_size, output_count)]
                ],
                self._names,
            )
            output_count -= self._batch_size

        self._epoch += 1


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
    `torch.utils.data.functional_datapipe` on purpose. This indicates it
    does not support function-like call. But any iterable datapipes from
    `torch.utils.data.datapipes` can be further appended.

    Parameters
    ----------
    item_set : Union[ItemSet, HeteroItemSet]
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
    seed: int
        The seed for reproducible stochastic shuffling. If None, a random seed
        will be generated.

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
        item_set: Union[ItemSet, HeteroItemSet],
        batch_size: int,
        minibatcher: Optional[Callable] = minibatcher_default,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
        drop_uneven_inputs: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            item_set,
            batch_size,
            minibatcher,
            drop_last,
            shuffle,
            seed,
        )
        self._distributed = True
        self._drop_uneven_inputs = drop_uneven_inputs
        if not dist.is_available():
            raise RuntimeError(
                "Distributed item sampler requires distributed package."
            )
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        if self._world_size > 1:
            # For the sake of reproducibility, the seed should be allowed to be
            # manually set by the user.
            self._align_seeds(src=0, seed=seed)

    def _align_seeds(
        self, src: Optional[int] = 0, seed: Optional[int] = None
    ) -> None:
        """Aligns seeds across distributed processes.

        This method synchronizes seeds across distributed processes, ensuring
        consistent randomness.

        Parameters
        ----------
        src: int, optional
            The source process rank. Defaults to 0.
        seed: int, optional
            The seed value to synchronize. If None, a random seed will be
            generated. Defaults to None.
        """
        device = (
            torch.cuda.current_device()
            if torch.cuda.is_available() and dist.get_backend() == "nccl"
            else "cpu"
        )
        if seed is None:
            seed = np.random.randint(0, np.iinfo(np.int32).max)
        if self._rank == src:
            seed_tensor = torch.tensor(seed, dtype=torch.int32, device=device)
        else:
            seed_tensor = torch.empty([], dtype=torch.int32, device=device)
        dist.broadcast(seed_tensor, src=src)
        self._seed = seed_tensor.item()


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
