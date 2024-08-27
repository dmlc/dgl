"""GraphBolt Itemset."""

import textwrap
from typing import Dict, Iterable, Tuple, Union

import torch

from .internal_utils import gb_warning

__all__ = ["ItemSet", "HeteroItemSet", "ItemSetDict"]


def is_scalar(x):
    """Checks if the input is a scalar."""
    return (
        len(x.shape) == 0 if isinstance(x, torch.Tensor) else isinstance(x, int)
    )


class ItemSet:
    r"""A wrapper of a tensor or tuple of tensors.

    Parameters
    ----------
    items: Union[int, torch.Tensor, Tuple[torch.Tensor]]
        The tensors to be wrapped.
        - If it is a single scalar (an integer or a tensor that holds a single
          value), the item would be considered as a range_tensor created by
          `torch.arange`.
        - If it is a multi-dimensional tensor, the indexing will be performed
          along the first dimension.
        - If it is a tuple, each item in the tuple must be a tensor.

    names: Union[str, Tuple[str]], optional
        The names of the items. If it is a tuple, each name must corresponds to
        an item in the `items` parameter. The naming is arbitrary, but in
        general practice, the names should be chosen from ['labels', 'seeds',
        'indexes'] to align with the attributes of class
        `dgl.graphbolt.MiniBatch`.

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb

    1. Integer: number of nodes.

    >>> num = 10
    >>> item_set = gb.ItemSet(num, names="seeds")
    >>> list(item_set)
    [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4), tensor(5),
     tensor(6), tensor(7), tensor(8), tensor(9)]
    >>> item_set[:]
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> item_set.names
    ('seeds',)

    2. Torch scalar: number of nodes. Customizable dtype compared to Integer.

    >>> num = torch.tensor(10, dtype=torch.int32)
    >>> item_set = gb.ItemSet(num, names="seeds")
    >>> list(item_set)
    [tensor(0, dtype=torch.int32), tensor(1, dtype=torch.int32),
     tensor(2, dtype=torch.int32), tensor(3, dtype=torch.int32),
     tensor(4, dtype=torch.int32), tensor(5, dtype=torch.int32),
     tensor(6, dtype=torch.int32), tensor(7, dtype=torch.int32),
     tensor(8, dtype=torch.int32), tensor(9, dtype=torch.int32)]
    >>> item_set[:]
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
    >>> item_set.names
    ('seeds',)

    3. Single tensor: seed nodes.

    >>> node_ids = torch.arange(0, 5)
    >>> item_set = gb.ItemSet(node_ids, names="seeds")
    >>> list(item_set)
    [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4)]
    >>> item_set[:]
    tensor([0, 1, 2, 3, 4])
    >>> item_set.names
    ('seeds',)

    4. Tuple of tensors with same shape: seed nodes and labels.

    >>> node_ids = torch.arange(0, 5)
    >>> labels = torch.arange(5, 10)
    >>> item_set = gb.ItemSet(
    ...     (node_ids, labels), names=("seeds", "labels"))
    >>> list(item_set)
    [(tensor(0), tensor(5)), (tensor(1), tensor(6)), (tensor(2), tensor(7)),
     (tensor(3), tensor(8)), (tensor(4), tensor(9))]
    >>> item_set[:]
    (tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9]))
    >>> item_set.names
    ('seeds', 'labels')

    5. Tuple of tensors with different shape: seeds and labels.

    >>> seeds = torch.arange(0, 10).reshape(-1, 2)
    >>> labels = torch.tensor([1, 1, 0, 0, 0])
    >>> item_set = gb.ItemSet(
    ...     (seeds, labels), names=("seeds", "lables"))
    >>> list(item_set)
    [(tensor([0, 1]), tensor([1])),
     (tensor([2, 3]), tensor([1])),
     (tensor([4, 5]), tensor([0])),
     (tensor([6, 7]), tensor([0])),
     (tensor([8, 9]), tensor([0]))]
    >>> item_set[:]
    (tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
     tensor([1, 1, 0, 0, 0]))
    >>> item_set.names
    ('seeds', 'labels')

    6. Tuple of tensors with different shape: hyperlink and labels.

    >>> seeds = torch.arange(0, 10).reshape(-1, 5)
    >>> labels = torch.tensor([1, 0])
    >>> item_set = gb.ItemSet(
    ...     (seeds, labels), names=("seeds", "lables"))
    >>> list(item_set)
    [(tensor([0, 1, 2, 3, 4]), tensor([1])),
     (tensor([5, 6, 7, 8, 9]), tensor([0]))]
    >>> item_set[:]
    (tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
     tensor([1, 0]))
    >>> item_set.names
    ('seeds', 'labels')
    """

    def __init__(
        self,
        items: Union[int, torch.Tensor, Tuple[torch.Tensor]],
        names: Union[str, Tuple[str]] = None,
    ) -> None:
        if is_scalar(items):
            self._length = int(items)
            self._items = items
        elif isinstance(items, tuple):
            self._length = len(items[0])
            if any(self._length != len(item) for item in items):
                raise ValueError("Size mismatch between items.")
            self._items = items
        else:
            self._length = len(items)
            self._items = (items,)
        self._num_items = (
            len(self._items) if isinstance(self._items, tuple) else 1
        )
        if names is not None:
            if isinstance(names, tuple):
                self._names = names
            else:
                self._names = (names,)
            assert self._num_items == len(self._names), (
                f"Number of items ({self._num_items}) and "
                f"names ({len(self._names)}) don't match."
            )
        else:
            self._names = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Union[int, slice, Iterable[int]]):
        if is_scalar(self._items):
            dtype = getattr(self._items, "dtype", torch.int64)
            if isinstance(index, slice):
                start, stop, step = index.indices(self._length)
                return torch.arange(start, stop, step, dtype=dtype)
            elif isinstance(index, int):
                if index < 0:
                    index += self._length
                if index < 0 or index >= self._length:
                    raise IndexError(
                        f"{type(self).__name__} index out of range."
                    )
                return torch.tensor(index, dtype=dtype)
            elif isinstance(index, torch.Tensor):
                return index.to(dtype)
            else:
                raise TypeError(
                    f"{type(self).__name__} indices must be int, slice, or "
                    f"torch.Tensor, not {type(index)}."
                )
        elif self._num_items == 1:
            return self._items[0][index]
        else:
            return tuple(item[index] for item in self._items)

    @property
    def names(self) -> Tuple[str]:
        """Return the names of the items."""
        return self._names

    @property
    def num_items(self) -> int:
        """Return the number of the items."""
        return self._num_items

    def __repr__(self) -> str:
        ret = (
            f"{self.__class__.__name__}(\n"
            f"    items={self._items},\n"
            f"    names={self._names},\n"
            f")"
        )
        return ret


class HeteroItemSet:
    r"""A collection of itemsets, each associated with a unique type.

    This class aims to assemble existing itemsets with different types, for
    example, seed_nodes of different node types in a graph.

    Parameters
    ----------
    itemsets: Dict[str, ItemSet]
        A dictionary whose keys are types and values are ItemSet instances.

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb

    1. Each itemset is a single tensor: seed nodes.

    >>> node_ids_user = torch.arange(0, 5)
    >>> node_ids_item = torch.arange(5, 10)
    >>> item_set = gb.HeteroItemSet({
    ...     "user": gb.ItemSet(node_ids_user, names="seeds"),
    ...     "item": gb.ItemSet(node_ids_item, names="seeds")})
    >>> list(item_set)
    [{"user": tensor(0)}, {"user": tensor(1)}, {"user": tensor(2)},
     {"user": tensor(3)}, {"user": tensor(4)}, {"item": tensor(5)},
     {"item": tensor(6)}, {"item": tensor(7)}, {"item": tensor(8)},
     {"item": tensor(9)}}]
    >>> item_set[:]
    {"user": tensor([0, 1, 2, 3, 4]), "item": tensor([5, 6, 7, 8, 9])}
    >>> item_set.names
    ('seeds',)

    2. Each itemset is a tuple of tensors with same shape: seed nodes and
    labels.

    >>> node_ids_user = torch.arange(0, 2)
    >>> labels_user = torch.arange(0, 2)
    >>> node_ids_item = torch.arange(2, 5)
    >>> labels_item = torch.arange(2, 5)
    >>> item_set = gb.HeteroItemSet({
    ...     "user": gb.ItemSet(
    ...         (node_ids_user, labels_user),
    ...         names=("seeds", "labels")),
    ...     "item": gb.ItemSet(
    ...         (node_ids_item, labels_item),
    ...         names=("seeds", "labels"))})
    >>> list(item_set)
    [{"user": (tensor(0), tensor(0))}, {"user": (tensor(1), tensor(1))},
     {"item": (tensor(2), tensor(2))}, {"item": (tensor(3), tensor(3))},
     {"item": (tensor(4), tensor(4))}}]
    >>> item_set[:]
    {"user": (tensor([0, 1]), tensor([0, 1])),
     "item": (tensor([2, 3, 4]), tensor([2, 3, 4]))}
    >>> item_set.names
    ('seeds', 'labels')

    3. Each itemset is a tuple of tensors with different shape: seeds and
    labels.

    >>> seeds_like = torch.arange(0, 4).reshape(-1, 2)
    >>> labels_like = torch.tensor([1, 0])
    >>> seeds_follow = torch.arange(0, 6).reshape(-1, 2)
    >>> labels_follow = torch.tensor([1, 1, 0])
    >>> item_set = gb.HeteroItemSet({
    ...     "user:like:item": gb.ItemSet(
    ...         (seeds_like, labels_like),
    ...         names=("seeds", "labels")),
    ...     "user:follow:user": gb.ItemSet(
    ...         (seeds_follow, labels_follow),
    ...         names=("seeds", "labels"))})
    >>> list(item_set)
    [{'user:like:item': (tensor([0, 1]), tensor(1))},
     {'user:like:item': (tensor([2, 3]), tensor(0))},
     {'user:follow:user': (tensor([0, 1]), tensor(1))},
     {'user:follow:user': (tensor([2, 3]), tensor(1))},
     {'user:follow:user': (tensor([4, 5]), tensor(0))}]
    >>> item_set[:]
    {'user:like:item': (tensor([[0, 1], [2, 3]]),
                        tensor([1, 0])),
     'user:follow:user': (tensor([[0, 1], [2, 3], [4, 5]]),
                          tensor([1, 1, 0]))}
    >>> item_set.names
    ('seeds', 'labels')

    4. Each itemset is a tuple of tensors with different shape: hyperlink and
    labels.

    >>> first_seeds = torch.arange(0, 6).reshape(-1, 3)
    >>> first_labels = torch.tensor([1, 0])
    >>> second_seeds = torch.arange(0, 2).reshape(-1, 1)
    >>> second_labels = torch.tensor([1, 0])
    >>> item_set = gb.HeteroItemSet({
    ...     "query:user:item": gb.ItemSet(
    ...         (first_seeds, first_labels),
    ...         names=("seeds", "labels")),
    ...     "user": gb.ItemSet(
    ...         (second_seeds, second_labels),
    ...         names=("seeds", "labels"))})
    >>> list(item_set)
    [{'query:user:item': (tensor([0, 1, 2]), tensor(1))},
     {'query:user:item': (tensor([3, 4, 5]), tensor(0))},
     {'user': (tensor([0]), tensor(1))},
     {'user': (tensor([1]), tensor(0))}]
    >>> item_set[:]
    {'query:user:item': (tensor([[0, 1, 2], [3, 4, 5]]),
                        tensor([1, 0])),
     'user': (tensor([[0], [1]]),tensor([1, 0]))}
    >>> item_set.names
    ('seeds', 'labels')
    """

    def __init__(self, itemsets: Dict[str, ItemSet]) -> None:
        self._itemsets = itemsets
        self._names = next(iter(itemsets.values())).names
        assert all(
            self._names == itemset.names for itemset in itemsets.values()
        ), "All itemsets must have the same names."
        offset = [0] + [len(itemset) for itemset in self._itemsets.values()]
        self._offsets = torch.tensor(offset).cumsum(0)
        self._length = int(self._offsets[-1])
        self._keys = list(self._itemsets.keys())

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Union[int, slice, Iterable[int]]):
        if isinstance(index, int):
            if index < 0:
                index += self._length
            if index < 0 or index >= self._length:
                raise IndexError(f"{type(self).__name__} index out of range.")
            offset_idx = torch.searchsorted(self._offsets, index, right=True)
            offset_idx -= 1
            index -= self._offsets[offset_idx]
            key = self._keys[offset_idx]
            return {key: self._itemsets[key][index]}
        elif isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            if step != 1:
                return self.__getitem__(torch.arange(start, stop, step))
            assert start < stop, "Start must be smaller than stop."
            data = {}
            offset_idx_start = max(
                1, torch.searchsorted(self._offsets, start, right=False)
            )
            for offset_idx in range(offset_idx_start, len(self._offsets)):
                key = self._keys[offset_idx - 1]
                data[key] = self._itemsets[key][
                    max(0, start - self._offsets[offset_idx - 1]) : stop
                    - self._offsets[offset_idx - 1]
                ]
                if stop <= self._offsets[offset_idx]:
                    break
            return data
        elif isinstance(index, Iterable):
            if not isinstance(index, torch.Tensor):
                index = torch.tensor(index)
            assert torch.all((index >= 0) & (index < self._length))
            key_indices = (
                torch.searchsorted(self._offsets, index, right=True) - 1
            )
            data = {}
            for key_id, key in enumerate(self._keys):
                mask = (key_indices == key_id).nonzero().squeeze(1)
                if len(mask) == 0:
                    continue
                data[key] = self._itemsets[key][
                    index[mask] - self._offsets[key_id]
                ]
            return data
        else:
            raise TypeError(
                f"{type(self).__name__} indices must be int, slice, or "
                f"iterable of int, not {type(index)}."
            )

    @property
    def names(self) -> Tuple[str]:
        """Return the names of the items."""
        return self._names

    def __repr__(self) -> str:
        ret = (
            "{Classname}(\n"
            "    itemsets={itemsets},\n"
            "    names={names},\n"
            ")"
        )

        itemsets_str = textwrap.indent(
            repr(self._itemsets), " " * len("    itemsets=")
        ).strip()

        return ret.format(
            Classname=self.__class__.__name__,
            itemsets=itemsets_str,
            names=self._names,
        )


class ItemSetDict:
    """`ItemSetDict` is a deprecated class and will be removed in a future
    version. Please use `HeteroItemSet` instead.

    This class is an alias for `HeteroItemSet` and serves as a wrapper to
    provide a smooth transition for users of the old class name. It issues a
    deprecation warning upon instantiation and forwards all attribute access
    and method calls to an instance of `HeteroItemSet`.
    """

    def __init__(self, itemsets: Dict[str, ItemSet]) -> None:
        gb_warning(
            "ItemSetDict is deprecated and will be removed in the future. "
            "Please use HeteroItemSet instead.",
            category=DeprecationWarning,
        )
        self._new_instance = HeteroItemSet(itemsets)

    def __getattr__(self, name: str):
        return getattr(self._new_instance, name)

    def __getitem__(self, index):
        return self._new_instance[index]

    def __len__(self) -> int:
        return len(self._new_instance)

    def __repr__(self) -> str:
        ret = (
            "{Classname}(\n"
            "    itemsets={itemsets},\n"
            "    names={names},\n"
            ")"
        )
        itemsets_str = textwrap.indent(
            repr(self._itemsets), " " * len("    itemsets=")
        ).strip()
        return ret.format(
            Classname=self.__class__.__name__,
            itemsets=itemsets_str,
            names=self._names,
        )
