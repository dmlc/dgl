"""GraphBolt Itemset."""

import textwrap
from typing import Dict, Iterable, Iterator, Mapping, Tuple, Union

import torch
from torch.utils.data import Dataset

__all__ = ["ItemSet", "ItemSetDict", "ItemSet4", "ItemSetDict4"]


def is_scalar(x):
    """Checks if the input is a scalar."""
    return (
        len(x.shape) == 0 if isinstance(x, torch.Tensor) else isinstance(x, int)
    )


class ItemSet:
    r"""A wrapper of iterable data or tuple of iterable data.

    All itemsets that represent an iterable of items should subclass it. Such
    form of itemset is particularly useful when items come from a stream. This
    class requires each input itemset to be iterable.

    Parameters
    ----------
    items: Union[int, Iterable, Tuple[Iterable]]
        The items to be iterated over. If it is a single integer, a `range()`
        object will be created and iterated over. If it's multi-dimensional
        iterable such as `torch.Tensor`, it will be iterated over the first
        dimension. If it is a tuple, each item in the tuple is an iterable of
        items.
    names: Union[str, Tuple[str]], optional
        The names of the items. If it is a tuple, each name corresponds to an
        item in the tuple. The naming is arbitrary, but in general practice,
        the names should be chosen from ['labels', 'seeds', 'indexes'] to align
        with the attributes of class `dgl.graphbolt.MiniBatch`.

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

    3. Single iterable: seed nodes.

    >>> node_ids = torch.arange(0, 5)
    >>> item_set = gb.ItemSet(node_ids, names="seeds")
    >>> list(item_set)
    [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4)]
    >>> item_set[:]
    tensor([0, 1, 2, 3, 4])
    >>> item_set.names
    ('seeds',)

    4. Tuple of iterables with same shape: seed nodes and labels.

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

    5. Tuple of iterables with different shape: seeds and labels.

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

    6. Tuple of iterables with different shape: hyperlink and labels.

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
        items: Union[int, torch.Tensor, Iterable, Tuple[Iterable]],
        names: Union[str, Tuple[str]] = None,
    ) -> None:
        if is_scalar(items):
            self._length = int(items)
            self._items = items
            self._num_items = 1
        elif isinstance(items, tuple):
            try:
                self._length = len(items[0])
            except TypeError:
                self._length = None
            if self._length is not None:
                if any(self._length != len(item) for item in items):
                    raise ValueError("Size mismatch between items.")
            self._items = items
            self._num_items = len(items)
        else:
            try:
                self._length = len(items)
            except TypeError:
                self._length = None
            self._items = (items,)
            self._num_items = 1

        if names is not None:
            if isinstance(names, tuple):
                self._names = names
            else:
                self._names = (names,)
            assert self._num_items == len(self._names), (
                f"Number of items ({self._num_items}) and "
                f"names ({len(self._names)}) must match."
            )
        else:
            self._names = None

    def __iter__(self) -> Iterator:
        if is_scalar(self._items):
            dtype = getattr(self._items, "dtype", torch.int64)
            yield from torch.arange(self._items, dtype=dtype)
            return

        if self._num_items == 1:
            yield from self._items[0]
            return

        if self._length is not None:
            # Use for-loop to iterate over the items. It can avoid a long
            # waiting time when the items are torch tensors. Since torch
            # tensors need to call self.unbind(0) to slice themselves.
            # While for-loops are slower than zip, they prevent excessive
            # wait times during the loading phase, and the impact on overall
            # performance during the training/testing stage is minimal.
            # For more details, see https://github.com/dmlc/dgl/pull/6293.
            for i in range(self._length):
                yield tuple(item[i] for item in self._items)
        else:
            # If the items are not Sized, we use zip to iterate over them.
            zip_items = zip(*self._items)
            for item in zip_items:
                yield tuple(item)

    def __getitem__(self, idx: Union[int, slice, Iterable]) -> Tuple:
        if self._length is None:
            raise TypeError(
                f"{type(self).__name__} instance doesn't support indexing."
            )
        if is_scalar(self._items):
            if isinstance(idx, slice):
                start, stop, step = idx.indices(self._length)
                dtype = getattr(self._items, "dtype", torch.int64)
                return torch.arange(start, stop, step, dtype=dtype)
            if isinstance(idx, int):
                if idx < 0:
                    idx += self._length
                if idx < 0 or idx >= self._length:
                    raise IndexError(
                        f"{type(self).__name__} index out of range."
                    )
                return (
                    torch.tensor(idx, dtype=self._items.dtype)
                    if isinstance(self._items, torch.Tensor)
                    else idx
                )
            raise TypeError(
                f"{type(self).__name__} indices must be integer or slice."
            )
        if self._num_items == 1:
            return self._items[0][idx]
        return tuple(item[idx] for item in self._items)

    @property
    def names(self) -> Tuple[str]:
        """Return the names of the items."""
        return self._names

    @property
    def num_items(self) -> int:
        """Return the number of the items."""
        return self._num_items

    def __len__(self):
        if self._length is None:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length."
            )
        return self._length

    def __repr__(self) -> str:
        ret = (
            f"{self.__class__.__name__}(\n"
            f"    items={self._items},\n"
            f"    names={self._names},\n"
            f")"
        )

        return ret


class ItemSetDict:
    r"""Dictionary wrapper of **ItemSet**.

    Each item is retrieved by iterating over each itemset and returned with
    corresponding key as a dict.

    Parameters
    ----------
    itemsets: Dict[str, ItemSet]

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb

    1. Single iterable: seed nodes.

    >>> node_ids_user = torch.arange(0, 5)
    >>> node_ids_item = torch.arange(5, 10)
    >>> item_set = gb.ItemSetDict({
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

    2. Tuple of iterables with same shape: seed nodes and labels.

    >>> node_ids_user = torch.arange(0, 2)
    >>> labels_user = torch.arange(0, 2)
    >>> node_ids_item = torch.arange(2, 5)
    >>> labels_item = torch.arange(2, 5)
    >>> item_set = gb.ItemSetDict({
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

    3. Tuple of iterables with different shape: seeds and labels.

    >>> seeds_like = torch.arange(0, 4).reshape(-1, 2)
    >>> labels_like = torch.tensor([1, 0])
    >>> seeds_follow = torch.arange(0, 6).reshape(-1, 2)
    >>> labels_follow = torch.tensor([1, 1, 0])
    >>> item_set = gb.ItemSetDict({
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

    4. Tuple of iterables with different shape: hyperlink and labels.

    >>> first_seeds = torch.arange(0, 6).reshape(-1, 3)
    >>> first_labels = torch.tensor([1, 0])
    >>> second_seeds = torch.arange(0, 2).reshape(-1, 1)
    >>> second_labels = torch.tensor([1, 0])
    >>> item_set = gb.ItemSetDict({
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
        self._names = itemsets[list(itemsets.keys())[0]].names
        assert all(
            self._names == itemset.names for itemset in itemsets.values()
        ), "All itemsets must have the same names."
        try:
            # For indexable itemsets, we compute the offsets for each itemset
            # in advance to speed up indexing.
            offsets = [0] + [
                len(itemset) for itemset in self._itemsets.values()
            ]
            self._offsets = torch.tensor(offsets).cumsum(0)
        except TypeError:
            self._offsets = None

    def __iter__(self) -> Iterator:
        for key, itemset in self._itemsets.items():
            for item in itemset:
                yield {key: item}

    def __len__(self) -> int:
        return sum(len(itemset) for itemset in self._itemsets.values())

    def __getitem__(self, idx: Union[int, slice]) -> Dict[str, Tuple]:
        if self._offsets is None:
            raise TypeError(
                f"{type(self).__name__} instance doesn't support indexing."
            )
        total_num = self._offsets[-1]
        if isinstance(idx, int):
            if idx < 0:
                idx += total_num
            if idx < 0 or idx >= total_num:
                raise IndexError(f"{type(self).__name__} index out of range.")
            offset_idx = torch.searchsorted(self._offsets, idx, right=True)
            offset_idx -= 1
            idx -= self._offsets[offset_idx]
            key = list(self._itemsets.keys())[offset_idx]
            return {key: self._itemsets[key][idx]}
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(total_num)
            assert step == 1, "Step must be 1."
            assert start < stop, "Start must be smaller than stop."
            data = {}
            offset_idx_start = max(
                1, torch.searchsorted(self._offsets, start, right=False)
            )
            keys = list(self._itemsets.keys())
            for offset_idx in range(offset_idx_start, len(self._offsets)):
                key = keys[offset_idx - 1]
                data[key] = self._itemsets[key][
                    max(0, start - self._offsets[offset_idx - 1]) : stop
                    - self._offsets[offset_idx - 1]
                ]
                if stop <= self._offsets[offset_idx]:
                    break
            return data
        else:
            raise TypeError(
                f"{type(self).__name__} indices must be int or slice."
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


class ItemSet4(Dataset):
    r"""Class for iterating over tensor-like data.
    Experimental. Implemented only __getitem__() accepting slice and list.
    """

    def __init__(
        self,
        items: Union[torch.Tensor, Mapping, Tuple[Mapping]],
        names: Union[str, Tuple[str]] = None,
    ):
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
        if names is not None:
            num_items = (
                len(self._items) if isinstance(self._items, tuple) else 1
            )
            if isinstance(names, tuple):
                self._names = names
            else:
                self._names = (names,)
            assert num_items == len(self._names), (
                f"Number of items ({num_items}) and "
                f"names ({len(self._names)}) don't match."
            )
        else:
            self._names = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Union[int, slice, Iterable[int]]):
        if is_scalar(self._items):
            if isinstance(index, slice):
                start, stop, step = index.indices(int(self._items))
                dtype = getattr(self._items, "dtype", torch.int64)
                return torch.arange(start, stop, step, dtype=dtype)
            elif isinstance(index, int):
                if index < 0:
                    index += int(self._items)
                if index < 0 or index >= int(self._items):
                    raise IndexError(
                        f"{type(self).__name__} index out of range."
                    )
                return torch.tensor(index, dtype=self._items.dtype)
            elif isinstance(index, Iterable):
                dtype = getattr(self._items, "dtype", torch.int64)
                return torch.tensor(index, dtype=dtype)
            else:
                raise TypeError(
                    f"{type(self).__name__} indices must be int, slice, or "
                    f"iterable of int, but got {type(index)}."
                )
        elif len(self._items) == 1:
            return self._items[0][index]
        else:
            return tuple(item[index] for item in self._items)

    @property
    def names(self) -> Tuple[str]:
        """Return the names of the items."""
        return self._names

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}(\n"
            f"    items={self._items},\n"
            f"    names={self._names},\n"
            f")"
        )
        return _repr


class ItemSetDict4(Dataset):
    r"""Experimental."""

    def __init__(self, itemsets: Dict[str, ItemSet4]) -> None:
        super().__init__()
        self._itemsets = itemsets
        self._names = next(iter(itemsets.values())).names
        if any(self._names != itemset.names for itemset in itemsets.values()):
            raise ValueError("All itemsets must have the same names.")
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
            # print(f"slice: {slice}, start, stop, step: {(start, stop, step)}")
            # print(f"res list: {list(range(start, stop, step))}")
            if step != 1:
                return self.__getitem__(list(range(start, stop, step)))
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
            data = {key: [] for key in self._keys}
            for idx in index:
                if idx < 0:
                    idx += self._length
                if idx < 0 or idx >= self._length:
                    raise IndexError(
                        f"{type(self).__name__} index out of range."
                    )
                offset_idx = torch.searchsorted(self._offsets, idx, right=True)
                offset_idx -= 1
                idx -= self._offsets[offset_idx]
                key = self._keys[offset_idx]
                data[key].append(int(idx))
            for key in self._keys:
                indices = data[key]
                if len(indices) == 0:
                    del data[key]
                    continue
                item_set = self._itemsets[key]
                try:
                    value = item_set[indices]
                except TypeError:
                    # In case the itemset doesn't support list indexing.
                    value = tuple(item_set[idx] for idx in indices)
                finally:
                    data[key] = value
            return data
        else:
            raise TypeError(
                f"{type(self).__name__} indices must be int, slice, or "
                f"iterable of int, but got {type(index)}."
            )

    @property
    def names(self) -> Tuple[str]:
        """Return the names of the items."""
        return self._names

    def __repr__(self) -> str:
        _repr = (
            "{Classname}(\n"
            "    itemsets={itemsets},\n"
            "    names={names},\n"
            ")"
        )
        itemsets_str = textwrap.indent(
            repr(self._itemsets), " " * len("    itemsets=")
        ).strip()
        return _repr.format(
            Classname=self.__class__.__name__,
            itemsets=itemsets_str,
            names=self._names,
        )
