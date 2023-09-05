"""GraphBolt Itemset."""

from typing import Dict, Iterable, Iterator, Sized, Tuple

__all__ = ["ItemSet", "ItemSetDict"]


class ItemSet:
    r"""An iterable itemset.

    All itemsets that represent an iterable of items should subclass it. Such
    form of itemset is particularly useful when items come from a stream. This
    class requires each input itemset to be iterable.

    Parameters
    ----------
    items: Iterable or Tuple[Iterable]
        The items to be iterated over. If it is a tuple, each item in the tuple
        is an iterable of items.
    names: str or Tuple[str], optional
        The names of the items. If it is a tuple, each name corresponds to an
        item in the tuple.

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb

    1. Single iterable: seed nodes.
    >>> node_ids = torch.arange(0, 5)
    >>> item_set = gb.ItemSet(node_ids, names="seed_nodes")
    >>> list(item_set)
    [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4)]
    >>> item_set.names
    ('seed_nodes',)

    2. Tuple of iterables with same shape: seed nodes and labels.
    >>> node_ids = torch.arange(0, 5)
    >>> labels = torch.arange(5, 10)
    >>> item_set = gb.ItemSet(
    ...     (node_ids, labels), names=("seed_nodes", "labels"))
    >>> list(item_set)
    [(tensor(0), tensor(5)), (tensor(1), tensor(6)), (tensor(2), tensor(7)),
     (tensor(3), tensor(8)), (tensor(4), tensor(9))]
    >>> item_set.names
    ('seed_nodes', 'labels')

    3. Tuple of iterables with different shape: node pairs and negative dsts.
    >>> node_pairs = torch.arange(0, 10).reshape(-1, 2)
    >>> neg_dsts = torch.arange(10, 25).reshape(-1, 3)
    >>> item_set = gb.ItemSet(
    ...     (node_pairs, neg_dsts), names=("node_pairs", "negative_dsts"))
    >>> list(item_set)
    [(tensor([0, 1]), tensor([10, 11, 12])),
     (tensor([2, 3]), tensor([13, 14, 15])),
     (tensor([4, 5]), tensor([16, 17, 18])),
     (tensor([6, 7]), tensor([19, 20, 21])),
     (tensor([8, 9]), tensor([22, 23, 24]))]
    >>> item_set.names
    ('node_pairs', 'negative_dsts')
    """

    def __init__(
        self,
        items: Iterable or Tuple[Iterable],
        names: str or Tuple[str] = None,
    ) -> None:
        if isinstance(items, tuple):
            self._items = items
        else:
            self._items = (items,)
        if names is not None:
            if isinstance(names, tuple):
                self._names = names
            else:
                self._names = (names,)
            assert len(self._items) == len(self._names), (
                f"Number of items ({len(self._items)}) and "
                f"names ({len(self._names)}) must match."
            )
        else:
            self._names = None

    def __iter__(self) -> Iterator:
        if len(self._items) == 1:
            yield from self._items[0]
            return
        zip_items = zip(*self._items)
        for item in zip_items:
            yield tuple(item)

    def __len__(self) -> int:
        if isinstance(self._items[0], Sized):
            return len(self._items[0])
        raise TypeError(
            f"{type(self).__name__} instance doesn't have valid length."
        )

    @property
    def names(self) -> Tuple[str]:
        """Return the names of the items."""
        return self._names


class ItemSetDict:
    r"""An iterable ItemsetDict.

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
    ...     "user": gb.ItemSet(node_ids_user, names="seed_nodes"),
    ...     "item": gb.ItemSet(node_ids_item, names="seed_nodes")})
    >>> list(item_set)
    [{"user": tensor(0)}, {"user": tensor(1)}, {"user": tensor(2)},
     {"user": tensor(3)}, {"user": tensor(4)}, {"item": tensor(5)},
     {"item": tensor(6)}, {"item": tensor(7)}, {"item": tensor(8)},
     {"item": tensor(9)}}]
    >>> item_set.names
    ('seed_nodes',)

    2. Tuple of iterables with same shape: seed nodes and labels.
    >>> node_ids_user = torch.arange(0, 2)
    >>> labels_user = torch.arange(0, 2)
    >>> node_ids_item = torch.arange(2, 5)
    >>> labels_item = torch.arange(2, 5)
    >>> item_set = gb.ItemSetDict({
    ...     "user": gb.ItemSet(
    ...         (node_ids_user, labels_user),
    ...         names=("seed_nodes", "labels")),
    ...     "item": gb.ItemSet(
    ...         (node_ids_item, labels_item),
    ...         names=("seed_nodes", "labels"))})
    >>> list(item_set)
    [{"user": (tensor(0), tensor(0))}, {"user": (tensor(1), tensor(1))},
     {"item": (tensor(2), tensor(2))}, {"item": (tensor(3), tensor(3))},
     {"item": (tensor(4), tensor(4))}}]
    >>> item_set.names
    ('seed_nodes', 'labels')

    3. Tuple of iterables with different shape: node pairs and negative dsts.
    >>> node_pairs_like = torch.arange(0, 4).reshape(-1, 2)
    >>> neg_dsts_like = torch.arange(4, 10).reshape(-1, 3)
    >>> node_pairs_follow = torch.arange(0, 6).reshape(-1, 2)
    >>> neg_dsts_follow = torch.arange(6, 15).reshape(-1, 3)
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet(
    ...         (node_pairs_like, neg_dsts_like),
    ...         names=("node_pairs", "negative_dsts")),
    ...     "user:follow:user": gb.ItemSet(
    ...         (node_pairs_follow, neg_dsts_follow),
    ...         names=("node_pairs", "negative_dsts"))})
    >>> list(item_set)
    [{"user:like:item": (tensor([0, 1]), tensor([4, 5, 6]))},
     {"user:like:item": (tensor([2, 3]), tensor([7, 8, 9]))},
     {"user:follow:user": (tensor([0, 1]), tensor([ 6,  7,  8,  9, 10, 11]))},
     {"user:follow:user": (tensor([2, 3]), tensor([12, 13, 14, 15, 16, 17]))},
     {"user:follow:user": (tensor([4, 5]), tensor([18, 19, 20, 21, 22, 23]))}]
    >>> item_set.names
    ('node_pairs', 'negative_dsts')
    """

    def __init__(self, itemsets: Dict[str, ItemSet]) -> None:
        self._itemsets = itemsets
        self._names = itemsets[list(itemsets.keys())[0]].names
        assert all(
            self._names == itemset.names for itemset in itemsets.values()
        ), "All itemsets must have the same names."

    def __iter__(self) -> Iterator:
        for key, itemset in self._itemsets.items():
            for item in itemset:
                yield {key: item}

    def __len__(self) -> int:
        return sum(len(itemset) for itemset in self._itemsets.values())

    @property
    def names(self) -> Tuple[str]:
        """Return the names of the items."""
        return self._names
