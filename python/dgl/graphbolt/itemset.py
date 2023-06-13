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

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb

    1. Single iterable.
    >>> node_ids = torch.arange(0, 5)
    >>> item_set = gb.ItemSet(node_ids)
    >>> list(item_set)
    [tensor(0), tensor(1), tensor(2), tensor(3), tensor(4)]

    2. Tuple of iterables with same shape.
    >>> node_pairs = (torch.arange(0, 5), torch.arange(5, 10))
    >>> item_set = gb.ItemSet(node_pairs)
    >>> list(item_set)
    [(tensor(0), tensor(5)), (tensor(1), tensor(6)), (tensor(2), tensor(7)),
     (tensor(3), tensor(8)), (tensor(4), tensor(9))]

    3. Tuple of iterables with different shape.
    >>> heads = torch.arange(0, 5)
    >>> tails = torch.arange(5, 10)
    >>> neg_tails = torch.arange(10, 20).reshape(5, 2)
    >>> item_set = gb.ItemSet((heads, tails, neg_tails))
    >>> list(item_set)
    [(tensor(0), tensor(5), tensor([10, 11])),
     (tensor(1), tensor(6), tensor([12, 13])),
     (tensor(2), tensor(7), tensor([14, 15])),
     (tensor(3), tensor(8), tensor([16, 17])),
     (tensor(4), tensor(9), tensor([18, 19]))]
    """

    def __init__(self, items: Iterable or Tuple[Iterable]) -> None:
        if isinstance(items, tuple):
            self._items = items
        else:
            self._items = (items,)

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

    1. Single iterable.
    >>> node_ids_user = torch.arange(0, 5)
    >>> node_ids_item = torch.arange(5, 10)
    >>> item_set = gb.ItemSetDict({
    ...     'user': gb.ItemSet(node_ids_user),
    ...     'item': gb.ItemSet(node_ids_item)})
    >>> list(item_set)
    [{'user': tensor(0)}, {'user': tensor(1)}, {'user': tensor(2)},
     {'user': tensor(3)}, {'user': tensor(4)}, {'item': tensor(5)},
     {'item': tensor(6)}, {'item': tensor(7)}, {'item': tensor(8)},
     {'item': tensor(9)}]

    2. Tuple of iterables with same shape.
    >>> node_pairs_like = (torch.arange(0, 2), torch.arange(0, 2))
    >>> node_pairs_follow = (torch.arange(0, 3), torch.arange(3, 6))
    >>> item_set = gb.ItemSetDict({
    ...     ('user', 'like', 'item'): gb.ItemSet(node_pairs_like),
    ...     ('user', 'follow', 'user'): gb.ItemSet(node_pairs_follow)})
    >>> list(item_set)
    [{('user', 'like', 'item'): (tensor(0), tensor(0))},
     {('user', 'like', 'item'): (tensor(1), tensor(1))},
     {('user', 'follow', 'user'): (tensor(0), tensor(3))},
     {('user', 'follow', 'user'): (tensor(1), tensor(4))},
     {('user', 'follow', 'user'): (tensor(2), tensor(5))}]

    3. Tuple of iterables with different shape.
    >>> like = (torch.arange(0, 2), torch.arange(0, 2),
    ...     torch.arange(0, 4).reshape(-1, 2))
    >>> follow = (torch.arange(0, 3), torch.arange(3, 6),
    ...     torch.arange(0, 6).reshape(-1, 2))
    >>> item_set = gb.ItemSetDict({
    ...     ('user', 'like', 'item'): gb.ItemSet(like),
    ...     ('user', 'follow', 'user'): gb.ItemSet(follow)})
    >>> list(item_set)
    [{('user', 'like', 'item'): (tensor(0), tensor(0), tensor([0, 1]))},
     {('user', 'like', 'item'): (tensor(1), tensor(1), tensor([2, 3]))},
     {('user', 'follow', 'user'): (tensor(0), tensor(3), tensor([0, 1]))},
     {('user', 'follow', 'user'): (tensor(1), tensor(4), tensor([2, 3]))},
     {('user', 'follow', 'user'): (tensor(2), tensor(5), tensor([4, 5]))}]
    """

    def __init__(self, itemsets: Dict[str, ItemSet]) -> None:
        self._itemsets = itemsets

    def __iter__(self) -> Iterator:
        for key, itemset in self._itemsets.items():
            for item in itemset:
                yield {key: item}

    def __len__(self) -> int:
        return sum(len(itemset) for itemset in self._itemsets.values())
