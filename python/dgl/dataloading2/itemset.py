"""GraphBolt Itemset."""

__all__ = ["ItemSet", "DictItemSet"]


class ItemSet:
    r"""An iterable itemset.

    All itemsets that represent an iterable of items should subclass it. Such
    form of itemset is particularly useful when items come from a stream. This
    class requires each input itemset to be iterable.

    Parameters
    ----------
    items: Iterable or Tuple[Iterable]
    """

    def __init__(self, items):
        if isinstance(items, tuple):
            self._items = items
        else:
            self._items = (items,)

    def __iter__(self):
        if len(self._items) == 1:
            yield from self._items[0]
            return
        zip_items = zip(*self._items)
        for item in zip_items:
            yield tuple(item)

    def __getitem__(self, _):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class DictItemSet(ItemSet):
    r"""Itemset wrapping multiple itemsets with keys.

    Each item is retrieved by iterating over each itemset and returned with
    corresponding key as a dict.

    Parameters
    ----------
    itemsets: Dict[str, ItemSet]
    """

    def __init__(self, itemsets):
        self._itemsets = itemsets

    def __iter__(self):
        for key, itemset in self._itemsets.items():
            for item in itemset:
                yield {key: item}
