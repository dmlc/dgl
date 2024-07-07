"""Miscallenous utils copied from DGL."""
from collections.abc import Mapping, Sequence


def is_listlike(data):
    """Return if the data is a sequence but not a string."""
    return isinstance(data, Sequence) and not isinstance(data, str)


def recursive_apply(data, fn, *args, **kwargs):
    """Recursively apply a function to every element in a container.

    If the input data is a list or any sequence other than a string, returns a list
    whose elements are the same elements applied with the given function.

    If the input data is a dict or any mapping, returns a dict whose keys are the same
    and values are the elements applied with the given function.

    If the input data is a nested container, the result will have the same nested
    structure where each element is transformed recursively.

    The first argument of the function will be passed with the individual elements from
    the input data, followed by the arguments in :attr:`args` and :attr:`kwargs`.

    Parameters
    ----------
    data : any
        Any object.
    fn : callable
        Any function.
    args, kwargs :
        Additional arguments and keyword-arguments passed to the function.

    Examples
    --------
    Applying a ReLU function to a dictionary of tensors:

    >>> h = {k: torch.randn(3) for k in ['A', 'B', 'C']}
    >>> h = recursive_apply(h, torch.nn.functional.relu)
    >>> assert all((v >= 0).all() for v in h.values())
    """
    if isinstance(data, Mapping):
        return {
            k: recursive_apply(v, fn, *args, **kwargs) for k, v in data.items()
        }
    elif isinstance(data, tuple):
        return tuple(recursive_apply(v, fn, *args, **kwargs) for v in data)
    elif is_listlike(data):
        return [recursive_apply(v, fn, *args, **kwargs) for v in data]
    else:
        return fn(data, *args, **kwargs)
