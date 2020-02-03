"""Decorator for deprecation message.

This is used in migrating the chem related code to DGL-LifeSci.
Todo(Mufei): remove it in v0.5.

The code is adapted from
https://stackoverflow.com/questions/2536307/
decorators-in-the-python-standard-lib-deprecated-specifically/48632082#48632082.
"""
import warnings

def deprecated(message, mode='func'):
    """Print formatted deprecation message.

    Parameters
    ----------
    message : str
    mode : str
        'func' for function and 'class' for class.

    Return
    ------
    callable
    """
    assert mode in ['func', 'class']

    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            if mode == 'func':
                warnings.warn("{} is deprecated and will be removed from dgl in v0.5. {}".format(
                    func.__name__, message), category=DeprecationWarning, stacklevel=2)
            else:
                warnings.warn("The class is deprecated and "
                              "will be removed from dgl in v0.5. {}".format(message),
                              category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator
