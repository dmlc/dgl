"""Built-in functions."""

from functools import update_wrapper

__all__ = ['create_bundled_function_class']

def create_bundled_function_class(name, cls):
    class Bundled(cls):
        def __init__(self, fn_list):
            if not isinstance(fn_list, (list, tuple)):
                fn_list = [fn_list]
            self.fn_list = fn_list

        def is_spmv_supported(self, *args, **kwargs):
            return all(isinstance(fn, cls) and
                       fn.is_spmv_supported(*args, **kwargs)
                       for fn in self.fn_list)

        def __call__(self, *args, **kwargs):
            ret = {}
            for fn in self.fn_list:
                result = fn(*args, **kwargs)
                ret.update(result)
            return ret

        def name(self):
            return "bundled"

    # Fake the names for introspection
    Bundled.__module__ = cls.__module__
    Bundled.__name__ = name
    Bundled.__qualname__ = name

    for method_name in ('__init__', '__call__', 'is_spmv_supported', 'name'):
        method = getattr(Bundled, method_name)
        method.__qualname__ = '{}.{}'.format(Bundled.__qualname__, method_name)

    for method_name in ('__call__', 'is_spmv_supported', 'name'):
        method = getattr(Bundled, method_name)
        method = update_wrapper(method,
                                cls.__dict__[method.__name__],
                                ('__module__', '__doc__', '__annotations__'))

    return Bundled
