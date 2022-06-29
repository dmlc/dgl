import inspect

class _PredicateDispatcher(object):
    def __init__(self, default_func):
        self.registry = []
        self.default_func = default_func

    def register(self, predicate, impl):
        self.registry.append((predicate, impl))

    def __call__(self, *args, **kwargs):
        for predicate, impl in self.registry:
            sig_predicate = inspect.signature(predicate)
            sig_impl = inspect.signature(impl)
            kwargs_predicate = {}

            for i, (name, param) in enumerate(sig_predicate.parameters.items()):
                if name in sig_impl.parameters.keys():
                    kwargs_predicate[name] = args[i] if i < len(args) else kwargs[name]

            if predicate(**kwargs_predicate):
                return impl(*args, **kwargs)
        return self.default_func(*args, **kwargs)

def predicate_dispatch(func):
    """Dispatch the input arguments according to registered predicates, in a similar
    fashion as ``functools.singledispatcher``.
    """
    return _PredicateDispatcher(func)
