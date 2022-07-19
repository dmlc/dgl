REGISTRY = {}

def register(name):
    def _deco(cls):
        REGISTRY[name] = cls
        return cls
    return _deco

def read(path, **fmt_meta):
    cls = REGISTRY[fmt_meta.pop('name')]
    return cls(**fmt_meta).read(path)

def write(path, arr, **fmt_meta):
    cls = REGISTRY[fmt_meta.pop('name')]
    return cls(**fmt_meta).write(path, arr)
