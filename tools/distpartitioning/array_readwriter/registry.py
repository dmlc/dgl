REGISTRY = {}


def register_array_parser(name):
    def _deco(cls):
        REGISTRY[name] = cls
        return cls

    return _deco


def get_array_parser(**fmt_meta):
    cls = REGISTRY[fmt_meta.pop("name")]
    return cls(**fmt_meta)
