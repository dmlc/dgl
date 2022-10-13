try:
    from .conv import *  # noqa
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "dgl.contrib.cugraph.nn requires pylibcugraphops to be installed."
    )
