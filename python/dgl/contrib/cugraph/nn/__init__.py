try:
    from .conv import *
except ModuleNotFoundError:
    raise ModuleNotFoundError("dgl.contrib.cugraph.nn requires pylibcugraphops to be installed.")
