import inspect
import json
import os
import pickle
import shutil
import time
import zipfile
from functools import partial, reduce, wraps
from timeit import default_timer

import dgl

import numpy as np
import pandas
import requests
import torch
from ogb.nodeproppred import DglNodePropPredDataset


def _download(url, path, filename):
    fn = os.path.join(path, filename)
    if os.path.exists(fn):
        return

    os.makedirs(path, exist_ok=True)
    f_remote = requests.get(url, stream=True)
    sz = f_remote.headers.get("content-length")
    assert f_remote.status_code == 200, "fail to open {}".format(url)
    with open(fn, "wb") as writer:
        for chunk in f_remote.iter_content(chunk_size=1024 * 1024):
            writer.write(chunk)
    print("Download finished.")


import traceback
from _thread import start_new_thread

# GRAPH_CACHE = {}
import torch.multiprocessing as mp


def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """

    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()

        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)

    return decorated_function


def get_graph(name, format=None):
    # global GRAPH_CACHE
    # if name in GRAPH_CACHE:
    #     return GRAPH_CACHE[name].to(format)
    if isinstance(format, str):
        format = [format]  # didn't specify format
    if format is None:
        format = ["csc", "csr", "coo"]
    g = None
    if name == "cora":
        g = dgl.data.CoraGraphDataset(verbose=False)[0]
    elif name == "pubmed":
        g = dgl.data.PubmedGraphDataset(verbose=False)[0]
    elif name == "livejournal":
        bin_path = "/tmp/dataset/livejournal/livejournal_{}.bin".format(format)
        if os.path.exists(bin_path):
            g_list, _ = dgl.load_graphs(bin_path)
            g = g_list[0]
        else:
            g = get_livejournal().formats(format)
            dgl.save_graphs(bin_path, [g])
    elif name == "friendster":
        bin_path = "/tmp/dataset/friendster/friendster_{}.bin".format(format)
        if os.path.exists(bin_path):
            g_list, _ = dgl.load_graphs(bin_path)
            g = g_list[0]
        else:
            # the original node IDs of friendster are not consecutive, so we compact it
            g = dgl.compact_graphs(get_friendster()).formats(format)
            dgl.save_graphs(bin_path, [g])
    elif name == "reddit":
        bin_path = "/tmp/dataset/reddit/reddit_{}.bin".format(format)
        if os.path.exists(bin_path):
            g_list, _ = dgl.load_graphs(bin_path)
            g = g_list[0]
        else:
            g = dgl.data.RedditDataset(self_loop=True)[0].formats(format)
            dgl.save_graphs(bin_path, [g])
    elif name.startswith("ogb"):
        g = get_ogb_graph(name)
    else:
        raise Exception("Unknown dataset")
    # GRAPH_CACHE[name] = g
    g = g.formats(format)
    return g


def get_ogb_graph(name):
    os.symlink("/tmp/dataset/", os.path.join(os.getcwd(), "dataset"))
    data = DglNodePropPredDataset(name=name)
    return data[0][0]


def get_livejournal():
    # Same as https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
    _download(
        "https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/livejournal/soc-LiveJournal1.txt.gz",
        "/tmp/dataset/livejournal",
        "soc-LiveJournal1.txt.gz",
    )
    df = pandas.read_csv(
        "/tmp/dataset/livejournal/soc-LiveJournal1.txt.gz",
        sep="\t",
        skiprows=4,
        header=None,
        names=["src", "dst"],
        compression="gzip",
    )
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    return dgl.graph((src, dst))


def get_friendster():
    # Same as https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
    _download(
        "https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/friendster/com-friendster.ungraph.txt.gz",
        "/tmp/dataset/friendster",
        "com-friendster.ungraph.txt.gz",
    )
    df = pandas.read_csv(
        "/tmp/dataset/friendster/com-friendster.ungraph.txt.gz",
        sep="\t",
        skiprows=4,
        header=None,
        names=["src", "dst"],
        compression="gzip",
    )
    src = df["src"].values
    dst = df["dst"].values
    print("construct the graph")
    return dgl.graph((src, dst))


class OGBDataset(object):
    def __init__(self, g, num_labels, predict_category=None):
        self._g = g
        self._num_labels = num_labels
        self._predict_category = predict_category

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def num_classes(self):
        return self._num_labels

    @property
    def predict_category(self):
        return self._predict_category

    def __getitem__(self, idx):
        return self._g


def load_ogb_product():
    name = "ogbn-products"
    os.symlink("/tmp/dataset/", os.path.join(os.getcwd(), "dataset"))

    print("load", name)
    data = DglNodePropPredDataset(name=name)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["label"] = labels
    in_feats = graph.ndata["feat"].shape[1]
    num_labels = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))])
    )

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.num_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask

    return OGBDataset(graph, num_labels)


def load_ogb_mag():
    name = "ogbn-mag"
    os.symlink("/tmp/dataset/", os.path.join(os.getcwd(), "dataset"))

    print("load", name)
    dataset = DglNodePropPredDataset(name=name)
    print("finish loading", name)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]["paper"]
    val_idx = split_idx["valid"]["paper"]
    test_idx = split_idx["test"]["paper"]
    hg_orig, labels = dataset[0]
    subgs = {}
    for etype in hg_orig.canonical_etypes:
        u, v = hg_orig.all_edges(etype=etype)
        subgs[etype] = (u, v)
        subgs[(etype[2], "rev-" + etype[1], etype[0])] = (v, u)
    hg = dgl.heterograph(subgs)
    hg.nodes["paper"].data["feat"] = hg_orig.nodes["paper"].data["feat"]
    hg.nodes["paper"].data["labels"] = labels["paper"].squeeze()
    train_mask = torch.zeros((hg.num_nodes("paper"),), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros((hg.num_nodes("paper"),), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros((hg.num_nodes("paper"),), dtype=torch.bool)
    test_mask[test_idx] = True
    hg.nodes["paper"].data["train_mask"] = train_mask
    hg.nodes["paper"].data["val_mask"] = val_mask
    hg.nodes["paper"].data["test_mask"] = test_mask

    num_classes = dataset.num_classes
    return OGBDataset(hg, num_classes, "paper")


class PinsageDataset:
    def __init__(self, g, user_ntype, item_ntype, textset):
        self._g = g
        self._user_ntype = user_ntype
        self._item_ntype = item_ntype
        self._textset = textset

    @property
    def user_ntype(self):
        return self._user_ntype

    @property
    def item_ntype(self):
        return self._item_ntype

    @property
    def textset(self):
        return self._textset

    def __getitem__(self, idx):
        return self._g


def load_nowplaying_rs():
    import torchtext.legacy as torchtext

    # follow examples/pytorch/pinsage/README to create train_g.bin
    name = "train_g.bin"
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    os.symlink("/tmp/dataset/", dataset_dir)

    dataset_path = os.path.join(dataset_dir, "nowplaying_rs", name)
    g_list, _ = dgl.load_graphs(dataset_path)
    g = g_list[0]
    user_ntype = "user"
    item_ntype = "track"

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data["id"] = torch.arange(g.num_nodes(user_ntype))
    g.nodes[item_ntype].data["id"] = torch.arange(g.num_nodes(item_ntype))

    # Prepare torchtext dataset and vocabulary
    fields = {}
    examples = []
    for i in range(g.num_nodes(item_ntype)):
        example = torchtext.data.Example.fromlist([], [])
        examples.append(example)
    textset = torchtext.data.Dataset(examples, fields)

    return PinsageDataset(g, user_ntype, item_ntype, textset)


def process_data(name):
    if name == "cora":
        return dgl.data.CoraGraphDataset()
    elif name == "pubmed":
        return dgl.data.PubmedGraphDataset()
    elif name == "aifb":
        return dgl.data.AIFBDataset()
    elif name == "mutag":
        return dgl.data.MUTAGDataset()
    elif name == "bgs":
        return dgl.data.BGSDataset()
    elif name == "am":
        return dgl.data.AMDataset()
    elif name == "reddit":
        return dgl.data.RedditDataset(self_loop=True)
    elif name == "ogbn-products":
        return load_ogb_product()
    elif name == "ogbn-mag":
        return load_ogb_mag()
    elif name == "nowplaying_rs":
        return load_nowplaying_rs()
    else:
        raise ValueError("Invalid dataset name:", name)


def get_bench_device():
    device = os.environ.get("DGL_BENCH_DEVICE", "cpu")
    if device.lower() == "gpu":
        return "cuda:0"
    else:
        return device


def setup_track_time(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)


def setup_track_acc(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)


def setup_track_flops(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)


TRACK_UNITS = {
    "time": "s",
    "acc": "%",
    "flops": "GFLOPS",
}

TRACK_SETUP = {
    "time": setup_track_time,
    "acc": setup_track_acc,
    "flops": setup_track_flops,
}


def parametrize(param_name, params):
    """Decorator for benchmarking over a set of parameters.

    Parameters
    ----------
    param_name : str
        Parameter name. Must be one of the arguments of the decorated function.
    params : list[any]
        List of values to benchmark for the given parameter name. Recommend
        to use Python's native object type (e.g., int, str, list[int]) because
        ASV will display them on the plot.

    Examples
    --------

    Benchmark function `foo` when argument `x` is equal to 10 or 20.

    .. code::
        @benchmark('time')
        @parametrize('x', [10, 20]):
        def foo(x):
            pass

    Benchmark function with multiple parametrizations. It will run the function
    with all possible combinations. The example below generates 6 benchmarks.

    .. code::
        @benchmark('time')
        @parametrize('x', [10, 20]):
        @parametrize('y', [-1, -2, -3]):
        def foo(x, y):
            pass

    When using multiple parametrizations, it can have arbitrary order. The example
    below is the same as the above one.

    .. code::
        @benchmark('time')
        @parametrize('y', [-1, -2, -3]):
        @parametrize('x', [10, 20]):
        def foo(x, y):
            pass
    """

    def _wrapper(func):
        sig_params = inspect.signature(func).parameters.keys()
        num_params = len(sig_params)
        if getattr(func, "params", None) is None:
            func.params = [None] * num_params
        if getattr(func, "param_names", None) is None:
            func.param_names = [None] * num_params
        found_param = False
        for i, sig_param in enumerate(sig_params):
            if sig_param == param_name:
                func.params[i] = params
                func.param_names[i] = param_name
                found_param = True
                break
        if not found_param:
            raise ValueError("Invalid parameter name:", param_name)
        return func

    return _wrapper


def noop_decorator(param_name, params):
    """noop decorator"""

    def _wrapper(func):
        return func

    return _wrapper


class TestFilter:
    def __init__(self):
        self.conf = None
        if "DGL_REG_CONF" in os.environ:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(
                current_dir, "../../", os.environ["DGL_REG_CONF"]
            )
            with open(path, "r") as f:
                self.conf = json.load(f)
            if "INSTANCE_TYPE" in os.environ:
                instance_type = os.environ["INSTANCE_TYPE"]
            else:
                raise Exception(
                    "Must set both DGL_REG_CONF and INSTANCE_TYPE as env"
                )
            self.enabled_tests = self.conf[instance_type]["tests"]
        else:
            import logging

            logging.warning("No regression test conf file specified")

    def check(self, func):
        funcfullname = inspect.getmodule(func).__name__ + "." + func.__name__
        if self.conf is None:
            return True
        else:
            for enabled_testname in self.enabled_tests:
                if enabled_testname in funcfullname:
                    return True
            return False


filter = TestFilter()


device = os.environ.get("DGL_BENCH_DEVICE", "cpu")

if device == "cpu":
    parametrize_cpu = parametrize
    parametrize_gpu = noop_decorator
elif device == "gpu":
    parametrize_cpu = noop_decorator
    parametrize_gpu = parametrize
else:
    raise Exception(
        "Unknown device. Must be one of ['cpu', 'gpu'], but got {}".format(
            device
        )
    )


def skip_if_gpu():
    """skip if DGL_BENCH_DEVICE is gpu"""
    device = os.environ.get("DGL_BENCH_DEVICE", "cpu")

    def _wrapper(func):
        if device == "gpu":
            # skip if not enabled
            func.benchmark_name = "skip_" + func.__name__
        return func

    return _wrapper


def _cuda_device_count(q):
    import torch

    q.put(torch.cuda.device_count())


def get_num_gpu():
    import multiprocessing as mp

    q = mp.Queue()
    p = mp.Process(target=_cuda_device_count, args=(q,))
    p.start()
    p.join()
    return q.get(block=False)


GPU_COUNT = get_num_gpu()


def skip_if_not_4gpu():
    """skip if DGL_BENCH_DEVICE is gpu"""

    def _wrapper(func):
        if GPU_COUNT < 4:
            # skip if not enabled
            print("Skip {}".format(func.__name__))
            func.benchmark_name = "skip_" + func.__name__
        return func

    return _wrapper


def benchmark(track_type, timeout=60):
    """Decorator for indicating the benchmark type.

    Parameters
    ----------
    track_type : str
        Type. Must be either:

            - 'time' : For timing. Unit: second.
            - 'acc' : For accuracy. Unit: percentage, value between 0 and 100.
            - 'flops' : Unit: GFlops, number of floating point operations per second.
    timeout : int
        Timeout threshold in second.

    Examples
    --------

    .. code::
        @benchmark('time')
        def foo():
            pass
    """
    assert track_type in ["time", "acc", "flops"]

    def _wrapper(func):
        func.unit = TRACK_UNITS[track_type]
        func.setup = TRACK_SETUP[track_type]
        func.timeout = timeout
        if not filter.check(func):
            # skip if not enabled
            func.benchmark_name = "skip_" + func.__name__
        return func

    return _wrapper


#####################################
# Timer
#####################################


class Timer:
    def __init__(self, device=None):
        self.timer = default_timer
        if device is None:
            self.device = get_bench_device()
        else:
            self.device = device

    def __enter__(self):
        if self.device == "cuda:0":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cuda:0":
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = (
                self.start_event.elapsed_time(self.end_event) / 1e3
            )
        else:
            self.elapsed_secs = self.timer() - self.tic
