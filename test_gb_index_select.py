import itertools
import os
import time
from enum import Enum

import dgl
import dgl.graphbolt as gb
import torch
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset


def get_ogbn_graph_test_indices(name):
    dataset = AsNodePredDataset(DglNodePropPredDataset(name))
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
    )
    train_idx = dataset.train_idx
    g = dataset[0]
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    ret_indices = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        ret_indices.append(input_nodes)
    feature = torch.rand(g.number_of_nodes(), g.ndata["feat"].shape[1]).float()
    return feature, ret_indices


class Device(Enum):
    GPU = 0
    Pinned = 1


def gen_random_testcase(n_rows, feat_size, num_indices):
    feature = torch.rand(n_rows, feat_size).float()
    indices = []
    for i in range(10):
        indices.append(torch.randint(0, n_rows, (num_indices,)))
    return feature, indices


def test_index_select_throughput(
    feature, indices, feature_device, indices_device
):

    feature = (
        feature.cuda() if feature_device == Device.GPU else feature.pin_memory()
    )
    indices = [
        index.cuda() if indices_device == Device.GPU else index.pin_memory()
        for index in indices
    ]

    # Warm up
    for _ in range(3):
        for index in indices:
            torch.ops.graphbolt.index_select(feature, index)
    torch.cuda.synchronize()
    n_iter = 5
    start = time.time()
    for _ in range(n_iter):
        for index in indices:
            torch.ops.graphbolt.index_select(feature, index)
    torch.cuda.synchronize()
    end = time.time()
    # Summarize all index sizes
    num_indices = sum([index.numel() for index in indices])
    average_time = (end - start) / n_iter
    feat_size = feature.shape[1]
    selected_size = num_indices * feat_size * 4 / 1024 / 1024
    # print(f">>> throughput: {selected_size / average_time} MB/s")
    return selected_size / average_time


n_rows = [100000]
feat_size = [1, 4, 16, 100, 300]
num_indices = [1, 1000, 10000]
feature_devices = [Device.Pinned]
indices_devices = [Device.GPU]
keys = [
    "n_rows",
    "feat_size",
    "num_indices",
    "feature_device",
    "indices_device",
]

use_perm_env = ["USE_PERM=1", "USE_PERM=0"]
use_single_env = ["USE_SINGLE=1", "USE_SINGLE=0"]
use_torch_sort_env = ["USE_TORCH_SORT=1", "USE_TORCH_SORT=0"]
use_align_env = ["USE_ALIGN=1", "USE_ALIGN=0"]


def _print_throughput(throughputs, envs1, envs2):
    col_width = 12
    print(
        f"|{'':<{col_width}}|{envs2[0]:^{col_width}}|{envs2[1]:^{col_width}}|"
    )
    print(
        "|"
        + "-" * col_width
        + "|"
        + "-" * col_width
        + "|"
        + "-" * col_width
        + "|"
    )
    print(
        f"|{envs1[0]:<{col_width}}|{throughputs[0]:>{col_width}.3f}|{throughputs[1]:>{col_width}.3f}|"
    )
    print(
        f"|{envs1[1]:<{col_width}}|{throughputs[2]:>{col_width}.3f}|{throughputs[3]:>{col_width}.3f}|"
    )


def test_random(envs1, envs2):
    for params in itertools.product(
        n_rows, feat_size, num_indices, feature_devices, indices_devices
    ):
        feature, indices = gen_random_testcase(params[0], params[1], params[2])
        print(
            "* params: ",
            ", ".join([f"{k}={v}" for k, v in zip(keys, params)]),
        )
        print("")
        throughputs = []
        params_dict = {
            "feature": feature,
            "indices": indices,
            "feature_device": params[3],
            "indices_device": params[4],
        }
        for envs in itertools.product(envs1, envs2):
            for env in envs:
                os.environ[env.split("=")[0]] = env.split("=")[1]
            throughputs.append(test_index_select_throughput(**params_dict))
        _print_throughput(throughputs, envs1, envs2)


def test_ogb(name, envs1, envs2):
    feature, indices = get_ogbn_graph_test_indices(name)
    for devices in itertools.product(feature_devices, indices_devices):
        print(
            "* params: ",
            f"dataset={name}",
            f"n_rows={feature.shape[0]}",
            f"feat_size={feature.shape[1]}",
            f"num_indices={len(indices)}",
            f"feature_device={devices[0]}, indices_device={devices[1]}",
        )
        print("")
        throughputs = []
        for envs in itertools.product(envs1, envs2):
            for env in envs:
                os.environ[env.split("=")[0]] = env.split("=")[1]
            throughputs.append(
                test_index_select_throughput(
                    feature, indices, devices[0], devices[1]
                )
            )
        _print_throughput(throughputs, envs1, envs2)


def check_correctness(name, envs):
    feature, indices = get_ogbn_graph_test_indices(name)
    feature = feature.cuda()
    indices = [index.cuda() for index in indices]
    for index in indices:
        os.environ[envs[0].split("=")[0]] = envs[0].split("=")[1]
        res1 = torch.ops.graphbolt.index_select(feature, index)
        os.environ[envs[1].split("=")[0]] = envs[1].split("=")[1]
        res2 = torch.ops.graphbolt.index_select(feature, index)
        assert torch.allclose(res1, res2)
    print(">>> Correctness check passed!")


os.environ["USE_PERM"] = "1"
os.environ["USE_SINGLE"] = "1"
os.environ["USE_TORCH_SORT"] = "0"
os.environ["USE_ALIGN"] = "1"
# check_correctness("ogbn-arxiv", use_torch_sort_env)
# test_ogb("ogbn-products", use_perm_env, use_align_env)
# test_ogb("ogbn-arxiv", use_perm_env, use_align_env)
test_random(use_perm_env, use_align_env)
