import itertools
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
    feature = torch.randint(
        0, 13, (g.number_of_nodes(), g.ndata["feat"].shape[1])
    )
    return feature, ret_indices


class Device(Enum):
    GPU = 0
    Pinned = 1


def gen_random_indices(n_rows, num_indices):
    indices = []
    for i in range(50):
        indices.append(torch.randint(0, n_rows, (num_indices,)))
    return indices


def test_index_select_throughput(feature, indices):
    # Warm up
    for _ in range(3):
        for index in indices:
            torch.ops.graphbolt.index_select(feature, index)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in indices:
        torch.ops.graphbolt.index_select(feature, index)
    end.record()
    end.synchronize()
    # Summarize all index sizes
    num_indices = sum([index.numel() for index in indices]) / len(indices)
    average_time = start.elapsed_time(end) / 1000 / len(indices)
    feat_size = feature.shape[1]
    selected_size = num_indices * feat_size * feature.element_size()
    return average_time, selected_size / average_time


available_RAM = 10 * (2**30)  ## 10 GiB
n_rows = [2000000, 20000000, 200000000]
feat_size = [1, 4, 47, 256, 353]
num_indices = [1, 1000, 100000, 1000000]
dtypes = [torch.float32, torch.int8]
feature_devices = [Device.Pinned]
indices_devices = [Device.GPU]
keys = [
    "n_rows",
    "feat_size",
    "dtype",
    "num_indices",
    "feature_device",
    "indices_device",
]

sum_of_runtimes = 0


def _print_result(runtime, throughput):
    print(
        f"Runtime in us: {int(runtime * 1000000)}, Throughput in MB/s: {int(throughput / (2 ** 20))}"
    )
    print("")
    print("")
    global sum_of_runtimes
    sum_of_runtimes += runtime


def test_random():
    for rows, size, feature_device, dtype in itertools.product(
        n_rows, feat_size, feature_devices, dtypes
    ):
        if (
            rows * size * torch.tensor([], dtype=dtype).element_size()
            >= available_RAM
        ):
            continue
        feature = torch.randint(0, 13, size=[rows, size], dtype=dtype)
        feature = (
            feature.cuda()
            if feature_device == Device.GPU
            else feature.pin_memory()
        )
        for indices_size, indices_device in itertools.product(
            num_indices, indices_devices
        ):
            indices = gen_random_indices(rows, indices_size)
            indices = [
                index.cuda()
                if indices_device == Device.GPU
                else index.pin_memory()
                for index in indices
            ]
            params = (
                rows,
                size,
                dtype,
                indices_size,
                feature_device,
                indices_device,
            )
            print(
                "* params: ",
                ", ".join([f"{k}={v}" for k, v in zip(keys, params)]),
            )
            print("")
            params_dict = {
                "feature": feature,
                "indices": indices,
            }
            runtime, throughput = test_index_select_throughput(**params_dict)
            _print_result(runtime, throughput)


def test_ogb(name):
    original_feature, original_indices = get_ogbn_graph_test_indices(name)
    for feature_device, dtype in zip(feature_devices, dtypes):
        feature = original_feature.to(dtype)
        feature = (
            feature.cuda()
            if feature_device == Device.GPU
            else feature.pin_memory()
        )
        for indices_device in indices_devices:
            indices = [
                index.cuda()
                if indices_device == Device.GPU
                else index.pin_memory()
                for index in original_indices
            ]
            print(
                "* params: ",
                f"dataset={name}",
                f"n_rows={feature.shape[0]}",
                f"feat_size={feature.shape[1]}",
                f"dtype={dtype}",
                f"num_indices={int(sum(idx.numel() for idx in indices) / len(indices))}",
                f"feature_device={feature_device}, indices_device={indices_device}",
            )
            print("")
            runtime, throughput = test_index_select_throughput(feature, indices)
            _print_result(runtime, throughput)


test_ogb("ogbn-products")
test_random()
print("Total runtimes in us: ", int(sum_of_runtimes * 1000000))
