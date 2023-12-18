import itertools
from enum import Enum

import dgl
import dgl.graphbolt as gb
import torch


class Device(Enum):
    GPU = 0
    Pinned = 1


def torch_device(device):
    return torch.device("cuda" if device == Device.GPU else "cpu")


class Regularity(Enum):
    REGULAR = 0
    IRREGULAR = 1


def gen_random_graph(
    n_rows,
    avg_degree,
    indptr_dtype,
    indices_dtype,
    regularity,
    indptr_device,
    tensor_device,
):
    indptr_device = torch_device(indptr_device)
    tensor_device = torch_device(tensor_device)
    if regularity == Regularity.IRREGULAR:
        degree = torch.randint(
            0,
            avg_degree * 2 + 1,
            (n_rows,),
            dtype=indptr_dtype,
            device=indptr_device,
        )
    else:
        degree = (
            torch.ones(n_rows, dtype=indptr_dtype, device=indptr_device)
            * avg_degree
        )

    indptr = torch.empty(n_rows + 1, dtype=degree.dtype, device=indptr_device)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(degree, dim=0)
    num_edges = indptr[-1]
    indices = torch.randint(
        0, n_rows, (num_edges,), dtype=indices_dtype, device=tensor_device
    )
    if not indptr.is_cuda:
        indptr = indptr.pin_memory()
    if not indices.is_cuda:
        indices = indices.pin_memory()
    return indptr, indices


def gen_random_indices(n_rows, num_indices):
    indices = []
    weights = torch.ones(n_rows)
    for _ in range(50):
        indices.append(
            torch.multinomial(weights, num_indices, replacement=False)
        )
    return indices


def test_unique_and_compact_throughput(graph, indices):
    indptr, tensor = graph
    problems = []
    problems1 = []
    problems2 = []
    for index in indices:
        subindptr, subindices = torch.ops.graphbolt.index_select_csc(
            indptr, tensor, index
        )
        index = index.to(subindices.dtype)
        dst_idx = -1 + torch.searchsorted(
            subindptr,
            torch.arange(0, subindices.shape[0], device=subindices.device),
            right=True,
        )
        dst = index[dst_idx]
        src = subindices
        problems.append((src, dst, index))
        problems1.append((gb.CSCFormatBase(subindptr, src), index))
        problems2.append((dgl.graph((src, dst)), index))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for src, dst, index in problems:
        torch.ops.graphbolt.unique_and_compact(src, dst, index)
    end.record()
    end.synchronize()
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    start1.record()
    for g, index in problems1:
        gb.unique_and_compact_csc_formats(g, index)
    end1.record()
    end1.synchronize()
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for g, index in problems2:
        dgl.to_block(g, index)
    end2.record()
    end2.synchronize()
    # Summarize all index sizes
    # num_indices = sum([index.numel() for index in indices]) / len(indices)
    num_items_copied = sum(
        [
            torch.sum(
                indptr[index.to(indptr.device) + 1]
                - indptr[index.to(indptr.device)]
            ).item()
            for index in indices
        ]
    ) / len(indices)
    average_time = start.elapsed_time(end) / 1000 / len(indices)
    average_time1 = start1.elapsed_time(end1) / 1000 / len(indices)
    average_time2 = start2.elapsed_time(end2) / 1000 / len(indices)
    selected_size = 2 * num_items_copied * tensor.element_size()
    return (
        (average_time, selected_size / average_time),
        (average_time1, selected_size / average_time1),
        (average_time2, selected_size / average_time2),
    )


available_RAM = 10 * (2**30)  ## 10 GiB
n_rows = [2**24]
avg_degrees = [8, 16, 32]
num_indices = [1000, 100000, 1000000]
indptr_dtypes = [torch.int64]
tensor_dtypes = [torch.int32]
graph_devices = [Device.Pinned]
indices_devices = [Device.GPU]
regularity = [Regularity.IRREGULAR]
keys = [
    "n_rows",
    "avg_degree",
    "indptr_dtype",
    "tensor_dtype",
    "num_indices",
    "indptr_device",
    "tensor_device",
    "indices_device",
    "regular",
]
names = [
    "torch.ops.graphbolt.unique_and_compact",
    "gb.unique_and_compact_csc_formats",
    "dgl.to_block",
]

sum_of_runtimes = {k: 0 for k in names}


def _print_result(runtime, throughput, name):
    runtime = int(runtime * 1000000)
    print(
        f"{name} --- Runtime in us: {runtime}, Throughput in MiB/s: {int(throughput / (2 ** 20))}"
    )
    print("")
    print("")
    global sum_of_runtimes
    sum_of_runtimes[name] += runtime


def test_random():
    for (
        rows,
        avg_degree,
        graph_device,
        indptr_dtype,
        tensor_dtype,
        regular,
    ) in itertools.product(
        n_rows,
        avg_degrees,
        graph_devices,
        indptr_dtypes,
        tensor_dtypes,
        regularity,
    ):
        if (rows + 1) * torch.tensor(
            [], dtype=indptr_dtype
        ).element_size() + rows * avg_degree * torch.tensor(
            [], dtype=tensor_dtype
        ).element_size() >= available_RAM:
            continue
        torch.cuda.empty_cache()
        graph = gen_random_graph(
            rows,
            avg_degree,
            indptr_dtype,
            tensor_dtype,
            regular,
            graph_device,
            graph_device,
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
                avg_degree,
                indptr_dtype,
                tensor_dtype,
                indices_size,
                graph_device,
                graph_device,
                indices_device,
                regular,
            )
            print(
                "* params: ",
                ", ".join([f"{k}={v}" for k, v in zip(keys, params)]),
            )
            print("")
            params_dict = {
                "graph": graph,
                "indices": indices,
            }
            (
                (runtime, throughput),
                (runtime1, throughput1),
                (runtime2, throughput2),
            ) = test_unique_and_compact_throughput(**params_dict)
            _print_result(runtime, throughput, names[0])
            _print_result(runtime1, throughput1, names[1])
            _print_result(runtime2, throughput2, names[2])


test_random()
print("Total runtimes in us: ", sum_of_runtimes)
