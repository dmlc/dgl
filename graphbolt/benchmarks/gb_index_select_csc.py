import itertools
from enum import Enum

import dgl
import dgl.graphbolt as gb
import torch


class Device(Enum):
    GPU = 0
    Pinned = 1

def torch_device(device):
    return torch.device('cuda' if device == Device.GPU else 'cpu')

class Regularity(Enum):
    REGULAR = 0
    IRREGULAR = 1

def gen_random_graph(n_rows, avg_degree, indptr_dtype, indices_dtype, regularity, indptr_device, tensor_device):
    indptr_device = torch_device(indptr_device)
    tensor_device = torch_device(tensor_device)
    if regularity == Regularity.IRREGULAR:
        degree = torch.randint(0, avg_degree * 2 + 1, (n_rows,), dtype=indptr_dtype, device=indptr_device)
    else:
        degree = torch.ones(n_rows, dtype=indptr_dtype, device=indptr_device) * avg_degree

    indptr = torch.empty(n_rows + 1, dtype=degree.dtype, device=indptr_device)
    indptr[0] = 0
    indptr[1:] = torch.cumsum(degree, dim=0)
    num_edges = indptr[-1]
    indices = torch.randint(0, n_rows, (num_edges,), dtype=indices_dtype, device=tensor_device)
    if not indptr.is_cuda:
        indptr = indptr.pin_memory()
    if not indices.is_cuda:
        indices = indices.pin_memory()
    return indptr, indices


def gen_random_indices(n_rows, num_indices):
    indices = []
    for _ in range(50):
        indices.append(torch.randint(0, n_rows, (num_indices,)))
    return indices


def test_index_select_csc_throughput(graph, indices):
    indptr, tensor = graph
    # Warm up
    for _ in range(3):
        for index in indices:
            torch.ops.graphbolt.index_select_csc(indptr, tensor, index)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in indices:
        torch.ops.graphbolt.index_select_csc(indptr, tensor, index)
    end.record()
    end.synchronize()
    # Summarize all index sizes
    num_indices = sum([index.numel() for index in indices]) / len(indices)
    num_items_copied = sum([torch.sum(indptr[index.to(indptr.device) + 1] - indptr[index.to(indptr.device)]).item() for index in indices]) / len(indices)
    average_time = start.elapsed_time(end) / 1000 / len(indices)
    selected_size = num_indices * 2 * indptr.element_size() + num_items_copied * tensor.element_size()
    return average_time, selected_size / average_time


available_RAM = 10 * (2**30)  ## 10 GiB
n_rows = [2000000 * factor for factor in [1, 10, 100, 1000]]
avg_degrees = [8, 64]
num_indices = [1000, 100000, 1000000]
indptr_dtypes = [torch.int64]
tensor_dtypes = [torch.int32, torch.int64]
graph_devices = [Device.GPU, Device.Pinned]
tensor_devices = [Device.GPU, Device.Pinned]
indices_devices = [Device.GPU]
regularity = [Regularity.IRREGULAR, Regularity.REGULAR]
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

sum_of_runtimes = {k: 0 for k in graph_devices}

def _print_result(runtime, throughput, graph_device):
    runtime = int(runtime * 1000000)
    print(
        f"Runtime in us: {runtime}, Throughput in MB/s: {int(throughput / (2 ** 20))}"
    )
    print("")
    print("")
    global sum_of_runtimes
    sum_of_runtimes[graph_device] += runtime


def test_random():
    for rows, avg_degree, graph_device, indptr_dtype, tensor_dtype, regular in itertools.product(
        n_rows, avg_degrees, graph_devices, indptr_dtypes, tensor_dtypes, regularity
    ):
        if (
            (rows + 1) * torch.tensor([], dtype=indptr_dtype).element_size() + rows * avg_degree * torch.tensor([], dtype=tensor_dtype).element_size()
            >= available_RAM
        ):
            continue
        graph = gen_random_graph(rows, avg_degree, indptr_dtype, tensor_dtype, regular, graph_device, graph_device)
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
            runtime, throughput = test_index_select_csc_throughput(**params_dict)
            _print_result(runtime, throughput, graph_device)


test_random()
print("Total runtimes in us: ", sum_of_runtimes)