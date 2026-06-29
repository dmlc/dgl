"""API wrapping HCCL primitives for Huawei NPU (Ascend)."""

import torch
import torch.distributed as dist


class AscendNDArrayPartitionWrapper:
    """NPU-based partition wrapper that uses pure NPU operations.

    This wrapper provides partition functionality using torch operations
    on NPU, avoiding CPU data transfer.
    """

    def __init__(self, array_size, num_parts, mode="remainder", part_ranges=None):
        self._array_size = array_size
        self._num_parts = num_parts
        self.mode = mode
        self.part_ranges = part_ranges

    @property
    def num_parts(self):
        return self._num_parts

    @property
    def array_size(self):
        return self._array_size

    def local_size(self, part_id):
        if self.mode == 'remainder':
            return self._array_size // self._num_parts + (1 if part_id < self._array_size % self._num_parts else 0)
        elif self.mode == 'range':
            if self.part_ranges is not None:
                ranges = self.part_ranges.tolist() if hasattr(self.part_ranges, 'tolist') else list(self.part_ranges)
                return ranges[part_id + 1] - ranges[part_id]
            return self._array_size // self._num_parts
        return self._array_size

    @property
    def range(self):
        if self.mode == "range" and self.part_ranges is not None:
            if hasattr(self.part_ranges, 'tolist'):
                return self.part_ranges.tolist()
            return list(self.part_ranges)
        return None

    def generate_permutation(self, idx):
        num_parts = self._num_parts
        num_in = idx.shape[0]

        if num_parts == 1:
            perm = torch.arange(num_in, dtype=torch.int64, device=idx.device)
            splits = torch.ones(num_parts, dtype=torch.int64, device=idx.device) * num_in
            return perm, splits

        if self.mode == 'remainder':
            proc_ids = idx % num_parts
            sorted_proc_ids, perm = torch.sort(proc_ids, stable=True)
            counts = torch.bincount(proc_ids, minlength=num_parts)
            return perm, counts

        elif self.mode == 'range':
            if self.part_ranges is None:
                raise ValueError("part_ranges must be provided for range mode")

            if not torch.is_tensor(self.part_ranges):
                part_ranges_tensor = torch.tensor(
                    self.part_ranges, dtype=idx.dtype, device=idx.device
                )
            else:
                part_ranges_tensor = self.part_ranges

            if part_ranges_tensor.device != idx.device:
                part_ranges_tensor = part_ranges_tensor.to(idx.device)

            proc_ids = torch.bucketize(idx, part_ranges_tensor[1:], right=True)
            proc_ids = torch.clamp(proc_ids, max=num_parts - 1)

            sorted_proc_ids, perm = torch.sort(proc_ids, stable=True)
            counts = torch.bincount(proc_ids, minlength=num_parts)
            return perm, counts

        else:
            raise ValueError(f"Unknown partition mode: {self.mode}")

    def map_to_local(self, idxs):
        """Map global indices to local indices on NPU."""
        num_parts = self._num_parts
        device = idxs.device

        if self.mode == 'remainder':
            return (idxs // num_parts).long()
        elif self.mode == 'range':
            if self.part_ranges is None:
                raise ValueError("part_ranges must be provided for range mode")

            if not torch.is_tensor(self.part_ranges):
                part_ranges_tensor = torch.tensor(
                    self.part_ranges, dtype=idxs.dtype, device=device
                )
            else:
                part_ranges_tensor = self.part_ranges
                if part_ranges_tensor.device != device:
                    part_ranges_tensor = part_ranges_tensor.to(device)

            which_part = torch.bucketize(idxs, part_ranges_tensor[1:], right=True)
            which_part = torch.clamp(which_part, max=num_parts - 1)
            local_idx = idxs - part_ranges_tensor[which_part]
            return local_idx
        else:
            return idxs

    def map_to_global(self, idxs, part_id):
        """Map local indices to global indices on NPU."""
        num_parts = self._num_parts
        device = idxs.device

        if self.mode == 'remainder':
            return (idxs * num_parts + part_id).long()
        elif self.mode == 'range':
            if self.part_ranges is None:
                raise ValueError("part_ranges must be provided for range mode")

            if not torch.is_tensor(self.part_ranges):
                part_ranges_tensor = torch.tensor(
                    self.part_ranges, dtype=idxs.dtype, device=device
                )
            else:
                part_ranges_tensor = self.part_ranges
                if part_ranges_tensor.device != device:
                    part_ranges_tensor = part_ranges_tensor.to(device)

            return idxs + part_ranges_tensor[part_id]
        else:
            return idxs


def _is_npu_tensor(tensor):
    """Check if tensor is on NPU."""
    return tensor.device.type == 'npu'


def sparse_all_to_all_push(idx, value, partition):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return idx, value

    assert dist.get_backend() == "hccl", "requires HCCL backend for NPU communication."

    if not _is_npu_tensor(idx):
        idx = idx.to('npu')
    if not _is_npu_tensor(value):
        value = value.to('npu')

    perm, send_splits = partition.generate_permutation(idx)

    recv_splits = torch.empty_like(send_splits)
    dist.all_to_all_single(recv_splits, send_splits)

    send_idx = idx[perm]
    send_value = value[perm]

    recv_splits_list = recv_splits.tolist()
    send_splits_list = send_splits.tolist()
    recv_sum = sum(recv_splits_list)

    value_is_1d = value.dim() == 1
    send_value_2d = send_value.unsqueeze(1) if value_is_1d else send_value
    send_packed = torch.cat(
        [send_idx.to(send_value.dtype).unsqueeze(1), send_value_2d], dim=1
    )

    recv_packed = torch.empty(
        (recv_sum, 1 + send_value_2d.shape[1]),
        dtype=send_value.dtype, device=send_value.device
    )
    dist.all_to_all_single(recv_packed, send_packed, recv_splits_list, send_splits_list)

    recv_idx = recv_packed[:, 0].to(idx.dtype)
    recv_value = recv_packed[:, 1:].squeeze(1) if value_is_1d else recv_packed[:, 1:]

    return recv_idx, recv_value


def sparse_all_to_all_pull(req_idx, value, partition):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value[req_idx.long()]

    assert dist.get_backend() == "hccl", "requires HCCL backend for NPU communication."

    if not _is_npu_tensor(req_idx):
        req_idx = req_idx.to('npu')
    if not _is_npu_tensor(value):
        value = value.to('npu')

    perm, req_splits = partition.generate_permutation(req_idx)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    all_splits_list = [torch.empty(world_size, dtype=req_splits.dtype, device=req_splits.device) for _ in range(world_size)]
    dist.all_gather(all_splits_list, req_splits)
    all_splits = torch.stack(all_splits_list)
    resp_splits = all_splits[:, rank]

    req_idx_perm = req_idx[perm]

    resp_splits_list = resp_splits.tolist()
    req_splits_list = req_splits.tolist()
    resp_sum = sum(resp_splits_list)

    resp_idx = torch.empty(
        (resp_sum,), dtype=req_idx.dtype, device=req_idx.device
    )
    dist.all_to_all_single(resp_idx, req_idx_perm, resp_splits_list, req_splits_list)

    if resp_sum > 0:
        resp_idx = partition.map_to_local(resp_idx)

    req_value = torch.empty(
        (req_idx.size(0), *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_to_all_single(req_value, value[resp_idx], req_splits_list, resp_splits_list)

    return_value = torch.empty_like(req_value)
    return_value[perm] = req_value

    return return_value

def create_range_partition(global_size, world_size, device):
    """Create a range partition and its part_ranges tensor."""
    chunk_size = (global_size + world_size - 1) // world_size
    ranges = [min(i * chunk_size, global_size) for i in range(world_size + 1)]
    part_ranges = torch.tensor(ranges, dtype=torch.int64, device=device)
    partition = AscendNDArrayPartitionWrapper(
        global_size, world_size, mode="range", part_ranges=part_ranges
    )
    return partition, part_ranges