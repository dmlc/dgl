/*!
 *  Copyright (c) 2021 by Contributors
 * \file ndarray_partition.h
 * \brief Operations on partition implemented in CUDA.
 */

#include "../partition_op.h"

#include <dgl/runtime/device_api.h>

#include "../../array/cuda/dgl_cub.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/workspace.h"

using namespace dgl::runtime;

namespace dgl {
namespace partition {
namespace impl {

template<typename IdType> __global__ void _MapProcByRemainder(
    const IdType * const index,
    const int64_t num_index,
    const int64_t num_proc,
    IdType * const proc_id) {
  assert(num_index <= gridDim.x*blockDim.x);
  const int64_t idx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;

  if (idx < num_index) {
    proc_id[idx] = index[idx] % num_proc;
  }
}

template<typename IdType>
__global__ void _MapProcByMaskRemainder(
    const IdType * const index,
    const int64_t num_index,
    const IdType mask,
    IdType * const proc_id) {
  assert(num_index <= gridDim.x*blockDim.x);
  const int64_t idx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;

  if (idx < num_index) {
    proc_id[idx] = index[idx] & mask;
  }
}

template<typename IdType>
__global__ void _MapLocalIndexByRemainder(
    const IdType * const in,
    IdType * const out,
    const int64_t num_items,
    const int comm_size) {
  assert(num_items <= gridDim.x*blockDim.x);
  const int64_t idx = threadIdx.x+blockDim.x*blockIdx.x;

  if (idx < num_items) {
    out[idx] = in[idx] / comm_size;
  }
}

template<typename IdType>
__global__ void _MapGlobalIndexByRemainder(
    const IdType * const in,
    IdType * const out,
    const int part_id,
    const int64_t num_items,
    const int comm_size) {
  assert(num_items <= gridDim.x*blockDim.x);
  const int64_t idx = threadIdx.x+blockDim.x*blockIdx.x;

  assert(part_id < comm_size);

  if (idx < num_items) {
    out[idx] = (in[idx] * comm_size) + part_id;
  }
}

template <DLDeviceType XPU, typename IdType>
std::pair<IdArray, NDArray>
GeneratePermutationFromRemainder(
        int64_t array_size,
        int num_parts,
        IdArray in_idx) {
  std::pair<IdArray, NDArray> result;

  const auto& ctx = in_idx->ctx;
  auto device = DeviceAPI::Get(ctx);
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;

  const int64_t num_in = in_idx->shape[0];

  CHECK_GE(num_parts, 1) << "The number of partitions (" << num_parts <<
      ") must be at least 1.";
  if (num_parts == 1) {
    // no permutation
    result.first = aten::Range(0, num_in, sizeof(IdType)*8, ctx);
    result.second = aten::Full(num_in, num_parts, sizeof(int64_t)*8, ctx);

    return result;
  }

  result.first = aten::NewIdArray(num_in, ctx, sizeof(IdType)*8);
  result.second = aten::Full(0, num_parts, sizeof(int64_t)*8, ctx);
  int64_t * out_counts = static_cast<int64_t*>(result.second->data);
  if (num_in == 0) {
    // now that we've zero'd out_counts, nothing left to do for an empty
    // mapping
    return result;
  }

  const int64_t part_bits =
      static_cast<int64_t>(std::ceil(std::log2(num_parts)));

  // First, generate a mapping of indexes to processors
  Workspace<IdType> proc_id_in(device, ctx, num_in);
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    if (num_parts < (1 << part_bits)) {
      // num_parts is not a power of 2
      CUDA_KERNEL_CALL(_MapProcByRemainder, grid, block, 0, stream,
          static_cast<const IdType*>(in_idx->data),
          num_in,
          num_parts,
          proc_id_in.get());
    } else {
      // num_parts is a power of 2
      CUDA_KERNEL_CALL(_MapProcByMaskRemainder, grid, block, 0, stream,
          static_cast<const IdType*>(in_idx->data),
          num_in,
          static_cast<IdType>(num_parts-1),  // bit mask
          proc_id_in.get());
    }
  }

  // then create a permutation array that groups processors together by
  // performing a radix sort
  Workspace<IdType> proc_id_out(device, ctx, num_in);
  IdType * perm_out = static_cast<IdType*>(result.first->data);
  {
    IdArray perm_in = aten::Range(0, num_in, sizeof(IdType)*8, ctx);

    size_t sort_workspace_size;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, sort_workspace_size,
        proc_id_in.get(), proc_id_out.get(), static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, part_bits, stream));

    Workspace<void> sort_workspace(device, ctx, sort_workspace_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(sort_workspace.get(), sort_workspace_size,
        proc_id_in.get(), proc_id_out.get(), static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, part_bits, stream));
  }
  // explicitly free so workspace can be re-used
  proc_id_in.free();

  // perform a histogram and then prefixsum on the sorted proc_id vector

  // Count the number of values to be sent to each processor
  {
    using AtomicCount = unsigned long long; // NOLINT
    static_assert(sizeof(AtomicCount) == sizeof(*out_counts),
        "AtomicCount must be the same width as int64_t for atomicAdd "
        "in cub::DeviceHistogram::HistogramEven() to work");

    // TODO(dlasalle): Once https://github.com/NVIDIA/cub/pull/287 is merged,
    // add a compile time check against the cub version to allow
    // num_in > (2 << 31).
    CHECK(num_in < static_cast<int64_t>(std::numeric_limits<int>::max())) <<
        "number of values to insert into histogram must be less than max "
        "value of int.";

    size_t hist_workspace_size;
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        nullptr,
        hist_workspace_size,
        proc_id_out.get(),
        reinterpret_cast<AtomicCount*>(out_counts),
        num_parts+1,
        static_cast<IdType>(0),
        static_cast<IdType>(num_parts+1),
        static_cast<int>(num_in),
        stream));

    Workspace<void> hist_workspace(device, ctx, hist_workspace_size);
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        hist_workspace.get(),
        hist_workspace_size,
        proc_id_out.get(),
        reinterpret_cast<AtomicCount*>(out_counts),
        num_parts+1,
        static_cast<IdType>(0),
        static_cast<IdType>(num_parts+1),
        static_cast<int>(num_in),
        stream));
  }

  return result;
}


template std::pair<IdArray, IdArray>
GeneratePermutationFromRemainder<kDLGPU, int32_t>(
        int64_t array_size,
        int num_parts,
        IdArray in_idx);
template std::pair<IdArray, IdArray>
GeneratePermutationFromRemainder<kDLGPU, int64_t>(
        int64_t array_size,
        int num_parts,
        IdArray in_idx);


template <DLDeviceType XPU, typename IdType>
IdArray MapToLocalFromRemainder(
    const int num_parts,
    IdArray global_idx) {
  const auto& ctx = global_idx->ctx;
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;

  if (num_parts > 1) {
    IdArray local_idx = aten::NewIdArray(global_idx->shape[0], ctx,
        sizeof(IdType)*8);

    const dim3 block(128);
    const dim3 grid((global_idx->shape[0] +block.x-1)/block.x);

    CUDA_KERNEL_CALL(
        _MapLocalIndexByRemainder,
        grid,
        block,
        0,
        stream,
        static_cast<const IdType*>(global_idx->data),
        static_cast<IdType*>(local_idx->data),
        global_idx->shape[0],
        num_parts);

    return local_idx;
  } else {
    // no mapping to be done
    return global_idx;
  }
}

template IdArray
MapToLocalFromRemainder<kDLGPU, int32_t>(
        int num_parts,
        IdArray in_idx);
template IdArray
MapToLocalFromRemainder<kDLGPU, int64_t>(
        int num_parts,
        IdArray in_idx);

template <DLDeviceType XPU, typename IdType>
IdArray MapToGlobalFromRemainder(
    const int num_parts,
    IdArray local_idx,
    const int part_id) {
  CHECK_LT(part_id, num_parts) << "Invalid partition id " << part_id <<
      "/" << num_parts;
  CHECK_GE(part_id, 0) << "Invalid partition id " << part_id <<
      "/" << num_parts;

  const auto& ctx = local_idx->ctx;
  cudaStream_t stream = CUDAThreadEntry::ThreadLocal()->stream;

  if (num_parts > 1) {
    IdArray global_idx = aten::NewIdArray(local_idx->shape[0], ctx,
        sizeof(IdType)*8);

    const dim3 block(128);
    const dim3 grid((local_idx->shape[0] +block.x-1)/block.x);

    CUDA_KERNEL_CALL(
        _MapGlobalIndexByRemainder,
        grid,
        block,
        0,
        stream,
        static_cast<const IdType*>(local_idx->data),
        static_cast<IdType*>(global_idx->data),
        part_id,
        global_idx->shape[0],
        num_parts);

    return global_idx;
  } else {
    // no mapping to be done
    return local_idx;
  }
}

template IdArray
MapToGlobalFromRemainder<kDLGPU, int32_t>(
        int num_parts,
        IdArray in_idx,
        int part_id);
template IdArray
MapToGlobalFromRemainder<kDLGPU, int64_t>(
        int num_parts,
        IdArray in_idx,
        int part_id);



}  // namespace impl
}  // namespace partition
}  // namespace dgl
