/**
 *  Copyright (c) 2021 by Contributors
 * @file ndarray_partition.h
 * @brief Operations on partition implemented in CUDA.
 */

#include <dgl/runtime/device_api.h>

#include <cub/cub.cuh>

#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/workspace.h"
#include "../partition_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace partition {
namespace impl {

namespace {

/**
 * @brief Kernel to map global element IDs to partition IDs by remainder.
 *
 * @tparam IdType The type of ID.
 * @param global The global element IDs.
 * @param num_elements The number of element IDs.
 * @param num_parts The number of partitions.
 * @param part_id The mapped partition ID (outupt).
 */
template <typename IdType>
__global__ void _MapProcByRemainderKernel(
    const IdType* const global, const int64_t num_elements,
    const int64_t num_parts, IdType* const part_id) {
  assert(num_elements <= gridDim.x * blockDim.x);
  const int64_t idx =
      blockDim.x * static_cast<int64_t>(blockIdx.x) + threadIdx.x;

  if (idx < num_elements) {
    part_id[idx] = global[idx] % num_parts;
  }
}

/**
 * @brief Kernel to map global element IDs to partition IDs, using a bit-mask.
 * The number of partitions must be a power a two.
 *
 * @tparam IdType The type of ID.
 * @param global The global element IDs.
 * @param num_elements The number of element IDs.
 * @param mask The bit-mask with 1's for each bit to keep from the element ID to
 * extract the partition ID (e.g., an 8 partition mask would be 0x07).
 * @param part_id The mapped partition ID (outupt).
 */
template <typename IdType>
__global__ void _MapProcByMaskRemainderKernel(
    const IdType* const global, const int64_t num_elements, const IdType mask,
    IdType* const part_id) {
  assert(num_elements <= gridDim.x * blockDim.x);
  const int64_t idx =
      blockDim.x * static_cast<int64_t>(blockIdx.x) + threadIdx.x;

  if (idx < num_elements) {
    part_id[idx] = global[idx] & mask;
  }
}

/**
 * @brief Kernel to map global element IDs to local element IDs.
 *
 * @tparam IdType The type of ID.
 * @param global The global element IDs.
 * @param num_elements The number of IDs.
 * @param num_parts The number of partitions.
 * @param local The local element IDs (output).
 */
template <typename IdType>
__global__ void _MapLocalIndexByRemainderKernel(
    const IdType* const global, const int64_t num_elements, const int num_parts,
    IdType* const local) {
  assert(num_elements <= gridDim.x * blockDim.x);
  const int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < num_elements) {
    local[idx] = global[idx] / num_parts;
  }
}

/**
 * @brief Kernel to map local element IDs within a partition to their global
 * IDs, using the remainder over the number of partitions.
 *
 * @tparam IdType The type of ID.
 * @param local The local element IDs.
 * @param part_id The partition to map local elements from.
 * @param num_elements The number of elements to map.
 * @param num_parts The number of partitions.
 * @param global The global element IDs (output).
 */
template <typename IdType>
__global__ void _MapGlobalIndexByRemainderKernel(
    const IdType* const local, const int part_id, const int64_t num_elements,
    const int num_parts, IdType* const global) {
  assert(num_elements <= gridDim.x * blockDim.x);
  const int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  assert(part_id < num_parts);

  if (idx < num_elements) {
    global[idx] = (local[idx] * num_parts) + part_id;
  }
}

/**
 * @brief Device function to perform a binary search to find to which partition
 * a given ID belongs.
 *
 * @tparam RangeType The type of range.
 * @param range The prefix-sum of IDs assigned to partitions.
 * @param num_parts The number of partitions.
 * @param target The element ID to find the partition of.
 *
 * @return The partition.
 */
template <typename RangeType>
__device__ RangeType _SearchRange(
    const RangeType* const range, const int num_parts, const RangeType target) {
  int start = 0;
  int end = num_parts;
  int cur = (end + start) / 2;

  assert(range[0] == 0);
  assert(target < range[num_parts]);

  while (start + 1 < end) {
    if (target < range[cur]) {
      end = cur;
    } else {
      start = cur;
    }
    cur = (start + end) / 2;
  }

  return cur;
}

/**
 * @brief Kernel to map element IDs to partition IDs.
 *
 * @tparam IdType The type of element ID.
 * @tparam RangeType The type of of the range.
 * @param range The prefix-sum of IDs assigned to partitions.
 * @param global The global element IDs.
 * @param num_elements The number of element IDs.
 * @param num_parts The number of partitions.
 * @param part_id The partition ID assigned to each element (output).
 */
template <typename IdType, typename RangeType>
__global__ void _MapProcByRangeKernel(
    const RangeType* const range, const IdType* const global,
    const int64_t num_elements, const int64_t num_parts,
    IdType* const part_id) {
  assert(num_elements <= gridDim.x * blockDim.x);
  const int64_t idx =
      blockDim.x * static_cast<int64_t>(blockIdx.x) + threadIdx.x;

  // rely on caching to load the range into L1 cache
  if (idx < num_elements) {
    part_id[idx] = static_cast<IdType>(_SearchRange(
        range, static_cast<int>(num_parts),
        static_cast<RangeType>(global[idx])));
  }
}

/**
 * @brief Kernel to map global element IDs to their ID within their respective
 * partition.
 *
 * @tparam IdType The type of element ID.
 * @tparam RangeType The type of the range.
 * @param range The prefix-sum of IDs assigned to partitions.
 * @param global The global element IDs.
 * @param num_elements The number of elements.
 * @param num_parts The number of partitions.
 * @param local The local element IDs (output).
 */
template <typename IdType, typename RangeType>
__global__ void _MapLocalIndexByRangeKernel(
    const RangeType* const range, const IdType* const global,
    const int64_t num_elements, const int num_parts, IdType* const local) {
  assert(num_elements <= gridDim.x * blockDim.x);
  const int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  // rely on caching to load the range into L1 cache
  if (idx < num_elements) {
    const int proc = _SearchRange(
        range, static_cast<int>(num_parts),
        static_cast<RangeType>(global[idx]));
    local[idx] = global[idx] - range[proc];
  }
}

/**
 * @brief Kernel to map local element IDs within a partition to their global
 * IDs.
 *
 * @tparam IdType The type of ID.
 * @tparam RangeType The type of the range.
 * @param range The prefix-sum of IDs assigend to partitions.
 * @param local The local element IDs.
 * @param part_id The partition to map local elements from.
 * @param num_elements The number of elements to map.
 * @param num_parts The number of partitions.
 * @param global The global element IDs (output).
 */
template <typename IdType, typename RangeType>
__global__ void _MapGlobalIndexByRangeKernel(
    const RangeType* const range, const IdType* const local, const int part_id,
    const int64_t num_elements, const int num_parts, IdType* const global) {
  assert(num_elements <= gridDim.x * blockDim.x);
  const int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  assert(part_id < num_parts);

  // rely on caching to load the range into L1 cache
  if (idx < num_elements) {
    global[idx] = local[idx] + range[part_id];
  }
}
}  // namespace

// Remainder Based Partition Operations

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, NDArray> GeneratePermutationFromRemainder(
    int64_t array_size, int num_parts, IdArray in_idx) {
  std::pair<IdArray, NDArray> result;

  const auto& ctx = in_idx->ctx;
  auto device = DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_in = in_idx->shape[0];

  CHECK_GE(num_parts, 1) << "The number of partitions (" << num_parts
                         << ") must be at least 1.";
  if (num_parts == 1) {
    // no permutation
    result.first = aten::Range(0, num_in, sizeof(IdType) * 8, ctx);
    result.second = aten::Full(num_in, num_parts, sizeof(int64_t) * 8, ctx);

    return result;
  }

  result.first = aten::NewIdArray(num_in, ctx, sizeof(IdType) * 8);
  result.second = aten::Full(0, num_parts, sizeof(int64_t) * 8, ctx);
  int64_t* out_counts = static_cast<int64_t*>(result.second->data);
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
    const dim3 grid((num_in + block.x - 1) / block.x);

    if (num_parts < (1 << part_bits)) {
      // num_parts is not a power of 2
      CUDA_KERNEL_CALL(
          _MapProcByRemainderKernel, grid, block, 0, stream,
          static_cast<const IdType*>(in_idx->data), num_in, num_parts,
          proc_id_in.get());
    } else {
      // num_parts is a power of 2
      CUDA_KERNEL_CALL(
          _MapProcByMaskRemainderKernel, grid, block, 0, stream,
          static_cast<const IdType*>(in_idx->data), num_in,
          static_cast<IdType>(num_parts - 1),  // bit mask
          proc_id_in.get());
    }
  }

  // then create a permutation array that groups processors together by
  // performing a radix sort
  Workspace<IdType> proc_id_out(device, ctx, num_in);
  IdType* perm_out = static_cast<IdType*>(result.first->data);
  {
    IdArray perm_in = aten::Range(0, num_in, sizeof(IdType) * 8, ctx);

    size_t sort_workspace_size;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_workspace_size, proc_id_in.get(), proc_id_out.get(),
        static_cast<IdType*>(perm_in->data), perm_out, num_in, 0, part_bits,
        stream));

    Workspace<void> sort_workspace(device, ctx, sort_workspace_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
        sort_workspace.get(), sort_workspace_size, proc_id_in.get(),
        proc_id_out.get(), static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, part_bits, stream));
  }
  // explicitly free so workspace can be re-used
  proc_id_in.free();

  // perform a histogram and then prefixsum on the sorted proc_id vector

  // Count the number of values to be sent to each processor
  {
    using AtomicCount = unsigned long long;  // NOLINT
    static_assert(
        sizeof(AtomicCount) == sizeof(*out_counts),
        "AtomicCount must be the same width as int64_t for atomicAdd "
        "in cub::DeviceHistogram::HistogramEven() to work");

    // TODO(dlasalle): Once https://github.com/NVIDIA/cub/pull/287 is merged,
    // add a compile time check against the cub version to allow
    // num_in > (2 << 31).
    CHECK(num_in < static_cast<int64_t>(std::numeric_limits<int>::max()))
        << "number of values to insert into histogram must be less than max "
           "value of int.";

    size_t hist_workspace_size;
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        nullptr, hist_workspace_size, proc_id_out.get(),
        reinterpret_cast<AtomicCount*>(out_counts), num_parts + 1,
        static_cast<IdType>(0), static_cast<IdType>(num_parts),
        static_cast<int>(num_in), stream));

    Workspace<void> hist_workspace(device, ctx, hist_workspace_size);
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        hist_workspace.get(), hist_workspace_size, proc_id_out.get(),
        reinterpret_cast<AtomicCount*>(out_counts), num_parts + 1,
        static_cast<IdType>(0), static_cast<IdType>(num_parts),
        static_cast<int>(num_in), stream));
  }

  return result;
}

template std::pair<IdArray, IdArray> GeneratePermutationFromRemainder<
    kDGLCUDA, int32_t>(int64_t array_size, int num_parts, IdArray in_idx);
template std::pair<IdArray, IdArray> GeneratePermutationFromRemainder<
    kDGLCUDA, int64_t>(int64_t array_size, int num_parts, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType>
IdArray MapToLocalFromRemainder(const int num_parts, IdArray global_idx) {
  const auto& ctx = global_idx->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  if (num_parts > 1) {
    IdArray local_idx =
        aten::NewIdArray(global_idx->shape[0], ctx, sizeof(IdType) * 8);

    const dim3 block(128);
    const dim3 grid((global_idx->shape[0] + block.x - 1) / block.x);

    CUDA_KERNEL_CALL(
        _MapLocalIndexByRemainderKernel, grid, block, 0, stream,
        static_cast<const IdType*>(global_idx->data), global_idx->shape[0],
        num_parts, static_cast<IdType*>(local_idx->data));

    return local_idx;
  } else {
    // no mapping to be done
    return global_idx;
  }
}

template IdArray MapToLocalFromRemainder<kDGLCUDA, int32_t>(
    int num_parts, IdArray in_idx);
template IdArray MapToLocalFromRemainder<kDGLCUDA, int64_t>(
    int num_parts, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType>
IdArray MapToGlobalFromRemainder(
    const int num_parts, IdArray local_idx, const int part_id) {
  CHECK_LT(part_id, num_parts)
      << "Invalid partition id " << part_id << "/" << num_parts;
  CHECK_GE(part_id, 0) << "Invalid partition id " << part_id << "/"
                       << num_parts;

  const auto& ctx = local_idx->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  if (num_parts > 1) {
    IdArray global_idx =
        aten::NewIdArray(local_idx->shape[0], ctx, sizeof(IdType) * 8);

    const dim3 block(128);
    const dim3 grid((local_idx->shape[0] + block.x - 1) / block.x);

    CUDA_KERNEL_CALL(
        _MapGlobalIndexByRemainderKernel, grid, block, 0, stream,
        static_cast<const IdType*>(local_idx->data), part_id,
        global_idx->shape[0], num_parts,
        static_cast<IdType*>(global_idx->data));

    return global_idx;
  } else {
    // no mapping to be done
    return local_idx;
  }
}

template IdArray MapToGlobalFromRemainder<kDGLCUDA, int32_t>(
    int num_parts, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRemainder<kDGLCUDA, int64_t>(
    int num_parts, IdArray in_idx, int part_id);

// Range Based Partition Operations

template <DGLDeviceType XPU, typename IdType, typename RangeType>
std::pair<IdArray, NDArray> GeneratePermutationFromRange(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx) {
  std::pair<IdArray, NDArray> result;

  const auto& ctx = in_idx->ctx;
  auto device = DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_in = in_idx->shape[0];

  CHECK_GE(num_parts, 1) << "The number of partitions (" << num_parts
                         << ") must be at least 1.";
  if (num_parts == 1) {
    // no permutation
    result.first = aten::Range(0, num_in, sizeof(IdType) * 8, ctx);
    result.second = aten::Full(num_in, num_parts, sizeof(int64_t) * 8, ctx);

    return result;
  }

  result.first = aten::NewIdArray(num_in, ctx, sizeof(IdType) * 8);
  result.second = aten::Full(0, num_parts, sizeof(int64_t) * 8, ctx);
  int64_t* out_counts = static_cast<int64_t*>(result.second->data);
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
    const dim3 grid((num_in + block.x - 1) / block.x);

    CUDA_KERNEL_CALL(
        _MapProcByRangeKernel, grid, block, 0, stream,
        static_cast<const RangeType*>(range->data),
        static_cast<const IdType*>(in_idx->data), num_in, num_parts,
        proc_id_in.get());
  }

  // then create a permutation array that groups processors together by
  // performing a radix sort
  Workspace<IdType> proc_id_out(device, ctx, num_in);
  IdType* perm_out = static_cast<IdType*>(result.first->data);
  {
    IdArray perm_in = aten::Range(0, num_in, sizeof(IdType) * 8, ctx);

    size_t sort_workspace_size;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_workspace_size, proc_id_in.get(), proc_id_out.get(),
        static_cast<IdType*>(perm_in->data), perm_out, num_in, 0, part_bits,
        stream));

    Workspace<void> sort_workspace(device, ctx, sort_workspace_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
        sort_workspace.get(), sort_workspace_size, proc_id_in.get(),
        proc_id_out.get(), static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, part_bits, stream));
  }
  // explicitly free so workspace can be re-used
  proc_id_in.free();

  // perform a histogram and then prefixsum on the sorted proc_id vector

  // Count the number of values to be sent to each processor
  {
    using AtomicCount = unsigned long long;  // NOLINT
    static_assert(
        sizeof(AtomicCount) == sizeof(*out_counts),
        "AtomicCount must be the same width as int64_t for atomicAdd "
        "in cub::DeviceHistogram::HistogramEven() to work");

    // TODO(dlasalle): Once https://github.com/NVIDIA/cub/pull/287 is merged,
    // add a compile time check against the cub version to allow
    // num_in > (2 << 31).
    CHECK(num_in < static_cast<int64_t>(std::numeric_limits<int>::max()))
        << "number of values to insert into histogram must be less than max "
           "value of int.";

    size_t hist_workspace_size;
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        nullptr, hist_workspace_size, proc_id_out.get(),
        reinterpret_cast<AtomicCount*>(out_counts), num_parts + 1,
        static_cast<IdType>(0), static_cast<IdType>(num_parts),
        static_cast<int>(num_in), stream));

    Workspace<void> hist_workspace(device, ctx, hist_workspace_size);
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        hist_workspace.get(), hist_workspace_size, proc_id_out.get(),
        reinterpret_cast<AtomicCount*>(out_counts), num_parts + 1,
        static_cast<IdType>(0), static_cast<IdType>(num_parts),
        static_cast<int>(num_in), stream));
  }

  return result;
}

template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLCUDA, int32_t, int32_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);
template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLCUDA, int64_t, int32_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);
template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLCUDA, int32_t, int64_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);
template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLCUDA, int64_t, int64_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType, typename RangeType>
IdArray MapToLocalFromRange(
    const int num_parts, IdArray range, IdArray global_idx) {
  const auto& ctx = global_idx->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  if (num_parts > 1 && global_idx->shape[0] > 0) {
    IdArray local_idx =
        aten::NewIdArray(global_idx->shape[0], ctx, sizeof(IdType) * 8);

    const dim3 block(128);
    const dim3 grid((global_idx->shape[0] + block.x - 1) / block.x);

    CUDA_KERNEL_CALL(
        _MapLocalIndexByRangeKernel, grid, block, 0, stream,
        static_cast<const RangeType*>(range->data),
        static_cast<const IdType*>(global_idx->data), global_idx->shape[0],
        num_parts, static_cast<IdType*>(local_idx->data));

    return local_idx;
  } else {
    // no mapping to be done
    return global_idx;
  }
}

template IdArray MapToLocalFromRange<kDGLCUDA, int32_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx);
template IdArray MapToLocalFromRange<kDGLCUDA, int64_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx);
template IdArray MapToLocalFromRange<kDGLCUDA, int32_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx);
template IdArray MapToLocalFromRange<kDGLCUDA, int64_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType, typename RangeType>
IdArray MapToGlobalFromRange(
    const int num_parts, IdArray range, IdArray local_idx, const int part_id) {
  CHECK_LT(part_id, num_parts)
      << "Invalid partition id " << part_id << "/" << num_parts;
  CHECK_GE(part_id, 0) << "Invalid partition id " << part_id << "/"
                       << num_parts;

  const auto& ctx = local_idx->ctx;
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  if (num_parts > 1 && local_idx->shape[0] > 0) {
    IdArray global_idx =
        aten::NewIdArray(local_idx->shape[0], ctx, sizeof(IdType) * 8);

    const dim3 block(128);
    const dim3 grid((local_idx->shape[0] + block.x - 1) / block.x);

    CUDA_KERNEL_CALL(
        _MapGlobalIndexByRangeKernel, grid, block, 0, stream,
        static_cast<const RangeType*>(range->data),
        static_cast<const IdType*>(local_idx->data), part_id,
        global_idx->shape[0], num_parts,
        static_cast<IdType*>(global_idx->data));

    return global_idx;
  } else {
    // no mapping to be done
    return local_idx;
  }
}

template IdArray MapToGlobalFromRange<kDGLCUDA, int32_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRange<kDGLCUDA, int64_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRange<kDGLCUDA, int32_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRange<kDGLCUDA, int64_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);

}  // namespace impl
}  // namespace partition
}  // namespace dgl
