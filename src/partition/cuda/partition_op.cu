/*!
 *  Copyright (c) 2021 by Contributors
 * \file ndarray_partition.h 
 * \brief DGL utilities for working with the partitioned NDArrays 
 */


namespace dgl {
namespace partition {
namespace impl {

template<typename IdType> __global__ void _MapProcByRemainder(
    const IdType * const index,
    const int64_t num_index,
    const int64_t num_proc,
    IdType * const proc_id) {
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
  const int64_t idx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;

  if (idx < num_index) {
    proc_id[idx] = index[idx] & mask;
  }
}



template <DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray>
GeneratePermutationFromRemainder(
        int64_t array_size,
        int num_parts,
        IdArray in_idx)
{
  std::pair<IdArray, IdArray> result;

  const auto& ctx = in_idx->ctx;
  auto device = DeviceAPI::Get(ctx);

  const int64_t num_in = in_idx->shape[0];

  CHECK_GE(num_parts, 1);
  if (num_parts == 1) {
    // no permutation
    result.first = aten::Range(0, num_in, ctx);
    result.second = aten::Full(num_in, num_parts, ctx); 

    return results;
  }

  IdArray& out_counts = results.second = aten::Full(0, num_parts, ctx);
  if (num_in == 0) {
    // now that we've zero'd out_counts, nothing left to do for an empty
    // mapping
    return results;
  }

  const int64_t part_bits =
      static_cast<int64_t>(std::ceil(std::log2(num_parts)));

  // First, generate a mapping of indexes to processors
  IdType * proc_id_in = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    if (comm_size < (1 << part_bits)) {
      // comm_size is not a power of 2
      _MapProcByRemainder<<<grid, block, 0, stream>>>(
          static_cast<const IdType*>(in_idx->data),
          num_in,
          num_parts,
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    } else {
      // comm_size is a power of 2
      _MapProcByMaskRemainder<<<grid, block, 0, stream>>>(
          static_cast<const IdType*>(in_idx->data),
          num_in,
          static_cast<IdType>(num_parts-1),  // bit mask
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    }
  }

  // then create a permutation array that groups processors together by
  // performing a radix sort
  IdType * proc_id_out = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  results.first = aten::NewIdArray(num_in, ctx, sizeof(IdType)*8);
  IdType * perm_out = static_cast<IdType*>(results.first->data);
  {
    IdArray perm_in = aten::Range(0, num_in, sizeof(IdType)*8, ctx);

    size_t sort_workspace_size;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, comm_bits, stream));

    void * sort_workspace = device->AllocWorkspace(ctx, sort_workspace_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(sort_workspace, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, comm_bits, stream));
    device->FreeWorkspace(ctx, sort_workspace);
  }
  device->FreeWorkspace(ctx, proc_id_in);

  // perform a histogram and then prefixsum on the sorted proc_id vector

  // Count the number of values to be sent to each processor
  {
    using AtomicCount = unsigned long long; // NOLINT
    static_assert(sizeof(AtomicCount) == sizeof(int64_t),
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
        proc_id_out,
        reinterpret_cast<AtomicCount*>(out_counts),
        comm_size+1,
        static_cast<IdType>(0),
        static_cast<IdType>(comm_size+1),
        static_cast<int>(num_in),
        stream));

    void * hist_workspace = device->AllocWorkspace(ctx, hist_workspace_size);
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        hist_workspace,
        hist_workspace_size,
        proc_id_out,
        reinterpret_cast<AtomicCount*>(out_counts),
        comm_size+1,
        static_cast<IdType>(0),
        static_cast<IdType>(comm_size+1),
        static_cast<int>(num_in),
        stream));
    device->FreeWorkspace(ctx, hist_workspace);
  }

  return results;
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

}
}
}

