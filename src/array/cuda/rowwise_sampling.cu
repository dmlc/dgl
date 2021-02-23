/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/rowwise_sampling.cu
 * \brief rowwise sampling
 */
#include <dgl/random.h>
#include <numeric>

#include "../../kernel/cuda/atomic.cuh"

namespace dgl {
namespace aten {
namespace impl {

namespace {

constexpr int LogPow2(
    const int num)
{
  if (num == 0) {
    return 0;
  } else {
    return LogPow2(num>>1)+1;
  }
}

template<typename IdType>
__global__ void CSRRowWiseSampleDegreeKernel(
    const int num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    IdType * const out_deg)
{
  const int tIdx = threadIdx.x + blockIdx.x*blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(static_cast<IdType>(num_picks), in_ptr[in_row+1]-in_ptr[in_row]);

    if (out_row == num_rows-1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

template<typename IdType, int BLOCK_SIZE>
__global__ void CSRRowWiseSampleKernel(
    const unsigned long rand_seed,
    const int num_picks,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const in_rows,
    const IdType * const out_ptr,
    IdType * const out_index)
{
  typedef cub::BlockRadixSort<int, BLOCK_SIZE, 1> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  __shared__ int shared_keys[BLOCK_SIZE];
  __shared__ int shared_indexes[BLOCK_SIZE];

  const int64_t out_row = threadIdx.x + blockIdx.x*blockDim.x
  const int64_t in_row = in_rows[out_row];

  const int64_t in_row_start = in_ptr[in_row];
  const int deg = in_ptr[in_row+1] - in_row_start;

  const int64_t out_row_start = out_ptr[out_row];

  if (deg <= num_picks) {
    // just copy row
    for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
      out_index[out_row_start+idx] = in_index[in_row_start+idx];
    }
  } else {
    // each thread needs to initialize it's random state
    curandState rng;
    curand_init(rand_seed, out_row, 0, &rng);

    if (deg <= BLOCK_SIZE) {
      // shuffle index array, and select based on that
      constexpr int BLOCK_BITS = LogPow2(BlockSize);
      constexpr int BLOCK_MASK = (1<<BLOCK_BITS)-1;

      // make sure block size is a power of two
      static_assert((1 << (BLOCK_BITS-1)) == BLOCK_SIZE);

      // generate a list of unique indexes, and select those
      int key = threadIdx.x < deg ? static_cast<int>(curand(&rng) &
          BlockMask) : BLOCK_SIZE;
      int value = threadIdx.x;
      BlockradixSort(temp_storage).Sort(key, value);

      // copy permutation
      const int idx = threadIdx.x;
      if (value != BLOCK_SIZE) {
        index_out[out_row_start+idx] = index_in[in_row_start+value];
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; ++idx) {
        index_out[out_row_start+idx] = idx;
      }
      for (int idx = num_picks+threadIdx.x; idx < deg; ++idx) {
        const int num = curand(&rng)%(idx+1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(index_out+out_row_start+num, idx);
        }
      }

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; ++idx) {
        const int perm_idx = index_out[out_row_start+idx];
        index_out[out_row_start+idx] = in_index[in_row_start+perm_idx];
      }
    }
  }
}

}  // namespace

/////////////////////////////// CSR ///////////////////////////////

template <DLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(CSRMatrix mat, IdArray rows,
                                    const int64_t num_samples,
                                    const bool replace) {

  const auto& ctx = mat.indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);

  // TODO: get stream from context
  cudaStream_t stream = 0;

  const int64_t num_rows = rows->shape[0];

  IdType * out_ptr = device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType));

  IdArray picked_row = NewIdArray(-1, num_rows * num_picks, sizeof(IdType) * 8, ctx);
  IdArray picked_col = NewIdArray(-1, num_rows * num_picks, sizeof(IdType) * 8, ctx);
  IdArray picked_idx = NewIdArray(-1, num_rows * num_picks, sizeof(IdType) * 8, ctx);
  IdType* const out_rows = static_cast<IdxType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdxType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdxType*>(picked_idx->data);

  if (!replace) {
    constexpr int BLOCK_SIZE = 128;

    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows+block.x-1)/block.x);

    // compute degree
    CSRRowwiseSampleDegreeKernel<<<grid, block, 0, stream>>>(
      num_picks, num_rows,
      mat->rowptr,
      out_deg);

    // fill out_ptr
    size_t prefix_temp_size;
    cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
        out_ptr,
        out_ptr,
        num_rows+1,
        stream);
    void * prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
    cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
        out_ptr,
        out_ptr,
        num_rows+1,
        stream);
    device->FreeWorkspace(ctx, prefix_temp);

    // select edges
    CSRRowWiseSampleKernel<IdType, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        rand_seed,

    picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
    picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
    picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);
  } else {
    // we can do everything in one kernel since the degree is constant
    throw std::runtime_error("CSRRowWiseSamplingUniform is not implemented for CUDA");
  }

  return COOMatrix(mat.num_rows, mat.num_cols, picked_row,
      picked_col, picked_idx);
}

template COOMatrix CSRRowWiseSamplingUniform<kDLGPU, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDLGPU, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
