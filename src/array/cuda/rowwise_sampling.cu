/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/rowwise_sampling.cu
 * \brief rowwise sampling
 */
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <numeric>
#include <cub/cub.cuh>
#include <curand_kernel.h>

#include "../../kernel/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"

using namespace dgl::kernel::cuda;

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

template<typename IdType>
__global__ void CSRRowWiseSampleDegreeReplaceKernel(
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

    if (in_ptr[in_row+1]-in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    printf("out_deg[%d] = %d\n", (int)out_row, (int)out_deg[out_row]);

    if (out_row == num_rows-1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
      printf("out_deg[%d] = %d\n", (int)num_rows, (int)out_deg[num_rows]);
    }
  }
}

template<typename IdType, int BLOCK_SIZE>
__global__ void CSRRowWiseSampleKernel(
    const unsigned long rand_seed,
    const int num_picks,
    const int num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs)
{
  typedef cub::BlockRadixSort<int, BLOCK_SIZE, 1, int> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int64_t out_row = threadIdx.x + blockIdx.x*BLOCK_SIZE;
  const int64_t row = in_rows[out_row];

  const int64_t in_row_start = in_ptr[row];
  const int deg = in_ptr[row+1] - in_row_start;

  const int64_t out_row_start = out_ptr[out_row];

  if (deg <= num_picks) {
    // just copy row
    for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
      const IdType in_idx = in_row_start+idx;
      out_rows[out_row_start+idx] = row;
      out_cols[out_row_start+idx] = in_index[in_idx];
      out_idxs[out_row_start+idx] = data ? data[in_idx] : in_idx;
    }
  } else {
    // each thread needs to initialize it's random state
    curandState rng;
    curand_init(rand_seed, out_row, 0, &rng);

    if (deg <= BLOCK_SIZE) {
      // shuffle index array, and select based on that
      constexpr int BLOCK_BITS = LogPow2(BLOCK_SIZE);
      constexpr int BLOCK_MASK = (1<<BLOCK_BITS)-1;

      // make sure block size is a power of two
      static_assert((1 << (BLOCK_BITS-1)) == BLOCK_SIZE);

      // generate a list of unique indexes, and select those
      int key[1] = {threadIdx.x < deg ?
          static_cast<int>(curand(&rng) & BLOCK_MASK) :
          BLOCK_SIZE};
      int value[1] = {static_cast<int>(threadIdx.x)};
      BlockRadixSort(temp_storage).Sort(key, value, 0, BLOCK_BITS);

      // copy permutation
      const int idx = threadIdx.x;
      if (value[0] != BLOCK_SIZE) {
        const IdType in_idx = in_row_start+value[0];
        out_rows[out_row_start+idx] = row;
        out_cols[out_row_start+idx] = in_index[in_idx];
        out_idxs[out_row_start+idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; ++idx) {
        out_idxs[out_row_start+idx] = in_row_start+idx;
      }
      for (int idx = num_picks+threadIdx.x; idx < deg; ++idx) {
        const int num = curand(&rng)%(idx+1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs+out_row_start+num, idx);
        }
      }

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; ++idx) {
        const int perm_idx = out_idxs[out_row_start+idx];
        out_rows[out_row_start+idx] = row;
        out_cols[out_row_start+idx] = in_index[perm_idx];
        if (data) {
          out_idxs[out_row_start+idx] = data[perm_idx];
        }
      }
    }
  }
}

template<typename IdType>
__global__ void CSRRowWiseSampleReplaceKernel(
    const unsigned long rand_seed,
    const int num_picks,
    const int num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs)
{
  const int64_t out_row = threadIdx.x + blockIdx.x*blockDim.x;
  const int64_t row = in_rows[out_row];

  const int64_t in_row_start = in_ptr[row];
  const int64_t out_row_start = out_ptr[out_row];

  // each thread needs to initialize it's random state
  curandState rng;
  curand_init(rand_seed, out_row, 0, &rng);

  const int deg = in_ptr[row+1] - in_row_start;

  // each thread then blindly copies in rows 
  for (int idx = threadIdx.x; idx < deg; idx += blockDim.x) {
    const int edge = curand(&rng) % deg;
    out_rows[out_row_start+idx] = row;
    out_cols[out_row_start+idx] = in_index[in_row_start+edge];
    out_idxs[out_row_start+idx] = data ? data[in_row_start+edge] : in_row_start+edge;
  }
}

}  // namespace

/////////////////////////////// CSR ///////////////////////////////

template <DLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(CSRMatrix mat,
                                    IdArray rows,
                                    const int64_t num_picks,
                                    const bool replace) {
  constexpr int BLOCK_SIZE = 128;

  const auto& ctx = mat.indptr->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);

  // TODO: get stream from context
  cudaStream_t stream = 0;

  const int64_t num_rows = rows->shape[0];
  const IdType * const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row = NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col = NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx = NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  const IdType * const in_ptr = static_cast<const IdType*>(mat.indptr->data);
  const IdType * const in_cols = static_cast<const IdType*>(mat.indices->data);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* const data = CSRHasData(mat)? static_cast<IdType*>(mat.data->data) : nullptr;

  const dim3 block(BLOCK_SIZE);
  const dim3 grid((num_rows+block.x-1)/block.x);

  // compute degree
  IdType * out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));
  if (replace) {
    CSRRowWiseSampleDegreeReplaceKernel<<<grid, block, 0, stream>>>(
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  } else {
    CSRRowWiseSampleDegreeKernel<<<grid, block, 0, stream>>>(
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  }

  // TODO: in_rows might have data associated with it

  // fill out_ptr
  IdType * out_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  void * prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_temp, prefix_temp_size,
      out_deg,
      out_ptr,
      num_rows+1,
      stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  // TODO: use pinned memory to overlap with the actual sampling, and wait on
  // a cudaevent
  IdType new_len;
  device->CopyDataFromTo(out_ptr, num_rows*sizeof(new_len), &new_len, 0,
        sizeof(new_len),
        ctx,
        DGLContext{kDLCPU, 0},
        mat.indptr->dtype,
        stream);
  device->StreamSync(ctx, stream);
  std::cout << "New length = " << new_len << ", num_rows = " << num_rows <<
      ", num_pick = " << num_picks << ", replace = " << replace <<
      " edges = " << mat.indices->shape[0] << std::endl;

  const unsigned int random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  if (replace) {
    CSRRowWiseSampleReplaceKernel<<<grid, block, 0, stream>>>(
        random_seed,
        num_picks, 
        num_rows,
        slice_rows,
        in_ptr,
        in_cols,
        data,
        out_ptr,
        out_rows,
        out_cols,
        out_idxs);
  } else {
    CSRRowWiseSampleKernel<IdType, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        random_seed,
        num_picks,
        num_rows,
        slice_rows,
        in_ptr,
        in_cols,
        data,
        out_ptr,
        out_rows,
        out_cols,
        out_idxs);
  }
  device->FreeWorkspace(ctx, out_ptr);

  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

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
