/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/rowwise_sampling.cu
 * \brief rowwise sampling
 */

#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <curand_kernel.h>
#include <numeric>

#include "./dgl_cub.cuh"
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"

using namespace dgl::aten::cuda;

namespace dgl {
namespace aten {
namespace impl {

namespace {

constexpr int CTA_SIZE = 128;

/**
* @brief Compute the size of each row in the sampled CSR, without replacement.
*
* @tparam IdType The type of node and edge indexes.
* @param num_picks The number of non-zero entries to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The index where each row's edges start.
* @param out_deg The size of each row in the sampled matrix, as indexed by
* `in_rows` (output).
*/
template<typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    IdType * const out_deg) {
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

/**
* @brief Compute the size of each row in the sampled CSR, with replacement.
*
* @tparam IdType The type of node and edge indexes.
* @param num_picks The number of non-zero entries to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The index where each row's edges start.
* @param out_deg The size of each row in the sampled matrix, as indexed by
* `in_rows` (output).
*/
template<typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    IdType * const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x*blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row+1]-in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows-1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
* @brief Perform row-wise uniform sampling on a CSR matrix,
* and generate a COO matrix, without replacement.
*
* @tparam IdType The ID type used for matrices.
* @tparam BLOCK_CTAS The number of rows each thread block runs in parallel.
* @tparam TILE_SIZE The number of rows covered by each threadblock.
* @param rand_seed The random seed to use.
* @param num_picks The number of non-zeros to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The indptr array of the input CSR.
* @param in_index The indices array of the input CSR.
* @param data The data array of the input CSR.
* @param out_ptr The offset to write each row to in the output COO.
* @param out_rows The rows of the output COO (output).
* @param out_cols The columns of the output COO (output).
* @param out_idxs The data array of the output COO (output).
*/
template<typename IdType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == CTA_SIZE);

  int64_t out_row = blockIdx.x*TILE_SIZE+threadIdx.y;
  const int64_t last_row = min(static_cast<int64_t>(blockIdx.x+1)*TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init((rand_seed*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];

    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row+1] - in_row_start;

    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row
      for (int idx = threadIdx.x; idx < deg; idx += CTA_SIZE) {
        const IdType in_idx = in_row_start+idx;
        out_rows[out_row_start+idx] = row;
        out_cols[out_row_start+idx] = in_index[in_idx];
        out_idxs[out_row_start+idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx+=CTA_SIZE) {
        out_idxs[out_row_start+idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks+threadIdx.x; idx < deg; idx+=CTA_SIZE) {
        const int num = curand(&rng)%(idx+1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs+out_row_start+num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += CTA_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start+idx]+in_row_start;
        out_rows[out_row_start+idx] = row;
        out_cols[out_row_start+idx] = in_index[perm_idx];
        out_idxs[out_row_start+idx] = data ? data[perm_idx] : perm_idx;
      }
    }

    out_row += BLOCK_CTAS;
  }
}

/**
* @brief Perform row-wise uniform sampling on a CSR matrix,
* and generate a COO matrix, with replacement.
*
* @tparam IdType The ID type used for matrices.
* @tparam BLOCK_CTAS The number of rows each thread block runs in parallel.
* @tparam TILE_SIZE The number of rows covered by each threadblock.
* @param rand_seed The random seed to use.
* @param num_picks The number of non-zeros to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The indptr array of the input CSR.
* @param in_index The indices array of the input CSR.
* @param data The data array of the input CSR.
* @param out_ptr The offset to write each row to in the output COO.
* @param out_rows The rows of the output COO (output).
* @param out_cols The columns of the output COO (output).
* @param out_idxs The data array of the output COO (output).
*/
template<typename IdType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == CTA_SIZE);

  int64_t out_row = blockIdx.x*TILE_SIZE+threadIdx.y;
  const int64_t last_row = min(static_cast<int64_t>(blockIdx.x+1)*TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init((rand_seed*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];

    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];

    const int64_t deg = in_ptr[row+1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += CTA_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start+idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start+edge];
        out_idxs[out_idx] = data ? data[in_row_start+edge] : in_row_start+edge;
      }
    }
    out_row += BLOCK_CTAS;
  }
}

/**
* @brief Equivalent to numpy expression: array[idx[off:off + len]]
*
* @tparam IdType The ID type used for indices.
* @tparam FloatType The float type used for array values.
* @param array The array to be selected.
* @param idx_data The index mapping array.
* @param index The index of value to be selected.
* @param offset The offset to start.
* @param out The selected value (output).
*/
template<typename IdType, typename FloatType>
__device__ void _DoubleSlice(
    const FloatType * const array,
    const IdType * const idx_data,
    const IdType idx,
    const IdType offset,
    FloatType* const out) {
  if (idx_data) {
    *out = array[idx_data[offset+idx]];
  } else {
    *out = array[offset+idx];
  }
}

/**
* @brief Perform row-wise sampling on a CSR matrix, and generate a COO matrix,
* without replacement.
*
* @tparam IdType The ID type used for matrices.
* @tparam FloatType The float type used for probability.
* @tparam BLOCK_CTAS The number of rows each thread block runs in parallel.
* @tparam TILE_SIZE The number of rows covered by each threadblock.
* @param rand_seed The random seed to use.
* @param num_picks The number of non-zeros to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The indptr array of the input CSR.
* @param in_index The indices array of the input CSR.
* @param data The data array of the input CSR.
* @param out_ptr The offset to write each row to in the output COO.
* @param out_rows The rows of the output COO (output).
* @param out_cols The columns of the output COO (output).
* @param out_idxs The data array of the output COO (output).
*/
template<typename IdType, typename FloatType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const FloatType * const prob,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == CTA_SIZE);

  int64_t out_row = blockIdx.x*TILE_SIZE+threadIdx.y;
  const int64_t last_row = min(static_cast<int64_t>(blockIdx.x+1)*TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init((rand_seed*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];

    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row+1] - in_row_start;

    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row
      for (int idx = threadIdx.x; idx < deg; idx += CTA_SIZE) {
        const IdType in_idx = in_row_start+idx;
        out_rows[out_row_start+idx] = row;
        out_cols[out_row_start+idx] = in_index[in_idx];
        out_idxs[out_row_start+idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // Algorithm A-Chao https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Chao
      using BlockScanT = cub::BlockScan<FloatType, CTA_SIZE>;
      // Allocate shared memory for BlockScan
      __shared__ typename BlockScanT::TempStorage temp_storage;

      FloatType weights_sum = (FloatType)0.0;
      // for the following two loops, we use a block scan window of size CTA_SIZE
      // fill the reservoir array and compute the sum of their weights
      for (int i = 0; i < (num_picks + CTA_SIZE - 1) / CTA_SIZE; i++) {
        // Obtain input item for each thread
        IdType idx = threadIdx.x + i * CTA_SIZE;
        FloatType thread_prob;
        if (idx < num_picks) {
          _DoubleSlice<IdType, FloatType>(prob, data, idx, in_row_start, &thread_prob);
          out_idxs[out_row_start+idx] = idx;
        } else {
          thread_prob = (FloatType)0.0;
        }
        __syncthreads();

        // Collectively compute the block-wide inclusive prefix sum
        FloatType block_aggregate;
        BlockScanT(temp_storage).InclusiveSum(thread_prob, thread_prob, block_aggregate);

        weights_sum += block_aggregate;
      }

      // for items in [num_picks, deg), check if they can replace
      for (int i = 0; i < (deg - num_picks + CTA_SIZE - 1) / CTA_SIZE; i++) {
        // Obtain input item for each thread
        IdType idx = num_picks + threadIdx.x + i * CTA_SIZE;
        FloatType thread_prob, thread_sum;
        if (idx < deg) {
          _DoubleSlice<IdType, FloatType>(prob, data, idx, in_row_start, &thread_prob);
        } else {
          thread_prob = (FloatType)0.0;
        }
        __syncthreads();

        // Collectively compute the block-wide inclusive prefix sum
        FloatType block_aggregate;
        BlockScanT(temp_storage).InclusiveSum(thread_prob, thread_sum, block_aggregate);

        if (idx < deg) {
          thread_prob /= (thread_sum + weights_sum);
          if (curand_uniform(&rng) < thread_prob) {
            const int num = curand(&rng) % num_picks;
            // use max so as to achieve the replacement order the serial
            // algorithm would have
            AtomicMax(out_idxs+out_row_start+num, idx);
          }
        }
        __syncthreads();

        weights_sum += block_aggregate;
      }

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += CTA_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start+idx]+in_row_start;
        out_rows[out_row_start+idx] = row;
        out_cols[out_row_start+idx] = in_index[perm_idx];
        out_idxs[out_row_start+idx] = data ? data[perm_idx] : perm_idx;
      }
    }

    out_row += BLOCK_CTAS;
  }
}

/**
* @brief Perform row-wise sampling on a CSR matrix, and generate a COO matrix,
* with replacement.
*
* @tparam IdType The ID type used for matrices.
* @tparam FloatType The float type used for probability.
* @tparam BLOCK_CTAS The number of rows each thread block runs in parallel.
* @tparam TILE_SIZE The number of rows covered by each threadblock.
* @param rand_seed The random seed to use.
* @param num_picks The number of non-zeros to pick per row.
* @param num_rows The number of rows to pick.
* @param in_rows The set of rows to pick.
* @param in_ptr The indptr array of the input CSR.
* @param in_index The indices array of the input CSR.
* @param data The data array of the input CSR.
* @param out_ptr The offset to write each row to in the output COO.
* @param out_rows The rows of the output COO (output).
* @param out_cols The columns of the output COO (output).
* @param out_idxs The data array of the output COO (output).
*/
template<typename IdType, typename FloatType, int BLOCK_CTAS, int TILE_SIZE>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t rand_seed,
    const int64_t num_picks,
    const int64_t num_rows,
    const IdType * const in_rows,
    const FloatType * const prob,
    const IdType * const in_ptr,
    const IdType * const in_index,
    const IdType * const data,
    const IdType * const out_ptr,
    IdType * const out_rows,
    IdType * const out_cols,
    IdType * const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == CTA_SIZE);

  int64_t out_row = blockIdx.x*TILE_SIZE+threadIdx.y;
  const int64_t last_row = min(static_cast<int64_t>(blockIdx.x+1)*TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init((rand_seed*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];

    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];

    const int64_t deg = in_ptr[row+1] - in_row_start;

    if (deg > 0) {
      constexpr int ITEMS_PER_THREAD = 1024/CTA_SIZE;
      using BlockRadixSortT = cub::BlockRadixSort<float, CTA_SIZE, ITEMS_PER_THREAD>;
      using BlockScanT = cub::BlockScan<FloatType, CTA_SIZE>;
      using BlockReduceI = cub::BlockReduce<int, CTA_SIZE>;
      using BlockReduceT = cub::BlockReduce<FloatType, CTA_SIZE>;

      __shared__ FloatType prefix_sum[CTA_SIZE];
      __shared__ typename BlockRadixSortT::TempStorage sort_storage;
      __shared__ typename BlockScanT::TempStorage scan_storage;
      __shared__ typename BlockReduceI::TempStorage reduce_storage_i;
      __shared__ typename BlockReduceT::TempStorage reduce_storage_t;

      // initialize thread_rn
      // using float for the random number is enough
      float thread_rn[ITEMS_PER_THREAD];
      for (int idx = 0; idx < ITEMS_PER_THREAD; idx++) {
        if (idx * CTA_SIZE + threadIdx.x < num_picks) {
          // this is a workaround for that 1.0f is inclusive in curand_uniform
          // while the sum of normalized probabilities is usually a bit smaller than 1.0
          // (Xin): 1 - curand_uniform(&rng) doesn't work
          thread_rn[idx] = curand_uniform(&rng) - 1e-6f;
        } else {
          thread_rn[idx] = 1.0;
        }
      }
      __syncthreads();
      // sort thread_rn.
      BlockRadixSortT(sort_storage).SortBlockedToStriped(thread_rn);

      // get the sum of probs for normalization
      __shared__ FloatType weights_sum;
      if (threadIdx.x == 0) {
        weights_sum = (FloatType)0.0;
      } // don't need to sync here
      for (int i = 0; i < (deg + CTA_SIZE - 1) / CTA_SIZE; i++) {
        // Obtain input item for each thread
        IdType idx = threadIdx.x + i * CTA_SIZE;
        FloatType thread_prob;
        if (idx < deg) {
          _DoubleSlice<IdType, FloatType>(prob, data, idx, in_row_start, &thread_prob);
        } else {
          thread_prob = (FloatType)0.0;
        }
        __syncthreads();

        auto window_sum = BlockReduceT(reduce_storage_t).Sum(thread_prob);
        if (threadIdx.x == 0) {
          weights_sum += window_sum;
        } // don't need to sync here
      }

      FloatType moving_sum = (FloatType)0.0;
      __shared__ int64_t num_selected;
      if (threadIdx.x == 0) {
        num_selected = 0;
      } // we don't need to sync here because there will be a sync before the first use
      __shared__ int num_selected_this_round;
      // we use a moving window to compute the inclusive prefix sum
      // of [i * CTA_SIZE, (i + 1) * CTA_SIZE)
      for (int i = 0; i < (deg + CTA_SIZE - 1) / CTA_SIZE; i++) {
        // Obtain input item for each thread
        IdType idx = threadIdx.x + i * CTA_SIZE;
        FloatType thread_prob;
        if (idx < deg) {
          _DoubleSlice<IdType, FloatType>(prob, data, idx, in_row_start, &thread_prob);
        } else {
          thread_prob = (FloatType)0.0;
        }
        thread_prob /= weights_sum;
        __syncthreads();

        // Collectively compute the block-wide inclusive prefix sum
        FloatType block_aggregate;
        BlockScanT(scan_storage).InclusiveSum(thread_prob, thread_prob, block_aggregate);
        // Store the block-wide inclusive prefix sum in shared memory
        prefix_sum[threadIdx.x] = moving_sum + thread_prob;
        __syncthreads();

        // binary search to find the idx
        for (int j = num_selected / CTA_SIZE; j < ITEMS_PER_THREAD; j++) {
          IdType out_offset = j * CTA_SIZE + threadIdx.x;
          // selected if out_offset is in [num_selected, num_picks)
          // and its random number is found in this prefix sum window
          int flag_selected = 0;
          if (out_offset >= num_selected && out_offset < num_picks) {
            auto rn_val = thread_rn[j];
            // there could be a bank conflict here
            auto ptr = thrust::lower_bound(thrust::seq, prefix_sum, prefix_sum + CTA_SIZE, rn_val);
            auto idx_offset = thrust::distance(prefix_sum, ptr);
            if (idx_offset < CTA_SIZE) {
              flag_selected = 1;
              // this should always be true since we use lower_bound
              assert(idx_offset + i * CTA_SIZE < deg);
              out_idxs[out_row_start+out_offset] = idx_offset + i * CTA_SIZE;
            }
          }
          __syncthreads();

          auto flag_sum = BlockReduceI(reduce_storage_i).Sum(flag_selected);
          // let all threads see the results
          if (threadIdx.x == 0) {
            num_selected_this_round = flag_sum;
            num_selected += num_selected_this_round;
          }
          __syncthreads();

          // Move to the next window of prefix sum
          if (num_selected % CTA_SIZE != 0) break;
          // when num_selected % CTA_SIZE == 0, we move the window
          // of rn_array to make full use of the current window of prefix sum
        }
        moving_sum += block_aggregate;
        // if we have selected enough edges, skip the subsequent window
        if (num_selected == num_picks) break;
      }

      // check if we have selected enough edges when stop natually
      assert(num_selected == num_picks);

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += CTA_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start+idx]+in_row_start;
        out_rows[out_row_start+idx] = row;
        out_cols[out_row_start+idx] = in_index[perm_idx];
        out_idxs[out_row_start+idx] = data ? data[perm_idx] : perm_idx;
      }
    }

    out_row += BLOCK_CTAS;
  }
}

}  // namespace

/////////////////////////////// CSR uniform ///////////////////////////////

template <DLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(CSRMatrix mat,
                                    IdArray rows,
                                    const int64_t num_picks,
                                    const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);

  // TODO(dlasalle): Once the device api supports getting the stream from the
  // context, that should be used instead of the default stream here.
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

  const IdType* const data = CSRHasData(mat) ?
      static_cast<IdType*>(mat.data->data) : nullptr;

  // compute degree
  IdType * out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows+block.x-1)/block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeReplaceKernel,
        grid, block, 0, stream,
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows+block.x-1)/block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeKernel,
        grid, block, 0, stream,
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  }

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

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  // TODO(dlasalle): use pinned memory to overlap with the actual sampling, and wait on
  // a cudaevent
  IdType new_len;
  device->CopyDataFromTo(out_ptr, num_rows*sizeof(new_len), &new_len, 0,
        sizeof(new_len),
        ctx,
        DGLContext{kDLCPU, 0},
        mat.indptr->dtype,
        stream);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  if (replace) {
    constexpr int BLOCK_CTAS = 128/CTA_SIZE;
    // the number of rows each thread block will cover
    constexpr int TILE_SIZE = BLOCK_CTAS;
    const dim3 block(CTA_SIZE, BLOCK_CTAS);
    const dim3 grid((num_rows+TILE_SIZE-1)/TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformReplaceKernel<IdType, BLOCK_CTAS, TILE_SIZE>),
        grid, block, 0, stream,
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
    constexpr int BLOCK_CTAS = 128/CTA_SIZE;
    // the number of rows each thread block will cover
    constexpr int TILE_SIZE = BLOCK_CTAS;
    const dim3 block(CTA_SIZE, BLOCK_CTAS);
    const dim3 grid((num_rows+TILE_SIZE-1)/TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformKernel<IdType, BLOCK_CTAS, TILE_SIZE>),
        grid, block, 0, stream,
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

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

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


/////////////////////////////// CSR non-uniform ///////////////////////////////

template <DLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix CSRRowWiseSampling(CSRMatrix mat,
                             IdArray rows,
                             const int64_t num_picks,
                             FloatArray prob,
                             const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);

  // TODO(dlasalle): Once the device api supports getting the stream from the
  // context, that should be used instead of the default stream here.
  cudaStream_t stream = 0;

  CHECK_EQ(prob->shape[0], mat.indices->shape[0]);
  const FloatType* const prob_data = static_cast<const FloatType*>(prob->data);

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

  const IdType* const data = CSRHasData(mat) ?
      static_cast<IdType*>(mat.data->data) : nullptr;

  // compute degree
  IdType * out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows+1)*sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows+block.x-1)/block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeReplaceKernel,
        grid, block, 0, stream,
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows+block.x-1)/block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeKernel,
        grid, block, 0, stream,
        num_picks, num_rows, slice_rows, in_ptr, out_deg);
  }

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

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  // TODO(dlasalle): use pinned memory to overlap with the actual sampling, and wait on
  // a cudaevent
  IdType new_len;
  device->CopyDataFromTo(out_ptr, num_rows*sizeof(new_len), &new_len, 0,
        sizeof(new_len),
        ctx,
        DGLContext{kDLCPU, 0},
        mat.indptr->dtype,
        stream);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);

  // select edges
  if (replace) {
    constexpr int BLOCK_CTAS = 128/CTA_SIZE;
    // the number of rows each thread block will cover
    constexpr int TILE_SIZE = BLOCK_CTAS;
    const dim3 block(CTA_SIZE, BLOCK_CTAS);
    const dim3 grid((num_rows+TILE_SIZE-1)/TILE_SIZE);

    // (Xin): we need a constant array of random numbers for sorting
    // and 1024 should be enough for any reasonable use case
    CHECK_LE(num_picks, 1024) << "GPU sampling is not supported for fanout > 1024";
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleReplaceKernel<IdType, FloatType, BLOCK_CTAS, TILE_SIZE>),
        grid, block, 0, stream,
        random_seed,
        num_picks,
        num_rows,
        slice_rows,
        prob_data,
        in_ptr,
        in_cols,
        data,
        out_ptr,
        out_rows,
        out_cols,
        out_idxs);
  } else {
    constexpr int BLOCK_CTAS = 128/CTA_SIZE;
    // the number of rows each thread block will cover
    constexpr int TILE_SIZE = BLOCK_CTAS;
    const dim3 block(CTA_SIZE, BLOCK_CTAS);
    const dim3 grid((num_rows+TILE_SIZE-1)/TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleKernel<IdType, FloatType, BLOCK_CTAS, TILE_SIZE>),
        grid, block, 0, stream,
        random_seed,
        num_picks,
        num_rows,
        slice_rows,
        prob_data,
        in_ptr,
        in_cols,
        data,
        out_ptr,
        out_rows,
        out_cols,
        out_idxs);
  }
  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(mat.num_rows, mat.num_cols, picked_row,
      picked_col, picked_idx);
}

template COOMatrix CSRRowWiseSampling<kDLGPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLGPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLGPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLGPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
