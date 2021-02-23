/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/rowwise_sampling.cu
 * \brief rowwise sampling
 */
#include <dgl/random.h>
#include <numeric>
#include "./rowwise_pick.h"
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

template<typename DType, int BLOCK_SIZE>
__global__ void CSRRowWiseSampleKernel(,
    const unsigned long rand_seed,
    const int num_picks,
    const IdType * in_ptr,
    const IdType * in_index,
    const IdType * out_ptr,
    IdType * out_index)
{
  typedef cub::BlockRadixSort<int, BLOCK_SIZE, 1> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  __shared__ int shared_keys[BLOCK_SIZE];
  __shared__ int shared_indexes[BLOCK_SIZE];

  const int tIdx = threadIdx.x + blockIdx.x*blockDim.x;

  const int64_t in_row_start = in_ptr[row];
  const int deg = in_ptr[row+1] - in_row_start;

  const int64_t out_row_start = out_ptr[row];

  if (deg <= num_picks) {
    // just copy row
    for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
      out_index[out_row_start+idx] = in_index[in_row_start+idx];
    }
  } else {
    // each thread needs to initialize it's random state
    curandState rng;
    curand_init(rand_seed, tIdx, 0, &rng);

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

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseSampling(CSRMatrix mat, IdArray rows, int64_t num_samples,
                             FloatArray prob, bool replace) {
  CHECK(prob.defined());
  auto pick_fn = GetSamplingPickFn<IdxType, FloatType>(num_samples, prob, replace);
  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix CSRRowWiseSampling<kDLGPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLGPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLGPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLGPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);

template <DLDeviceType XPU, typename IdxType>
COOMatrix CSRRowWiseSamplingUniform(CSRMatrix mat, IdArray rows,
                                    int64_t num_samples, bool replace) {
  // launch 1 thread block per row

  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix CSRRowWiseSamplingUniform<kDLGPU, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDLGPU, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

/////////////////////////////// COO ///////////////////////////////

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix COORowWiseSampling(COOMatrix mat, IdArray rows, int64_t num_samples,
                             FloatArray prob, bool replace) {
  CHECK(prob.defined());
  auto pick_fn = GetSamplingPickFn<IdxType, FloatType>(num_samples, prob, replace);
  return COORowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix COORowWiseSampling<kDLGPU, int32_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDLGPU, int64_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDLGPU, int32_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDLGPU, int64_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);

template <DLDeviceType XPU, typename IdxType>
COOMatrix COORowWiseSamplingUniform(COOMatrix mat, IdArray rows,
                                    int64_t num_samples, bool replace) {
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return COORowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix COORowWiseSamplingUniform<kDLGPU, int32_t>(
    COOMatrix, IdArray, int64_t, bool);
template COOMatrix COORowWiseSamplingUniform<kDLGPU, int64_t>(
    COOMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
