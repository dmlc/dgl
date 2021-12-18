/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/negative_sampling.cu
 * \brief rowwise sampling
 */

#include <dgl/random.h>
#include <dgl/array.h>
#include <curand_kernel.h>

#include "./dgl_cub.cuh"
#include "./utils.h"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
namespace sampling {
namespace impl {

template <typename IdType>
__global__ void _GlobalUniformNegativeSamplingKernel(
    const __restrict__ IdType* indptr,
    const __restrict__ IdType* indices,
    __restrict__ IdType* row,
    __restrict__ IdType* col,
    int64_t num_row,
    int64_t num_col,
    int64_t num_samples,
    int num_trials,
    bool exclude_self_loops,
    int64_t random_seed) {
  int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;

  curandStatePhilox4_32_10_t rng; // this allows generating 4 32-bit ints at a time
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (tx < num_samples) {
    for (int i = 0; i < num_trials; ++i) {
      uint4 result = curand4(&rng);
      IdType u = ((result.x << 32) | result.y) % num_row;
      IdType v = ((result.z << 32) | result.w) % num_col;

      if (exclude_self_loops && (u == v))
        continue;

      // binary search of v among indptr[u:u+1]
      int64_t b = indptr[u], e = indptr[u + 1] - 1;
      bool found = false;
      while (b <= e) {
        int64_t m = (b + e) / 2;
        if (indices[m] == v) {
          found = true;
          break;
        } else if (indices[m] < v) {
          b = m + 1;
        } else {
          e = m - 1;
        }
      }

      if (!found) {
        row[tx] = u;
        col[tx] = v;
        break;
      }
    }

    tx += stride_x;
  }
}

template <typename DType>
struct IsNotMinusOne {
  __device__ __forceinline__ bool operator() (const DType& a) {
    return a != -1;
  }
};

template <DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling(
    const CSRMatrix& csr,
    int64_t num_samples,
    int num_trials,
    bool exclude_self_loops) {
  auto ctx = csr.indptr->ctx;
  auto dtype = csr.indptr->dtype;
  const int64_t num_row = csr.num_rows;
  const int64_t num_col = csr.num_cols;
  IdArray row = IdArray::Full<IdType>(-1, num_samples, ctx);
  IdArray col = IdArray::Full<IdType>(-1, num_samples, ctx);
  IdArray out_row = IdArray::Empty({num_samples}, dtype, ctx);
  IdArray out_col = IdArray::Empty({num_samples}, dtype, ctx);
  IdType* row_data = row.Ptr<IdType>();
  IdType* col_data = col.Ptr<IdType>();
  IdType* out_row_data = out_row.Ptr<IdType>();
  IdType* out_col_data = out_col.Ptr<IdType>();
  auto device = runtime::DeviceAPI::Get(ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int nt = cuda::FindNumThreads(num_samples);
  const int nb = (num_samples + nt - 1) / nt;

  CUDA_KERNEL_CALL(cuda::_GlobalUniformNegativeSamplingKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(),
      row_data, col_data, num_row, num_col, num_samples, num_trials, exclude_self_loops,
      RandomEngine::ThreadLocal()->RandInt32());

  size_t tmp_size_row = 0, tmp_size_col = 0;
  size_t* num_out_row = static_cast<size_t*>(device->AllocWorkspace(ctx, sizeof(size_t)));
  size_t* num_out_col = static_cast<size_t*>(device->AllocWorkspace(ctx, sizeof(size_t)));
  IsNotMinusOne op;
  CUDA_CALL(cub::DeviceSelect::If(
        nullptr, tmp_size_row, row_data, out_row_data, num_out_row, num_samples, op));
  CUDA_CALL(cub::DeviceSelect::If(
        nullptr, tmp_size_col, col_data, out_col_data, num_out_col, num_samples, op));
  void* tmp_row = device->AllocWorkspace(ctx, tmp_size_row);
  void* tmp_col = device->AllocWorkspace(ctx, tmp_size_col);
  CUDA_CALL(cub::DeviceSelect::If(
        tmp_row, tmp_size_row, row_data, out_row_data, num_out_row, num_samples, op));
  CUDA_CALL(cub::DeviceSelect::If(
        tmp_col, tmp_size_col, col_data, out_col_data, num_out_col, num_samples, op));
  size_t num_out = GetCUDAScalar(device, ctx, num_out_row, static_cast<cudaStream_t>(0));
  BUG_IF_FAIL(num_out == GetCUDAScalar(device, ctx, num_out_col, static_cast<cudaStream_t>(0)));

  return {out_row.CreateView({num_out}, dtype), out_col.CreateView({num_out}, dtype)};
}

template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<kDLGPU, int32_t>(
    CSRMatrix, int64_t, int, bool);
template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<kDLGPU, int64_t>(
    CSRMatrix, int64_t, int, bool);

};  // namespace impl
};  // namespace sampling
};  // namespace dgl
