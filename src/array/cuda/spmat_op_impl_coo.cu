/**
 *  Copyright (c) 2021 by contributors.
 * @file array/cuda/spmat_op_impl_coo.cu
 * @brief COO operator GPU implementation
 */
#include <dgl/array.h>

#include <numeric>
#include <unordered_set>
#include <vector>

#include "../../runtime/cuda/cuda_common.h"
#include "./atomic.cuh"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;
using namespace cuda;

namespace aten {
namespace impl {

template <typename IdType>
__device__ void _warpReduce(volatile IdType* sdata, IdType tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

template <typename IdType>
__global__ void _COOGetRowNNZKernel(
    const IdType* __restrict__ row_indices, IdType* __restrict__ glb_cnt,
    const int64_t row_query, IdType nnz) {
  __shared__ IdType local_cnt[1024];
  IdType tx = threadIdx.x;
  IdType bx = blockIdx.x;
  local_cnt[tx] = 0;
  IdType start = bx * blockDim.x;
  while (start < nnz) {
    if (start + tx < nnz)
      local_cnt[tx] = (row_indices[start + tx] == row_query);
    __syncthreads();
    if (tx < 512) {
      local_cnt[tx] += local_cnt[tx + 512];
      __syncthreads();
    }
    if (tx < 256) {
      local_cnt[tx] += local_cnt[tx + 256];
      __syncthreads();
    }
    if (tx < 128) {
      local_cnt[tx] += local_cnt[tx + 128];
      __syncthreads();
    }
    if (tx < 64) {
      local_cnt[tx] += local_cnt[tx + 64];
      __syncthreads();
    }
    if (tx < 32) {
      _warpReduce(local_cnt, tx);
    }
    if (tx == 0) {
      cuda::AtomicAdd(glb_cnt, local_cnt[tx]);
    }
    start += blockDim.x * gridDim.x;
  }
}

template <DGLDeviceType XPU, typename IdType>
int64_t COOGetRowNNZ(COOMatrix coo, int64_t row) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto& ctx = coo.row->ctx;
  IdType nnz = coo.row->shape[0];
  IdType nt = 1024;
  IdType nb = dgl::cuda::FindNumBlocks<'x'>((nnz + nt - 1) / nt);
  NDArray rst = NDArray::Empty({1}, coo.row->dtype, coo.row->ctx);
  _Fill(rst.Ptr<IdType>(), 1, IdType(0));
  CUDA_KERNEL_CALL(
      _COOGetRowNNZKernel, nb, nt, 0, stream, coo.row.Ptr<IdType>(),
      rst.Ptr<IdType>(), row, nnz);
  rst = rst.CopyTo(DGLContext{kDGLCPU, 0});
  return *rst.Ptr<IdType>();
}

template int64_t COOGetRowNNZ<kDGLCUDA, int32_t>(COOMatrix, int64_t);
template int64_t COOGetRowNNZ<kDGLCUDA, int64_t>(COOMatrix, int64_t);

template <typename IdType>
__global__ void _COOGetAllRowNNZKernel(
    const IdType* __restrict__ row_indices, IdType* __restrict__ glb_cnts,
    IdType nnz) {
  IdType eid = blockIdx.x * blockDim.x + threadIdx.x;
  while (eid < nnz) {
    IdType row = row_indices[eid];
    cuda::AtomicAdd(glb_cnts + row, IdType(1));
    eid += blockDim.x * gridDim.x;
  }
}

template <DGLDeviceType XPU, typename IdType>
NDArray COOGetRowNNZ(COOMatrix coo, NDArray rows) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto& ctx = coo.row->ctx;
  IdType nnz = coo.row->shape[0];
  IdType num_rows = coo.num_rows;
  IdType num_queries = rows->shape[0];
  if (num_queries == 1) {
    auto rows_cpu = rows.CopyTo(DGLContext{kDGLCPU, 0});
    int64_t row = *rows_cpu.Ptr<IdType>();
    IdType nt = 1024;
    IdType nb = dgl::cuda::FindNumBlocks<'x'>((nnz + nt - 1) / nt);
    NDArray rst = NDArray::Empty({1}, coo.row->dtype, coo.row->ctx);
    _Fill(rst.Ptr<IdType>(), 1, IdType(0));
    CUDA_KERNEL_CALL(
        _COOGetRowNNZKernel, nb, nt, 0, stream, coo.row.Ptr<IdType>(),
        rst.Ptr<IdType>(), row, nnz);
    return rst;
  } else {
    IdType nt = 1024;
    IdType nb = dgl::cuda::FindNumBlocks<'x'>((nnz + nt - 1) / nt);
    NDArray in_degrees = NDArray::Empty({num_rows}, rows->dtype, rows->ctx);
    _Fill(in_degrees.Ptr<IdType>(), num_rows, IdType(0));
    CUDA_KERNEL_CALL(
        _COOGetAllRowNNZKernel, nb, nt, 0, stream, coo.row.Ptr<IdType>(),
        in_degrees.Ptr<IdType>(), nnz);
    return IndexSelect(in_degrees, rows);
  }
}

template NDArray COOGetRowNNZ<kDGLCUDA, int32_t>(COOMatrix, NDArray);
template NDArray COOGetRowNNZ<kDGLCUDA, int64_t>(COOMatrix, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
