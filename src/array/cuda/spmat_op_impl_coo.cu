/*!
 *  Copyright (c) 2021 by contributors.
 * \file array/cuda/spmat_op_impl_coo.cu
 * \brief COO operator GPU implementation
 */
#include <dgl/array.h>
#include <vector>
#include <unordered_set>
#include <numeric>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {


template <typename IdType>
__device__ void _warpReduce(volatile IdType *sdata, IdType tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

template <typename IdType>
__global__ void _COOGetRowNNZKernel(
    const IdType* __restrict__ row_indices,
    IdType* __restrict__ glb_cnt,
    const IdType row_query,
    IdType nnz) {
  extern __shared__ IdType local_cnt[];
  IdType tx = threadIdx.x;
  IdType bx = blockIdx.x;
  local_cnt[tx] = 0;
  if (bx * blockDim.x + tx < nnz)
    local_cnt[tx] = (row_indices[bx * blockDim.x + tx] == row_query);
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 512];
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 256];
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 128];
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 64];
  __syncthreads();

  _warpReduce(local_cnt, tx);
  if (tx == 0) {
    cuda::AtomicAdd(glb_cnt, local_cnt[tx]); 
  }
}

template <DLDeviceType XPU, typename IdType>
int64_t COOGetRowNNZ(COOMatrix coo, int64_t row) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const auto& ctx = csr.indptr->ctx;
  int nnz = coo.row->shape[0];
  int nt = 1024;
  int nb = cuda::FindNumBlocks<'x'>((nnz + nt - 1) / nt);
  int smem_size = sizeof(IdType) * nt;
  auto rows = coo.row;
  NDArray rst = NDArray::Empty({1}, rows->dtype, rows->ctx)
  CUDA_KERNEL_CALL(_COOGetRowNNZKernel,
      nb, nt, smem_size, thr_entry->stream,
      rows.Ptr<IdType>(), rst.Ptr<IdType>(),
      row, nnz);
  rst = rst.CopyTo(DLContext{kDLCPU, 0});
  return rst.Ptr<IdType>();
}

template int64_t COOGetRowNNZ<kDLGPU, int32_t>(COOMatrix, int64_t);
template int64_t COOGetRowNNZ<kDLGPU, int64_t>(COOMatrix, int64_t);

template <typename IdType>
void _COOGetBatchRowNNZKernel(
    const IdType* __restrict__ row_indices,
    const IdType* __restrict__ row_queries,
    IdType __restrict__ *cnts,
    IdType nnz) {
  extern __shared__ IdType local_cnt[];
  IdType tx = threadIdx.x;
  IdType bx = blockIdx.x;
  IdType query = row_queries[bx];
  local_cnt[tx] = 0;
  for (int i = tx; i < nnz; i += blockDim.x)
    local_cnt[tx] += (row_indices[i] == query);
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 512];
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 256];
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 128];
  __syncthreads();

  local_cnt[tx] += local_cnt[tx + 64];
  __syncthreads();

  _warpReduce(local_cnt, tx);
  if (tx == 0) {
    cnts[bx] = local_cnt[tx];
  }
}

template <DLDeviceType XPU, typename IdType>
NDArray COOGetRowNNZ(COOMatrix coo, NDArray rows) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const auto& ctx = csr.indptr->ctx;
  int nnz = coo.row->shape[0];
  int num_queries = rows->shape[0];
  int nt = 1024;
  int nb = cuda::FindNumBlocks<'x'>((num_queries + nt - 1) / nt);
  int smem_size = sizeof(IdType) * nt;
  NDArray rst = NDArray::Empty({num_queries}, rows->dtype, rows->ctx)
  CUDA_KERNEL_CALL(_COOGetBatchRowNNZKernel,
      nb, nt, smem_size, thr_entry->stream,
      coo.rows.Ptr<IdType>(), rows.Ptr<IdType>(), rst.Ptr<IdType>(),
      nnz);
  return rst;
}

template NDArray COOGetRowNNZ<kDLGPU, int32_t>(COOMatrix, NDArray);
template NDArray COOGetRowNNZ<kDLGPU, int64_t>(COOMatrix, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl