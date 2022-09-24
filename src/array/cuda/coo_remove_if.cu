/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/coo_remove_if.cc
 * \brief COO matrix remove entries CPU implementation
 */
#include <dgl/array.h>
#include <utility>
#include <vector>
#include "./utils.h"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

namespace {

template <typename IdType, typename DType, typename BoolType>
__global__ void _GenerateFlagsKernel(
    int64_t n, const IdType* idx, const DType* values, DType criteria, BoolType* output) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < n) {
    output[tx] = (values[idx ? idx[tx] : tx] != criteria);
    tx += stride_x;
  }
}

};  // namespace

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix COORemoveIf(COOMatrix coo, NDArray values, DType criteria) {
  using namespace dgl::cuda;

  const auto idtype = coo.row->dtype;
  const auto ctx = coo.row->ctx;
  const int64_t nnz = coo.row->shape[0];
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdArray& eid = COOHasData(coo) ? coo.data :
    Range(0, nnz, sizeof(IdType) * 8, ctx);
  const IdType* data = coo.data.Ptr<IdType>();
  const DType* val = values.Ptr<DType>();
  IdArray new_row = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_col = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_eid = IdArray::Empty({nnz}, idtype, ctx);
  IdType* new_row_data = new_row.Ptr<IdType>();
  IdType* new_col_data = new_col.Ptr<IdType>();
  IdType* new_eid_data = new_eid.Ptr<IdType>();
  auto stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(ctx);

  int8_t* flags = static_cast<int8_t*>(device->AllocWorkspace(ctx, nnz));
  int nt = cuda::FindNumThreads(nnz);
  int nb = (nnz + nt - 1) / nt;
  CUDA_KERNEL_CALL((_GenerateFlagsKernel<IdType, DType, int8_t>),
      nb, nt, 0, stream,
      nnz, data, val, criteria, flags);

  int64_t* rst = static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
  MaskSelect(device, ctx, row, flags, new_row_data, nnz, rst, stream);
  MaskSelect(device, ctx, col, flags, new_col_data, nnz, rst, stream);
  MaskSelect(device, ctx, data, flags, new_eid_data, nnz, rst, stream);

  int64_t new_len = GetCUDAScalar(device, ctx, rst);

  device->FreeWorkspace(ctx, flags);
  device->FreeWorkspace(ctx, rst);
  return COOMatrix(
      coo.num_rows,
      coo.num_cols,
      new_row.CreateView({new_len}, idtype, 0),
      new_col.CreateView({new_len}, idtype, 0),
      new_eid.CreateView({new_len}, idtype, 0));
}

template COOMatrix COORemoveIf<kDGLCUDA, int32_t, int8_t>(COOMatrix, NDArray, int8_t);
template COOMatrix COORemoveIf<kDGLCUDA, int32_t, uint8_t>(COOMatrix, NDArray, uint8_t);
template COOMatrix COORemoveIf<kDGLCUDA, int32_t, float>(COOMatrix, NDArray, float);
template COOMatrix COORemoveIf<kDGLCUDA, int32_t, double>(COOMatrix, NDArray, double);
template COOMatrix COORemoveIf<kDGLCUDA, int64_t, int8_t>(COOMatrix, NDArray, int8_t);
template COOMatrix COORemoveIf<kDGLCUDA, int64_t, uint8_t>(COOMatrix, NDArray, uint8_t);
template COOMatrix COORemoveIf<kDGLCUDA, int64_t, float>(COOMatrix, NDArray, float);
template COOMatrix COORemoveIf<kDGLCUDA, int64_t, double>(COOMatrix, NDArray, double);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
