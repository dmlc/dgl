/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/array_op_impl.cu
 * \brief Array operator GPU implementation
 */
#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_hashtable.cuh"
#include "./utils.h"
#include "../arith.h"

namespace dgl {
using runtime::NDArray;
using namespace runtime::cuda;
namespace aten {
namespace impl {

///////////////////////////// BinaryElewise /////////////////////////////

template <typename IdType, typename Op>
__global__ void _BinaryElewiseKernel(
    const IdType* lhs, const IdType* rhs, IdType* out, int64_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = Op::Call(lhs[tx], rhs[tx]);
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdArray rhs) {
  const int64_t len = lhs->shape[0];
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  const IdType* rhs_data = static_cast<IdType*>(rhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL((_BinaryElewiseKernel<IdType, Op>),
      nb, nt, 0, thr_entry->stream,
      lhs_data, rhs_data, ret_data, len);
  return ret;
}

template IdArray BinaryElewise<kDLGPU, int32_t, arith::Add>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Sub>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Mul>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Div>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Mod>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::GT>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::LT>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::GE>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::LE>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::EQ>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::NE>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Add>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Sub>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Mul>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Div>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Mod>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::GT>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::LT>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::GE>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::LE>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::EQ>(IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::NE>(IdArray lhs, IdArray rhs);


template <typename IdType, typename Op>
__global__ void _BinaryElewiseKernel(
    const IdType* lhs, IdType rhs, IdType* out, int64_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = Op::Call(lhs[tx], rhs);
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs) {
  const int64_t len = lhs->shape[0];
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL((_BinaryElewiseKernel<IdType, Op>),
      nb, nt, 0, thr_entry->stream,
      lhs_data, rhs, ret_data, len);
  return ret;
}

template IdArray BinaryElewise<kDLGPU, int32_t, arith::Add>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Sub>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Mul>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Div>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Mod>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::GT>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::LT>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::GE>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::LE>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::EQ>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::NE>(IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Add>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Sub>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Mul>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Div>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Mod>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::GT>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::LT>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::GE>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::LE>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::EQ>(IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::NE>(IdArray lhs, int64_t rhs);



template <typename IdType, typename Op>
__global__ void _BinaryElewiseKernel(
    IdType lhs, const IdType* rhs, IdType* out, int64_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = Op::Call(lhs, rhs[tx]);
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdType lhs, IdArray rhs) {
  const int64_t len = rhs->shape[0];
  IdArray ret = NewIdArray(rhs->shape[0], rhs->ctx, rhs->dtype.bits);
  const IdType* rhs_data = static_cast<IdType*>(rhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL((_BinaryElewiseKernel<IdType, Op>),
      nb, nt, 0, thr_entry->stream,
      lhs, rhs_data, ret_data, len);
  return ret;
}

template IdArray BinaryElewise<kDLGPU, int32_t, arith::Add>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Sub>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Mul>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Div>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::Mod>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::GT>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::LT>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::GE>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::LE>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::EQ>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int32_t, arith::NE>(int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Add>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Sub>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Mul>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Div>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::Mod>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::GT>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::LT>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::GE>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::LE>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::EQ>(int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDLGPU, int64_t, arith::NE>(int64_t lhs, IdArray rhs);

template <typename IdType, typename Op>
__global__ void _UnaryElewiseKernel(
    const IdType* lhs, IdType* out, int64_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = Op::Call(lhs[tx]);
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray lhs) {
  const int64_t len = lhs->shape[0];
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL((_UnaryElewiseKernel<IdType, Op>),
      nb, nt, 0, thr_entry->stream,
      lhs_data, ret_data, len);
  return ret;
}

template IdArray UnaryElewise<kDLGPU, int32_t, arith::Neg>(IdArray lhs);
template IdArray UnaryElewise<kDLGPU, int64_t, arith::Neg>(IdArray lhs);

///////////////////////////// Full /////////////////////////////

template <typename DType>
__global__ void _FullKernel(
    DType* out, int64_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = val;
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename DType>
NDArray Full(DType val, int64_t length, DLContext ctx) {
  NDArray ret = NDArray::Empty({length}, DLDataTypeTraits<DType>::dtype, ctx);
  DType* ret_data = static_cast<DType*>(ret->data);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(length);
  int nb = (length + nt - 1) / nt;
  CUDA_KERNEL_CALL((_FullKernel<DType>), nb, nt, 0, thr_entry->stream,
      ret_data, length, val);
  return ret;
}

template IdArray Full<kDLGPU, int32_t>(int32_t val, int64_t length, DLContext ctx);
template IdArray Full<kDLGPU, int64_t>(int64_t val, int64_t length, DLContext ctx);
template IdArray Full<kDLGPU, float>(float val, int64_t length, DLContext ctx);
template IdArray Full<kDLGPU, double>(double val, int64_t length, DLContext ctx);


///////////////////////////// Range /////////////////////////////

template <typename IdType>
__global__ void _RangeKernel(IdType* out, IdType low, IdType length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = low + tx;
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DLContext ctx) {
  CHECK(high >= low) << "high must be bigger than low";
  const IdType length = high - low;
  IdArray ret = NewIdArray(length, ctx, sizeof(IdType) * 8);
  if (length == 0)
    return ret;
  IdType* ret_data = static_cast<IdType*>(ret->data);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(length);
  int nb = (length + nt - 1) / nt;
  CUDA_KERNEL_CALL((_RangeKernel<IdType>),
      nb, nt, 0, thr_entry->stream,
      ret_data, low, length);
  return ret;
}

template IdArray Range<kDLGPU, int32_t>(int32_t, int32_t, DLContext);
template IdArray Range<kDLGPU, int64_t>(int64_t, int64_t, DLContext);

///////////////////////////// Relabel_ //////////////////////////////

template <DLDeviceType XPU, typename IdType>
IdArray Relabel_(const std::vector<IdArray>& arrays) {
  cudaStream_t stream = 0;
  const auto& ctx = arrays[0]->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);

  int64_t max_vertex_cnt = 0;
  for (IdArray arr: arrays) {
    max_vertex_cnt += arr->shape[0];
  }

  // gather all the nodes
  IdArray all_nodes = NewIdArray(max_vertex_cnt, ctx, sizeof(IdType)*8);
  int64_t offset = 0;
  for (IdArray arr: arrays) {
    device->CopyDataFromTo(
      arr.Ptr<IdType>(), 0,
      all_nodes.Ptr<IdType>(), offset,
      sizeof(IdType)*arr->shape[0],
      arr->ctx,
      all_nodes->ctx,
      arr->dtype,
      stream);
    offset += sizeof(IdType)*arr->shape[0];
  }

  // build node maps and relabel
  OrderedHashTable<IdType> node_map(max_vertex_cnt, ctx, stream);
  int64_t num_induced_nodes = 0;
  int64_t * count_unique_device = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)));
  IdArray induced_nodes = NewIdArray(max_vertex_cnt, ctx, sizeof(IdType)*8);

  CUDA_CALL(cudaMemsetAsync(
    count_unique_device,
    0,
    sizeof(*count_unique_device),
    stream));
  CHECK_EQ(all_nodes->ctx.device_type, kDLGPU);
  node_map.FillWithDuplicates(
    all_nodes.Ptr<IdType>(),
    all_nodes->shape[0],
    induced_nodes.Ptr<IdType>(),
    count_unique_device,
    stream);

  device->CopyDataFromTo(
    count_unique_device, 0,
    &num_induced_nodes, 0,
    sizeof(num_induced_nodes),
    ctx,
    DGLContext{kDLCPU, 0},
    DGLType{kDLInt, 64, 1},
    stream);
  device->StreamSync(ctx, stream);
  device->FreeWorkspace(ctx, count_unique_device);

  // resize the induced nodes
  induced_nodes->shape[0] = num_induced_nodes;

  return induced_nodes;
}

template IdArray Relabel_<kDLGPU, int32_t>(const std::vector<IdArray>& arrays);
template IdArray Relabel_<kDLGPU, int64_t>(const std::vector<IdArray>& arrays);

///////////////////////////// AsNumBits /////////////////////////////

template <typename InType, typename OutType>
__global__ void _CastKernel(const InType* in, OutType* out, size_t length) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = in[tx];
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
IdArray AsNumBits(IdArray arr, uint8_t bits) {
  const std::vector<int64_t> shape(arr->shape, arr->shape + arr->ndim);
  IdArray ret = IdArray::Empty(shape, DLDataType{kDLInt, bits, 1}, arr->ctx);
  const int64_t length = ret.NumElements();
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(length);
  int nb = (length + nt - 1) / nt;
  if (bits == 32) {
    CUDA_KERNEL_CALL((_CastKernel<IdType, int32_t>),
        nb, nt, 0, thr_entry->stream,
        static_cast<IdType*>(arr->data), static_cast<int32_t*>(ret->data), length);
  } else {
    CUDA_KERNEL_CALL((_CastKernel<IdType, int64_t>),
        nb, nt, 0, thr_entry->stream,
        static_cast<IdType*>(arr->data), static_cast<int64_t*>(ret->data), length);
  }
  return ret;
}


template IdArray AsNumBits<kDLGPU, int32_t>(IdArray arr, uint8_t bits);
template IdArray AsNumBits<kDLGPU, int64_t>(IdArray arr, uint8_t bits);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
