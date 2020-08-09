/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/array_op_impl.cu
 * \brief Array operator GPU implementation
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"
#include "../arith.h"

namespace dgl {
using runtime::NDArray;
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
  _BinaryElewiseKernel<IdType, Op><<<nb, nt, 0, thr_entry->stream>>>(
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
  _BinaryElewiseKernel<IdType, Op><<<nb, nt, 0, thr_entry->stream>>>(
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
  _BinaryElewiseKernel<IdType, Op><<<nb, nt, 0, thr_entry->stream>>>(
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
  _UnaryElewiseKernel<IdType, Op><<<nb, nt, 0, thr_entry->stream>>>(
      lhs_data, ret_data, len);
  return ret;
}

template IdArray UnaryElewise<kDLGPU, int32_t, arith::Neg>(IdArray lhs);
template IdArray UnaryElewise<kDLGPU, int64_t, arith::Neg>(IdArray lhs);

///////////////////////////// Full /////////////////////////////

template <typename IdType>
__global__ void _FullKernel(
    IdType* out, int64_t length, IdType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = val;
    tx += stride_x;
  }
}

template <DLDeviceType XPU, typename IdType>
IdArray Full(IdType val, int64_t length, DLContext ctx) {
  IdArray ret = NewIdArray(length, ctx, sizeof(IdType) * 8);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int nt = cuda::FindNumThreads(length);
  int nb = (length + nt - 1) / nt;
  _FullKernel<IdType><<<nb, nt, 0, thr_entry->stream>>>(ret_data, length, val);
  return ret;
}

template IdArray Full<kDLGPU, int32_t>(int32_t val, int64_t length, DLContext ctx);
template IdArray Full<kDLGPU, int64_t>(int64_t val, int64_t length, DLContext ctx);


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
  _RangeKernel<IdType><<<nb, nt, 0, thr_entry->stream>>>(ret_data, low, length);
  return ret;
}

template IdArray Range<kDLGPU, int32_t>(int32_t, int32_t, DLContext);
template IdArray Range<kDLGPU, int64_t>(int64_t, int64_t, DLContext);

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
    _CastKernel<IdType, int32_t><<<nb, nt, 0, thr_entry->stream>>>(
        static_cast<IdType*>(arr->data), static_cast<int32_t*>(ret->data), length);
  } else {
    _CastKernel<IdType, int64_t><<<nb, nt, 0, thr_entry->stream>>>(
        static_cast<IdType*>(arr->data), static_cast<int64_t*>(ret->data), length);
  }
  return ret;
}


template IdArray AsNumBits<kDLGPU, int32_t>(IdArray arr, uint8_t bits);
template IdArray AsNumBits<kDLGPU, int64_t>(IdArray arr, uint8_t bits);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
