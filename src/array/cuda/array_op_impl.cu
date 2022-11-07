/**
 *  Copyright (c) 2020-2021 by Contributors
 * @file array/cuda/array_op_impl.cu
 * @brief Array operator GPU implementation
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../../runtime/cuda/cuda_hashtable.cuh"
#include "../arith.h"
#include "./utils.h"

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

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdArray rhs) {
  const int64_t len = lhs->shape[0];
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  const IdType* rhs_data = static_cast<IdType*>(rhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      (_BinaryElewiseKernel<IdType, Op>), nb, nt, 0, stream, lhs_data, rhs_data,
      ret_data, len);
  return ret;
}

template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Add>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Sub>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Mul>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Div>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Mod>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::GT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::LT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::GE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::LE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::EQ>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::NE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Add>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Sub>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Mul>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Div>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Mod>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::GT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::LT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::GE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::LE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::EQ>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::NE>(
    IdArray lhs, IdArray rhs);

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

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs) {
  const int64_t len = lhs->shape[0];
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      (_BinaryElewiseKernel<IdType, Op>), nb, nt, 0, stream, lhs_data, rhs,
      ret_data, len);
  return ret;
}

template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Add>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Sub>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Mul>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Div>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Mod>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::GT>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::LT>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::GE>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::LE>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::EQ>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::NE>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Add>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Sub>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Mul>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Div>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Mod>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::GT>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::LT>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::GE>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::LE>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::EQ>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::NE>(
    IdArray lhs, int64_t rhs);

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

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdType lhs, IdArray rhs) {
  const int64_t len = rhs->shape[0];
  IdArray ret = NewIdArray(rhs->shape[0], rhs->ctx, rhs->dtype.bits);
  const IdType* rhs_data = static_cast<IdType*>(rhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      (_BinaryElewiseKernel<IdType, Op>), nb, nt, 0, stream, lhs, rhs_data,
      ret_data, len);
  return ret;
}

template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Add>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Sub>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Mul>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Div>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::Mod>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::GT>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::LT>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::GE>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::LE>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::EQ>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int32_t, arith::NE>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Add>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Sub>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Mul>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Div>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::Mod>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::GT>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::LT>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::GE>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::LE>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::EQ>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCUDA, int64_t, arith::NE>(
    int64_t lhs, IdArray rhs);

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

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray lhs) {
  const int64_t len = lhs->shape[0];
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = cuda::FindNumThreads(len);
  int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      (_UnaryElewiseKernel<IdType, Op>), nb, nt, 0, stream, lhs_data, ret_data,
      len);
  return ret;
}

template IdArray UnaryElewise<kDGLCUDA, int32_t, arith::Neg>(IdArray lhs);
template IdArray UnaryElewise<kDGLCUDA, int64_t, arith::Neg>(IdArray lhs);

///////////////////////////// Full /////////////////////////////

template <typename DType>
__global__ void _FullKernel(DType* out, int64_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = val;
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename DType>
NDArray Full(DType val, int64_t length, DGLContext ctx) {
  NDArray ret = NDArray::Empty({length}, DGLDataTypeTraits<DType>::dtype, ctx);
  DType* ret_data = static_cast<DType*>(ret->data);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = cuda::FindNumThreads(length);
  int nb = (length + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      (_FullKernel<DType>), nb, nt, 0, stream, ret_data, length, val);
  return ret;
}

template IdArray Full<kDGLCUDA, int32_t>(
    int32_t val, int64_t length, DGLContext ctx);
template IdArray Full<kDGLCUDA, int64_t>(
    int64_t val, int64_t length, DGLContext ctx);
template IdArray Full<kDGLCUDA, __half>(
    __half val, int64_t length, DGLContext ctx);
#if BF16_ENABLED
template IdArray Full<kDGLCUDA, __nv_bfloat16>(
    __nv_bfloat16 val, int64_t length, DGLContext ctx);
#endif  // BF16_ENABLED
template IdArray Full<kDGLCUDA, float>(
    float val, int64_t length, DGLContext ctx);
template IdArray Full<kDGLCUDA, double>(
    double val, int64_t length, DGLContext ctx);

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

template <DGLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DGLContext ctx) {
  CHECK(high >= low) << "high must be bigger than low";
  const IdType length = high - low;
  IdArray ret = NewIdArray(length, ctx, sizeof(IdType) * 8);
  if (length == 0) return ret;
  IdType* ret_data = static_cast<IdType*>(ret->data);
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = cuda::FindNumThreads(length);
  int nb = (length + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      (_RangeKernel<IdType>), nb, nt, 0, stream, ret_data, low, length);
  return ret;
}

template IdArray Range<kDGLCUDA, int32_t>(int32_t, int32_t, DGLContext);
template IdArray Range<kDGLCUDA, int64_t>(int64_t, int64_t, DGLContext);

///////////////////////////// Relabel_ //////////////////////////////

template <typename IdType>
__global__ void _RelabelKernel(
    IdType* out, int64_t length, DeviceOrderedHashTable<IdType> table) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;

  while (tx < length) {
    out[tx] = table.Search(out[tx])->local;
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
IdArray Relabel_(const std::vector<IdArray>& arrays) {
  IdArray all_nodes = Concat(arrays);
  const int64_t total_length = all_nodes->shape[0];

  if (total_length == 0) {
    return all_nodes;
  }

  const auto& ctx = arrays[0]->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  // build node maps and get the induced nodes
  OrderedHashTable<IdType> node_map(total_length, ctx, stream);
  int64_t num_induced = 0;
  int64_t* num_induced_device =
      static_cast<int64_t*>(device->AllocWorkspace(ctx, sizeof(int64_t)));
  IdArray induced_nodes = NewIdArray(total_length, ctx, sizeof(IdType) * 8);

  CUDA_CALL(cudaMemsetAsync(
      num_induced_device, 0, sizeof(*num_induced_device), stream));

  node_map.FillWithDuplicates(
      all_nodes.Ptr<IdType>(), all_nodes->shape[0], induced_nodes.Ptr<IdType>(),
      num_induced_device, stream);
  // copy using the internal current stream
  device->CopyDataFromTo(
      num_induced_device, 0, &num_induced, 0, sizeof(num_induced), ctx,
      DGLContext{kDGLCPU, 0}, DGLDataType{kDGLInt, 64, 1});

  device->StreamSync(ctx, stream);
  device->FreeWorkspace(ctx, num_induced_device);

  // resize the induced nodes
  induced_nodes->shape[0] = num_induced;

  // relabel
  const int nt = 128;
  for (IdArray arr : arrays) {
    const int64_t length = arr->shape[0];
    int nb = (length + nt - 1) / nt;
    CUDA_KERNEL_CALL(
        (_RelabelKernel<IdType>), nb, nt, 0, stream, arr.Ptr<IdType>(), length,
        node_map.DeviceHandle());
  }

  return induced_nodes;
}

template IdArray Relabel_<kDGLCUDA, int32_t>(
    const std::vector<IdArray>& arrays);
template IdArray Relabel_<kDGLCUDA, int64_t>(
    const std::vector<IdArray>& arrays);

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

template <DGLDeviceType XPU, typename IdType>
IdArray AsNumBits(IdArray arr, uint8_t bits) {
  const std::vector<int64_t> shape(arr->shape, arr->shape + arr->ndim);
  IdArray ret = IdArray::Empty(shape, DGLDataType{kDGLInt, bits, 1}, arr->ctx);
  const int64_t length = ret.NumElements();
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  int nt = cuda::FindNumThreads(length);
  int nb = (length + nt - 1) / nt;
  if (bits == 32) {
    CUDA_KERNEL_CALL(
        (_CastKernel<IdType, int32_t>), nb, nt, 0, stream,
        static_cast<IdType*>(arr->data), static_cast<int32_t*>(ret->data),
        length);
  } else {
    CUDA_KERNEL_CALL(
        (_CastKernel<IdType, int64_t>), nb, nt, 0, stream,
        static_cast<IdType*>(arr->data), static_cast<int64_t*>(ret->data),
        length);
  }
  return ret;
}

template IdArray AsNumBits<kDGLCUDA, int32_t>(IdArray arr, uint8_t bits);
template IdArray AsNumBits<kDGLCUDA, int64_t>(IdArray arr, uint8_t bits);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
