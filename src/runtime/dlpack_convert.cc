/**
 *  Copyright (c) 2022 by Contributors
 * @file src/runtime/dlpack_convert.cc
 * @brief Conversion between NDArray and DLPack.
 */
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/dlpack_convert.h>
#include <dgl/runtime/ndarray.h>
#include <dlpack/dlpack.h>

#include <cstdint>

#include "runtime_base.h"

// deleter for arrays used by DLPack exporter
extern "C" void NDArrayDLPackDeleter(DLManagedTensor* tensor);

namespace dgl {
namespace runtime {

void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
  static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
  delete tensor;
}

inline DGLContext ToDGLContext(const DLDevice& device) {
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device.device_type);
  ctx.device_id = device.device_id;
  return ctx;
}

inline DGLDataType ToDGLDataType(const DLDataType& src) {
  DGLDataType ret;
  ret.code = src.code;
  ret.bits = src.bits;
  ret.lanes = src.lanes;
  return ret;
}

inline DLDevice ToDLDevice(const DGLContext& ctx) {
  DLDevice device;
  device.device_type = static_cast<DLDeviceType>(ctx.device_type);
  device.device_id = ctx.device_id;
  return device;
}

inline DLDataType ToDLDataType(const DGLDataType& src) {
  DLDataType ret;
  ret.code = src.code;
  ret.bits = src.bits;
  ret.lanes = src.lanes;
  return ret;
}

NDArray DLPackConvert::FromDLPack(DLManagedTensor* tensor) {
  NDArray::Container* data = new NDArray::Container();
  data->deleter = DLPackConvert::DLPackDeleter;
  data->manager_ctx = tensor;
  data->dl_tensor.data = tensor->dl_tensor.data;
  data->dl_tensor.ctx = ToDGLContext(tensor->dl_tensor.device);
  data->dl_tensor.ndim = tensor->dl_tensor.ndim;
  data->dl_tensor.dtype = ToDGLDataType(tensor->dl_tensor.dtype);
  data->dl_tensor.shape = tensor->dl_tensor.shape;
  data->dl_tensor.strides = tensor->dl_tensor.strides;
  data->dl_tensor.byte_offset = tensor->dl_tensor.byte_offset;

  return NDArray(data);
}

void DLPackConvert::DLPackDeleter(NDArray::Container* ptr) {
  // if the array is pinned by dgl, unpin it before freeing
  if (ptr->pinned_by_dgl_) NDArray::UnpinContainer(ptr);
  DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
  if (tensor->deleter != nullptr) {
    (*tensor->deleter)(tensor);
  }
  delete ptr;
}

DLManagedTensor* ContainerToDLPack(NDArray::Container* from) {
  CHECK(from != nullptr);
  DLManagedTensor* ret = new DLManagedTensor();
  ret->dl_tensor.data = from->dl_tensor.data;
  ret->dl_tensor.device = ToDLDevice(from->dl_tensor.ctx);
  ret->dl_tensor.ndim = from->dl_tensor.ndim;
  ret->dl_tensor.dtype = ToDLDataType(from->dl_tensor.dtype);
  ret->dl_tensor.shape = from->dl_tensor.shape;
  ret->dl_tensor.strides = from->dl_tensor.strides;
  ret->dl_tensor.byte_offset = from->dl_tensor.byte_offset;

  ret->manager_ctx = from;
  from->IncRef();
  ret->deleter = NDArrayDLPackDeleter;
  return ret;
}

DLManagedTensor* DLPackConvert::ToDLPack(const NDArray& from) {
  return ContainerToDLPack(from.data_);
}

}  // namespace runtime
}  // namespace dgl

using namespace dgl::runtime;

void DGLDLManagedTensorCallDeleter(DLManagedTensor* dltensor) {
  (*(dltensor->deleter))(dltensor);
}

inline bool IsAligned(const void* ptr, std::uintptr_t alignment) noexcept {
  auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
  return !(iptr % alignment);
}

int DGLArrayFromDLPack(DLManagedTensor* from, DGLArrayHandle* out) {
  API_BEGIN();
  *out = NDArray::Internal::MoveAsDGLArray(DLPackConvert::FromDLPack(from));
  API_END();
}

int DGLArrayToDLPack(
    DGLArrayHandle from, DLManagedTensor** out, int alignment) {
  API_BEGIN();
  auto* nd_container = reinterpret_cast<NDArray::Container*>(from);
  DGLArray* nd = &(nd_container->dl_tensor);
  // If the source DGLArray is not aligned, we should create a new aligned one
  if (alignment != 0 && !IsAligned(nd->data, alignment)) {
    std::vector<int64_t> shape_vec(nd->shape, nd->shape + nd->ndim);
    NDArray copy_ndarray = NDArray::Empty(shape_vec, nd->dtype, nd->ctx);
    copy_ndarray.CopyFrom(nd);
    *out = DLPackConvert::ToDLPack(copy_ndarray);
  } else {
    *out = ContainerToDLPack(nd_container);
  }
  API_END();
}
