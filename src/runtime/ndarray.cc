/*!
 *  Copyright (c) 2017 by Contributors
 * \file ndarray.cc
 * \brief NDArray container infratructure.
 */
#include <string.h>
#include <dmlc/logging.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/device_api.h>
#include "runtime_base.h"

// deleter for arrays used by DLPack exporter
extern "C" void NDArrayDLPackDeleter(DLManagedTensor* tensor);

namespace dgl {
namespace runtime {

inline void VerifyDataType(DLDataType dtype) {
  CHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    CHECK_EQ(dtype.bits % 8, 0);
  } else {
    CHECK_EQ(dtype.bits % 8, 0);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataSize(const DLTensor& arr) {
  size_t size = 1;
  for (dgl_index_t i = 0; i < arr.ndim; ++i) {
    size *= arr.shape[i];
  }
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

struct NDArray::Internal {
  // Default deleter for the container
  static void DefaultDeleter(NDArray::Container* ptr) {
    using dgl::runtime::NDArray;
    if (ptr->manager_ctx != nullptr) {
      static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
#ifndef _WIN32
    } else if (ptr->mem) {
      ptr->mem = nullptr;
#endif  // _WIN32
    } else if (ptr->dl_tensor.data != nullptr) {
      dgl::runtime::DeviceAPI::Get(ptr->dl_tensor.ctx)->FreeDataSpace(
          ptr->dl_tensor.ctx, ptr->dl_tensor.data);
    }
    delete ptr;
  }
  // Deleter for NDArray converted from DLPack
  // This is used from data which is passed from external DLPack(DLManagedTensor)
  // that are not allocated inside of DGL.
  // This enables us to create NDArray from memory allocated by other
  // frameworks that are DLPack compatible
  static void DLPackDeleter(NDArray::Container* ptr) {
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete ptr;
  }
  // Local create function which allocates tensor metadata
  // but does not allocate space for the data.
  static NDArray Create(std::vector<int64_t> shape,
                        DLDataType dtype,
                        DLContext ctx) {
    VerifyDataType(dtype);
    // critical zone
    NDArray::Container* data = new NDArray::Container();
    data->deleter = DefaultDeleter;
    NDArray ret(data);
    ret.data_ = data;
    // RAII now in effect
    // setup shape
    data->shape_ = std::move(shape);
    data->dl_tensor.shape = dmlc::BeginPtr(data->shape_);
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup stride (this should be optional, but some framework
    //   does not support NULL stride and thus will crash the program).
    data->stride_.resize(data->dl_tensor.ndim, 1);
    for (int i = data->dl_tensor.ndim - 2; i >= 0; --i) {
      data->stride_[i] = data->shape_[i+1] * data->stride_[i+1];
    }
    data->dl_tensor.strides = dmlc::BeginPtr(data->stride_);
    // setup dtype
    data->dl_tensor.dtype = dtype;
    // setup ctx
    data->dl_tensor.ctx = ctx;
    return ret;
  }
  // Implementation of API function
  static DLTensor* MoveAsDLTensor(NDArray arr) {
    DLTensor* tensor = const_cast<DLTensor*>(arr.operator->());
    CHECK(reinterpret_cast<DLTensor*>(arr.data_) == tensor);
    arr.data_ = nullptr;
    return tensor;
  }
  // Container to DLManagedTensor
  static DLManagedTensor* ToDLPack(NDArray::Container* from) {
    CHECK(from != nullptr);
    DLManagedTensor* ret = new DLManagedTensor();
    ret->dl_tensor = from->dl_tensor;
    ret->manager_ctx = from;
    from->IncRef();
    ret->deleter = NDArrayDLPackDeleter;
    return ret;
  }
};

size_t NDArray::GetSize() const {
  return GetDataSize(data_->dl_tensor);
}

bool NDArray::IsContiguous() const {
  CHECK(data_ != nullptr);
  if (data_->dl_tensor.strides == nullptr)
    return true;
  for (int i = 0; i < data_->dl_tensor.ndim - 1; ++i) {
    if (data_->dl_tensor.strides[i] !=
        data_->dl_tensor.shape[i+1] * data_->dl_tensor.strides[i+1])
      return false;
  }
  return data_->dl_tensor.strides[data_->dl_tensor.ndim - 1] == 1;
}

NDArray NDArray::CreateView(std::vector<int64_t> shape,
                            DLDataType dtype,
                            int64_t offset) {
  CHECK(data_ != nullptr);
  CHECK(IsContiguous()) << "Can only create view for compact tensor";
  NDArray ret = Internal::Create(shape, dtype, data_->dl_tensor.ctx);
  ret.data_->dl_tensor.byte_offset =
      this->data_->dl_tensor.byte_offset;
  size_t curr_size = GetDataSize(this->data_->dl_tensor);
  size_t view_size = GetDataSize(ret.data_->dl_tensor);
  CHECK_LE(view_size, curr_size)
      << "Tries to create a view that has bigger memory than current one";
  // increase ref count
  this->data_->IncRef();
  ret.data_->manager_ctx = this->data_;
  ret.data_->dl_tensor.data =
    static_cast<char*>(this->data_->dl_tensor.data) + offset;
  return ret;
}

DLManagedTensor* NDArray::ToDLPack() const {
  return Internal::ToDLPack(data_);
}

NDArray NDArray::EmptyShared(const std::string &name,
                       std::vector<int64_t> shape,
                       DLDataType dtype,
                       DLContext ctx, bool is_create) {
  NDArray ret = Internal::Create(shape, dtype, ctx);
  // setup memory content
  size_t size = GetDataSize(ret.data_->dl_tensor);
#ifndef _WIN32
  auto mem = std::make_shared<SharedMemory>(name);
  if (is_create) {
    ret.data_->dl_tensor.data = mem->create_new(size);
  } else {
    ret.data_->dl_tensor.data = mem->open(size);
  }

  ret.data_->mem = mem;
#else
  LOG(FATAL) << "Windows doesn't support NDArray with shared memory";
#endif  // _WIN32
  return ret;
}

NDArray NDArray::Empty(std::vector<int64_t> shape,
                       DLDataType dtype,
                       DLContext ctx) {
  NDArray ret = Internal::Create(shape, dtype, ctx);
  // setup memory content
  size_t size = GetDataSize(ret.data_->dl_tensor);
  size_t alignment = GetDataAlignment(ret.data_->dl_tensor);
  ret.data_->dl_tensor.data =
      DeviceAPI::Get(ret->ctx)->AllocDataSpace(
          ret->ctx, size, alignment, ret->dtype);
  return ret;
}

NDArray NDArray::FromDLPack(DLManagedTensor* tensor) {
  NDArray::Container* data = new NDArray::Container();
  data->deleter = Internal::DLPackDeleter;
  data->manager_ctx = tensor;
  data->dl_tensor = tensor->dl_tensor;
  return NDArray(data);
}

void NDArray::CopyFromTo(DLTensor* from,
                         DLTensor* to,
                         DGLStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  CHECK_EQ(from_size, to_size)
    << "DGLArrayCopyFromTo: The size must exactly match";

  CHECK(from->ctx.device_type == to->ctx.device_type
        || from->ctx.device_type == kDLCPU
        || to->ctx.device_type == kDLCPU)
    << "Can not copy across different ctx types directly";

  // Use the context that is *not* a cpu context to get the correct device
  // api manager.
  DGLContext ctx = from->ctx.device_type != kDLCPU ? from->ctx : to->ctx;

  DeviceAPI::Get(ctx)->CopyDataFromTo(
    from->data, static_cast<size_t>(from->byte_offset),
    to->data, static_cast<size_t>(to->byte_offset),
    from_size, from->ctx, to->ctx, from->dtype, stream);
}

}  // namespace runtime
}  // namespace dgl

using namespace dgl::runtime;

void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
  static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
  delete tensor;
}

int DGLArrayAlloc(const dgl_index_t* shape,
                  int ndim,
                  int dtype_code,
                  int dtype_bits,
                  int dtype_lanes,
                  int device_type,
                  int device_id,
                  DGLArrayHandle* out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = NDArray::Internal::MoveAsDLTensor(
      NDArray::Empty(std::vector<int64_t>(shape, shape + ndim), dtype, ctx));
  API_END();
}

int DGLArrayAllocSharedMem(const char *mem_name,
                           const dgl_index_t *shape,
                           int ndim,
                           int dtype_code,
                           int dtype_bits,
                           int dtype_lanes,
                           bool is_create,
                           DGLArrayHandle* out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  std::vector<int64_t> shape_vec(shape, shape + ndim);
  NDArray arr = NDArray::EmptyShared(mem_name, shape_vec, dtype,
                                     DLContext{kDLCPU, 0}, is_create);
  *out = NDArray::Internal::MoveAsDLTensor(arr);
  API_END();
}

int DGLArrayFree(DGLArrayHandle handle) {
  API_BEGIN();
  reinterpret_cast<NDArray::Container*>(handle)->DecRef();
  API_END();
}

int DGLArrayCopyFromTo(DGLArrayHandle from,
                       DGLArrayHandle to,
                       DGLStreamHandle stream) {
  API_BEGIN();
  NDArray::CopyFromTo(from, to, stream);
  API_END();
}

int DGLArrayFromDLPack(DLManagedTensor* from,
                       DGLArrayHandle* out) {
  API_BEGIN();
  *out = NDArray::Internal::MoveAsDLTensor(NDArray::FromDLPack(from));
  API_END();
}

int DGLArrayToDLPack(DGLArrayHandle from,
                     DLManagedTensor** out) {
  API_BEGIN();
  *out = NDArray::Internal::ToDLPack(reinterpret_cast<NDArray::Container*>(from));
  API_END();
}

void DGLDLManagedTensorCallDeleter(DLManagedTensor* dltensor) {
  (*(dltensor->deleter))(dltensor);
}

int DGLArrayCopyFromBytes(DGLArrayHandle handle,
                          void* data,
                          size_t nbytes) {
  API_BEGIN();
  DGLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes)
      << "DGLArrayCopyFromBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      data, 0,
      handle->data, static_cast<size_t>(handle->byte_offset),
      nbytes, cpu_ctx, handle->ctx, handle->dtype, nullptr);
  API_END();
}

int DGLArrayCopyToBytes(DGLArrayHandle handle,
                        void* data,
                        size_t nbytes) {
  API_BEGIN();
  DGLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes)
      << "DGLArrayCopyToBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      handle->data, static_cast<size_t>(handle->byte_offset),
      data, 0,
      nbytes, handle->ctx, cpu_ctx, handle->dtype, nullptr);
  API_END();
}
