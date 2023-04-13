/**
 *  Copyright (c) 2017-2022 by Contributors
 * @file ndarray.cc
 * @brief NDArray container infratructure.
 */
#include <dgl/runtime/c_runtime_api.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/shared_mem.h>
#include <dgl/runtime/tensordispatch.h>
#include <dgl/zerocopy_serializer.h>
#include <dmlc/logging.h>
#include <string.h>

#include "runtime_base.h"

namespace dgl {

constexpr DGLDataType DGLDataTypeTraits<int8_t>::dtype;
constexpr DGLDataType DGLDataTypeTraits<uint8_t>::dtype;
constexpr DGLDataType DGLDataTypeTraits<int16_t>::dtype;
constexpr DGLDataType DGLDataTypeTraits<int32_t>::dtype;
constexpr DGLDataType DGLDataTypeTraits<int64_t>::dtype;
constexpr DGLDataType DGLDataTypeTraits<uint32_t>::dtype;
constexpr DGLDataType DGLDataTypeTraits<uint64_t>::dtype;
#ifdef DGL_USE_CUDA
constexpr DGLDataType DGLDataTypeTraits<__half>::dtype;
#if BF16_ENABLED
constexpr DGLDataType DGLDataTypeTraits<__nv_bfloat16>::dtype;
#endif  // BF16_ENABLED
#endif  // DGL_USE_CUDA
constexpr DGLDataType DGLDataTypeTraits<float>::dtype;
constexpr DGLDataType DGLDataTypeTraits<double>::dtype;

namespace runtime {

inline void VerifyDataType(DGLDataType dtype) {
  CHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDGLFloat) {
    CHECK_EQ(dtype.bits % 8, 0);
  } else {
    CHECK_EQ(dtype.bits % 8, 0);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataSize(const DGLArray& arr) {
  size_t size = 1;
  for (dgl_index_t i = 0; i < arr.ndim; ++i) {
    size *= arr.shape[i];
  }
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

inline size_t GetDataAlignment(const DGLArray& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

void NDArray::Internal::DefaultDeleter(NDArray::Container* ptr) {
  using dgl::runtime::NDArray;
  if (ptr->manager_ctx != nullptr) {
    static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
  } else if (ptr->mem) {
    ptr->mem = nullptr;
  } else if (ptr->dl_tensor.data != nullptr) {
    // if the array is still pinned before freeing, unpin it.
    if (ptr->pinned_by_dgl_) UnpinContainer(ptr);
    if (ptr->pinned_by_pytorch_) {
      DeviceAPI::Get(kDGLCUDA)->FreePinnedDataSpace(
          &(ptr->pytorch_raw_deleter_));
      CHECK(ptr->pytorch_raw_deleter_ == nullptr);
      ptr->pinned_by_pytorch_ = false;
      ptr->pytorch_ctx_ = nullptr;
    } else {
      dgl::runtime::DeviceAPI::Get(ptr->dl_tensor.ctx)
          ->FreeDataSpace(ptr->dl_tensor.ctx, ptr->dl_tensor.data);
    }
  }
  delete ptr;
}

NDArray NDArray::Internal::Create(
    std::vector<int64_t> shape, DGLDataType dtype, DGLContext ctx) {
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
    data->stride_[i] = data->shape_[i + 1] * data->stride_[i + 1];
  }
  data->dl_tensor.strides = dmlc::BeginPtr(data->stride_);
  // setup dtype
  data->dl_tensor.dtype = dtype;
  // setup ctx
  data->dl_tensor.ctx = ctx;
  return ret;
}

DGLArray* NDArray::Internal::MoveAsDGLArray(NDArray arr) {
  DGLArray* tensor = reinterpret_cast<DGLArray*>(arr.data_);
  CHECK(tensor == const_cast<DGLArray*>(arr.operator->()));
  arr.data_ = nullptr;
  return tensor;
}

size_t NDArray::GetSize() const { return GetDataSize(data_->dl_tensor); }

int64_t NDArray::NumElements() const {
  if (data_->dl_tensor.ndim == 0) return 0;
  int64_t size = 1;
  for (int i = 0; i < data_->dl_tensor.ndim; ++i) {
    size *= data_->dl_tensor.shape[i];
  }
  return size;
}

bool NDArray::IsContiguous() const {
  CHECK(data_ != nullptr);
  if (data_->dl_tensor.strides == nullptr) return true;

  // See https://github.com/dmlc/dgl/issues/2118 and PyTorch's
  // compute_contiguous() implementation
  int64_t z = 1;
  for (int64_t i = data_->dl_tensor.ndim - 1; i >= 0; --i) {
    if (data_->dl_tensor.shape[i] != 1) {
      if (data_->dl_tensor.strides[i] == z)
        z *= data_->dl_tensor.shape[i];
      else
        return false;
    }
  }
  return true;
}

NDArray NDArray::CreateView(
    std::vector<int64_t> shape, DGLDataType dtype, int64_t offset) {
  CHECK(data_ != nullptr);
  CHECK(IsContiguous()) << "Can only create view for compact tensor";
  NDArray ret = Internal::Create(shape, dtype, data_->dl_tensor.ctx);
  ret.data_->dl_tensor.byte_offset = this->data_->dl_tensor.byte_offset;
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

NDArray NDArray::EmptyShared(
    const std::string& name, std::vector<int64_t> shape, DGLDataType dtype,
    DGLContext ctx, bool is_create) {
  NDArray ret = Internal::Create(shape, dtype, ctx);
  size_t size = GetDataSize(ret.data_->dl_tensor);
  auto mem = std::make_shared<SharedMemory>(name);
  if (is_create) {
    ret.data_->dl_tensor.data = mem->CreateNew(size);
  } else {
    ret.data_->dl_tensor.data = mem->Open(size);
  }

  ret.data_->mem = mem;
  return ret;
}

NDArray NDArray::Empty(
    std::vector<int64_t> shape, DGLDataType dtype, DGLContext ctx) {
  NDArray ret = Internal::Create(shape, dtype, ctx);
  size_t size = GetDataSize(ret.data_->dl_tensor);
  size_t alignment = GetDataAlignment(ret.data_->dl_tensor);
  if (size > 0)
    ret.data_->dl_tensor.data = DeviceAPI::Get(ret->ctx)->AllocDataSpace(
        ret->ctx, size, alignment, ret->dtype);
  return ret;
}

void NDArray::CopyFromTo(DGLArray* from, DGLArray* to) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  CHECK_EQ(from_size, to_size)
      << "DGLArrayCopyFromTo: The size must exactly match";

  CHECK(
      from->ctx.device_type == to->ctx.device_type ||
      from->ctx.device_type == kDGLCPU || to->ctx.device_type == kDGLCPU)
      << "Can not copy across different ctx types directly";

  // Use the context that is *not* a cpu context to get the correct device
  // api manager.
  DGLContext ctx = from->ctx.device_type != kDGLCPU ? from->ctx : to->ctx;

  // default: local current cuda stream
  DeviceAPI::Get(ctx)->CopyDataFromTo(
      from->data, static_cast<size_t>(from->byte_offset), to->data,
      static_cast<size_t>(to->byte_offset), from_size, from->ctx, to->ctx,
      from->dtype);
}

void NDArray::RecordedCopyFromTo(
    DGLArray* from, DGLArray* to, void* pytorch_ctx) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  CHECK_EQ(from_size, to_size)
      << "DGLArrayCopyFromTo: The size must exactly match.";

  CHECK(from->ctx.device_type != to->ctx.device_type)
      << "Recoding event is only called for the copy between CPU and GPU.";

  CHECK(from->ctx.device_type == kDGLCUDA || to->ctx.device_type == kDGLCUDA)
      << "At least one CUDA ctx needs to be involved.";

  DeviceAPI::Get(kDGLCUDA)->RecordedCopyDataFromTo(
      from->data, static_cast<size_t>(from->byte_offset), to->data,
      static_cast<size_t>(to->byte_offset), from_size, from->ctx, to->ctx,
      from->dtype, pytorch_ctx);
}

NDArray NDArray::PinnedEmpty(
    std::vector<int64_t> shape, DGLDataType dtype, DGLContext ctx) {
  CHECK_EQ(ctx.device_type, kDGLCPU) << "Only NDArray on CPU can be pinned";
  NDArray ret = Internal::Create(shape, dtype, ctx);
  size_t size = GetDataSize(ret.data_->dl_tensor);
  if (size > 0) {
    ret.data_->dl_tensor.data = DeviceAPI::Get(kDGLCUDA)->AllocPinnedDataSpace(
        size, &(ret.data_->pytorch_ctx_), &(ret.data_->pytorch_raw_deleter_));
    CHECK(
        ret.data_->pytorch_ctx_ != nullptr &&
        ret.data_->pytorch_raw_deleter_ != nullptr)
        << "The allocation failed in PyTorch's CachingHostAllocator. "
        << "The returned context pointer is " << ret.data_->pytorch_ctx_
        << " and the function deleter is " << ret.data_->pytorch_raw_deleter_;
    ret.data_->pinned_by_pytorch_ = true;
  }
  return ret;
}

void NDArray::PinContainer(NDArray::Container* ptr) {
  if (IsContainerPinned(ptr)) return;
  auto* tensor = &(ptr->dl_tensor);
  CHECK_EQ(tensor->ctx.device_type, kDGLCPU)
      << "Only NDArray on CPU can be pinned";
  ptr->pinned_by_dgl_ =
      DeviceAPI::Get(kDGLCUDA)->PinData(tensor->data, GetDataSize(*tensor));
}

void NDArray::UnpinContainer(NDArray::Container* ptr) {
  auto container_is_pinned = IsContainerPinned(ptr);
  // The tensor may be pinned outside of DGL via a different CUDA API,
  // so we cannot unpin it with cudaHostUnregister.
  CHECK(ptr->pinned_by_dgl_ || !container_is_pinned)
      << "Cannot unpin a tensor that is pinned outside of DGL.";
  // 1. not pinned, do nothing
  if (!container_is_pinned) return;
  // 2. pinned by DGL, unpin it
  DeviceAPI::Get(kDGLCUDA)->UnpinData(ptr->dl_tensor.data);
  ptr->pinned_by_dgl_ = false;
}

void NDArray::RecordStream(DGLArray* tensor, DGLStreamHandle stream) {
  TensorDispatcher* tensor_dispatcher = TensorDispatcher::Global();
  CHECK(tensor_dispatcher->IsAvailable())
      << "RecordStream only works when TensorAdapter is available.";
  CHECK_EQ(tensor->ctx.device_type, kDGLCUDA)
      << "RecordStream only works with GPU tensors.";

  tensor_dispatcher->RecordStream(tensor->data, stream, tensor->ctx.device_id);
}

template <typename T>
NDArray NDArray::FromVector(const std::vector<T>& vec, DGLContext ctx) {
  const DGLDataType dtype = DGLDataTypeTraits<T>::dtype;
  int64_t size = static_cast<int64_t>(vec.size());
  NDArray ret = NDArray::Empty({size}, dtype, ctx);
  DeviceAPI::Get(ctx)->CopyDataFromTo(
      vec.data(), 0, static_cast<T*>(ret->data), 0, size * sizeof(T),
      DGLContext{kDGLCPU, 0}, ctx, dtype);
  return ret;
}

NDArray NDArray::CreateFromRaw(
    const std::vector<int64_t>& shape, DGLDataType dtype, DGLContext ctx,
    void* raw, bool auto_free) {
  NDArray ret = Internal::Create(shape, dtype, ctx);
  ret.data_->dl_tensor.data = raw;
  if (!auto_free) ret.data_->deleter = nullptr;
  return ret;
}

// export specializations
template NDArray NDArray::FromVector<int32_t>(
    const std::vector<int32_t>&, DGLContext);
template NDArray NDArray::FromVector<int64_t>(
    const std::vector<int64_t>&, DGLContext);
template NDArray NDArray::FromVector<uint32_t>(
    const std::vector<uint32_t>&, DGLContext);
template NDArray NDArray::FromVector<uint64_t>(
    const std::vector<uint64_t>&, DGLContext);
template NDArray NDArray::FromVector<float>(
    const std::vector<float>&, DGLContext);
template NDArray NDArray::FromVector<double>(
    const std::vector<double>&, DGLContext);

template <typename T>
std::vector<T> NDArray::ToVector() const {
  const DGLDataType dtype = DGLDataTypeTraits<T>::dtype;
  CHECK(data_->dl_tensor.ndim == 1)
      << "ToVector() only supported for 1D arrays";
  CHECK(data_->dl_tensor.dtype == dtype) << "dtype mismatch";

  int64_t size = data_->dl_tensor.shape[0];
  std::vector<T> vec(size);
  const DGLContext& ctx = data_->dl_tensor.ctx;
  DeviceAPI::Get(ctx)->CopyDataFromTo(
      static_cast<T*>(data_->dl_tensor.data), 0, vec.data(), 0,
      size * sizeof(T), ctx, DGLContext{kDGLCPU, 0}, dtype);
  return vec;
}

template std::vector<int32_t> NDArray::ToVector<int32_t>() const;
template std::vector<int64_t> NDArray::ToVector<int64_t>() const;
template std::vector<uint32_t> NDArray::ToVector<uint32_t>() const;
template std::vector<uint64_t> NDArray::ToVector<uint64_t>() const;
template std::vector<float> NDArray::ToVector<float>() const;
template std::vector<double> NDArray::ToVector<double>() const;

std::shared_ptr<SharedMemory> NDArray::GetSharedMem() const {
  return this->data_->mem;
}

bool NDArray::IsContainerPinned(NDArray::Container* ptr) {
  if (ptr->pinned_by_dgl_ || ptr->pinned_by_pytorch_) return true;
  auto* tensor = &(ptr->dl_tensor);
  // Can only be pinned if on CPU...
  if (tensor->ctx.device_type != kDGLCPU) return false;
  // ... and CUDA device API is enabled, and the tensor is indeed in pinned
  // memory.
  auto device = DeviceAPI::Get(kDGLCUDA, true);
  return device && device->IsPinned(tensor->data);
}

void NDArray::Save(dmlc::Stream* strm) const {
  auto zc_strm = dynamic_cast<StreamWithBuffer*>(strm);
  if (zc_strm) {
    zc_strm->PushNDArray(*this);
    return;
  }
  SaveDGLArray(strm, const_cast<DGLArray*>(operator->()));
}

bool NDArray::Load(dmlc::Stream* strm) {
  auto zc_strm = dynamic_cast<StreamWithBuffer*>(strm);
  if (zc_strm) {
    *this = zc_strm->PopNDArray();
    return true;
  }
  uint64_t header, reserved;
  CHECK(strm->Read(&header)) << "Invalid DGLArray file format";
  CHECK(strm->Read(&reserved)) << "Invalid DGLArray file format";
  CHECK(header == kDGLNDArrayMagic) << "Invalid DGLArray file format";
  DGLContext ctx;
  int ndim;
  DGLDataType dtype;
  CHECK(strm->Read(&ctx)) << "Invalid DGLArray file format";
  CHECK(strm->Read(&ndim)) << "Invalid DGLArray file format";
  CHECK(strm->Read(&dtype)) << "Invalid DGLArray file format";
  CHECK_EQ(ctx.device_type, kDGLCPU)
      << "Invalid DGLArray context: can only save as CPU tensor";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    CHECK(strm->ReadArray(&shape[0], ndim)) << "Invalid DGLArray file format";
  }
  NDArray ret = NDArray::Empty(shape, dtype, ctx);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  CHECK(strm->Read(&data_byte_size)) << "Invalid DGLArray file format";
  CHECK(data_byte_size == num_elems * elem_bytes)
      << "Invalid DGLArray file format";
  if (data_byte_size != 0) {
    // strm->Read will return the total number of elements successfully read.
    // Therefore if data_byte_size is zero, the CHECK below would fail.
    CHECK(strm->Read(ret->data, data_byte_size))
        << "Invalid DGLArray file format";
  }
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(ret->data, elem_bytes, num_elems);
  }
  *this = ret;
  return true;
}

}  // namespace runtime
}  // namespace dgl

using namespace dgl::runtime;

int DGLArrayAlloc(
    const dgl_index_t* shape, int ndim, int dtype_code, int dtype_bits,
    int dtype_lanes, int device_type, int device_id, DGLArrayHandle* out) {
  API_BEGIN();
  DGLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  DGLContext ctx;
  ctx.device_type = static_cast<DGLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = NDArray::Internal::MoveAsDGLArray(
      NDArray::Empty(std::vector<int64_t>(shape, shape + ndim), dtype, ctx));
  API_END();
}

int DGLArrayAllocSharedMem(
    const char* mem_name, const dgl_index_t* shape, int ndim, int dtype_code,
    int dtype_bits, int dtype_lanes, bool is_create, DGLArrayHandle* out) {
  API_BEGIN();
  DGLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  std::vector<int64_t> shape_vec(shape, shape + ndim);
  NDArray arr = NDArray::EmptyShared(
      mem_name, shape_vec, dtype, DGLContext{kDGLCPU, 0}, is_create);
  *out = NDArray::Internal::MoveAsDGLArray(arr);
  API_END();
}

int DGLArrayFree(DGLArrayHandle handle) {
  API_BEGIN();
  reinterpret_cast<NDArray::Container*>(handle)->DecRef();
  API_END();
}

int DGLArrayCopyFromTo(DGLArrayHandle from, DGLArrayHandle to) {
  API_BEGIN();
  NDArray::CopyFromTo(from, to);
  API_END();
}

int DGLArrayCopyFromBytes(DGLArrayHandle handle, void* data, size_t nbytes) {
  API_BEGIN();
  DGLContext cpu_ctx;
  cpu_ctx.device_type = kDGLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes) << "DGLArrayCopyFromBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)
      ->CopyDataFromTo(
          data, 0, handle->data, static_cast<size_t>(handle->byte_offset),
          nbytes, cpu_ctx, handle->ctx, handle->dtype);
  API_END();
}

int DGLArrayCopyToBytes(DGLArrayHandle handle, void* data, size_t nbytes) {
  API_BEGIN();
  DGLContext cpu_ctx;
  cpu_ctx.device_type = kDGLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes) << "DGLArrayCopyToBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)
      ->CopyDataFromTo(
          handle->data, static_cast<size_t>(handle->byte_offset), data, 0,
          nbytes, handle->ctx, cpu_ctx, handle->dtype);
  API_END();
}

int DGLArrayPinData(DGLArrayHandle handle, DGLContext ctx) {
  API_BEGIN();
  auto* nd_container = reinterpret_cast<NDArray::Container*>(handle);
  NDArray::PinContainer(nd_container);
  API_END();
}

int DGLArrayUnpinData(DGLArrayHandle handle, DGLContext ctx) {
  API_BEGIN();
  auto* nd_container = reinterpret_cast<NDArray::Container*>(handle);
  NDArray::UnpinContainer(nd_container);
  API_END();
}

int DGLArrayRecordStream(DGLArrayHandle handle, DGLStreamHandle stream) {
  API_BEGIN();
  NDArray::RecordStream(handle, stream);
  API_END();
}
