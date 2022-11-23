/**
 *  Copyright (c) 2020-2022 by Contributors
 * @file graph/serailize/zerocopy_serializer.cc
 * @brief serializer implementation.
 */

#include <dgl/zerocopy_serializer.h>

#include "dgl/runtime/ndarray.h"
#include "dmlc/memory_io.h"

namespace dgl {

using dgl::runtime::NDArray;

NDArray CreateNDArrayFromRawData(
    std::vector<int64_t> shape, DGLDataType dtype, DGLContext ctx, void* raw) {
  return NDArray::CreateFromRaw(shape, dtype, ctx, raw, true);
}

void StreamWithBuffer::PushNDArray(const NDArray& tensor) {
#ifndef _WIN32
  this->Write(tensor->ndim);
  this->Write(tensor->dtype);
  int ndim = tensor->ndim;
  this->WriteArray(tensor->shape, ndim);
  CHECK(tensor.IsContiguous())
      << "StreamWithBuffer only supports contiguous tensor";
  CHECK_EQ(tensor->byte_offset, 0)
      << "StreamWithBuffer only supports zero byte offset tensor";
  int type_bytes = tensor->dtype.bits / 8;
  int64_t num_elems = 1;
  for (int i = 0; i < ndim; ++i) {
    num_elems *= tensor->shape[i];
  }
  int64_t data_byte_size = type_bytes * num_elems;

  auto mem = tensor.GetSharedMem();
  if (send_to_remote_ || !mem) {
    // If the stream is for remote communication or the data is not stored in
    // shared memory, serialize the data content as a buffer.
    this->Write<bool>(false);
    // If this is a null ndarray, we will not push it into the underlying
    // buffer_list
    if (data_byte_size != 0) {
      buffer_list_.emplace_back(tensor, tensor->data, data_byte_size);
    }
  } else {
    CHECK(mem) << "Tried to send non-shared-memroy tensor to local "
                  "StreamWithBuffer";
    // Serialize only the shared memory name.
    this->Write<bool>(true);
    this->Write(mem->GetName());
  }
#else
  LOG(FATAL) << "StreamWithBuffer is not supported on windows";
#endif  // _WIN32
  return;
}

NDArray StreamWithBuffer::PopNDArray() {
#ifndef _WIN32
  int ndim;
  DGLDataType dtype;

  CHECK(this->Read(&ndim)) << "Invalid DGLArray file format";
  CHECK(this->Read(&dtype)) << "Invalid DGLArray file format";

  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    CHECK(this->ReadArray(&shape[0], ndim)) << "Invalid DGLArray file format";
  }

  DGLContext cpu_ctx;
  cpu_ctx.device_type = kDGLCPU;
  cpu_ctx.device_id = 0;

  bool is_shared_mem;
  CHECK(this->Read(&is_shared_mem)) << "Invalid stream read";
  std::string sharedmem_name;
  if (is_shared_mem) {
    CHECK(!send_to_remote_) << "Invalid attempt to deserialize from shared "
                               "memory with send_to_remote=true";
    CHECK(this->Read(&sharedmem_name)) << "Invalid stream read";
    return NDArray::EmptyShared(sharedmem_name, shape, dtype, cpu_ctx, false);
  } else {
    CHECK(send_to_remote_) << "Invalid attempt to deserialize from raw data "
                              "pointer with send_to_remote=false";
    NDArray ret;
    if (ndim == 0 || shape[0] == 0) {
      // Mean this is a null ndarray
      ret = CreateNDArrayFromRawData(shape, dtype, cpu_ctx, nullptr);
    } else {
      ret = CreateNDArrayFromRawData(
          shape, dtype, cpu_ctx, buffer_list_.front().data);
      buffer_list_.pop_front();
    }
    return ret;
  }
#else
  LOG(FATAL) << "StreamWithBuffer is not supported on windows";
  return NDArray();
#endif  // _WIN32
}

}  // namespace dgl
