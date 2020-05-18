/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/serailize/zerocopy_serializer.cc
 * \brief serializer implementation.
 */

#include <dgl/zerocopy_serializer.h>

#include "dgl/runtime/ndarray.h"

namespace dgl {

using dgl::runtime::NDArray;

struct RawDataTensorCtx {
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  DLManagedTensor tensor;
};

void RawDataTensoDLPackDeleter(DLManagedTensor* tensor) {
  auto ctx = static_cast<RawDataTensorCtx*>(tensor->manager_ctx);
  free(ctx->tensor.dl_tensor.data);
  delete ctx;
}

NDArray CreateNDArrayFromRawData(std::vector<int64_t> shape, DLDataType dtype,
                                 DLContext ctx, void* raw) {
  auto dlm_tensor_ctx = new RawDataTensorCtx();
  DLManagedTensor* dlm_tensor = &dlm_tensor_ctx->tensor;
  dlm_tensor_ctx->shape = shape;
  dlm_tensor->manager_ctx = dlm_tensor_ctx;
  dlm_tensor->dl_tensor.shape = dmlc::BeginPtr(dlm_tensor_ctx->shape);
  dlm_tensor->dl_tensor.ctx = ctx;
  dlm_tensor->dl_tensor.ndim = static_cast<int>(shape.size());
  dlm_tensor->dl_tensor.dtype = dtype;

  dlm_tensor_ctx->stride.resize(dlm_tensor->dl_tensor.ndim, 1);
  for (int i = dlm_tensor->dl_tensor.ndim - 2; i >= 0; --i) {
    dlm_tensor_ctx->stride[i] =
      dlm_tensor_ctx->shape[i + 1] * dlm_tensor_ctx->stride[i + 1];
  }
  dlm_tensor->dl_tensor.strides = dmlc::BeginPtr(dlm_tensor_ctx->stride);
  dlm_tensor->dl_tensor.data = raw;
  dlm_tensor->deleter = RawDataTensoDLPackDeleter;
  return NDArray::FromDLPack(dlm_tensor);
}

void StringStreamWithBuffer::PushNDArray(const NDArray& tensor) {
#ifndef _WIN32
  auto strm = static_cast<dmlc::Stream*>(this);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  int ndim = tensor->ndim;
  strm->WriteArray(tensor->shape, ndim);
  CHECK(tensor.IsContiguous())
    << "StringStreamWithBuffer only supports contiguous tensor";
  CHECK_EQ(tensor->byte_offset, 0)
    << "StringStreamWithBuffer only supports zero byte offset tensor";
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
    strm->Write<bool>(false);
    buffer_list_.emplace_back(tensor, tensor->data, data_byte_size);
  } else {
    CHECK(mem) << "Tried to send non-shared-memroy tensor to local "
                  "StringStreamWithBuffer";
    // Serialize only the shared memory name.
    strm->Write<bool>(true);
    strm->Write(mem->GetName());
  }
#else
  LOG(FATAL) << "StringStreamWithBuffer is not supported on windows";
  return nullptr;
#endif  // _WIN32
}

NDArray StringStreamWithBuffer::PopNDArray() {
#ifndef _WIN32
  auto strm = static_cast<dmlc::Stream*>(this);
  int ndim;
  DLDataType dtype;

  CHECK(strm->Read(&ndim)) << "Invalid DLTensor file format";
  CHECK(strm->Read(&dtype)) << "Invalid DLTensor file format";

  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    CHECK(strm->ReadArray(&shape[0], ndim)) << "Invalid DLTensor file format";
  }

  DLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;

  bool is_shared_mem;
  CHECK(strm->Read(&is_shared_mem)) << "Invalid stream read";
  std::string sharedmem_name;
  if (is_shared_mem) {
    CHECK(!send_to_remote_) << "Invalid attempt to deserialize from shared "
                               "memory with send_to_remote=true";
    CHECK(strm->Read(&sharedmem_name)) << "Invalid stream read";
    return NDArray::EmptyShared(sharedmem_name, shape, dtype, cpu_ctx, false);
  } else {
    CHECK(send_to_remote_) << "Invalid attempt to deserialize from raw data "
                              "pointer with send_to_remote=false";
    auto ret = CreateNDArrayFromRawData(shape, dtype, cpu_ctx,
                                        buffer_list_.front().data);
    buffer_list_.pop_front();
    return ret;
  }
#else
  LOG(FATAL) << "StringStreamWithBuffer is not supported on windows";
  return nullptr;
#endif  // _WIN32
}

}  // namespace dgl
