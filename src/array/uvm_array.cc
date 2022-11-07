/**
 *  Copyright (c) 2019-2022 by Contributors
 * @file array/uvm_array.cc
 * @brief DGL array utilities implementation
 */
#include <dgl/array.h>

#include <sstream>

#include "../c_api_common.h"
#include "./uvm_array_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

NDArray IndexSelectCPUFromGPU(NDArray array, IdArray index) {
#ifdef DGL_USE_CUDA
  CHECK(array.IsPinned()) << "Input array must be in pinned memory.";
  CHECK_EQ(index->ctx.device_type, kDGLCUDA) << "Index must be on the GPU.";
  CHECK_GE(array->ndim, 1) << "Input array must have at least 1 dimension.";
  CHECK_EQ(index->ndim, 1) << "Index must be a 1D array.";

  ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      return impl::IndexSelectCPUFromGPU<DType, IdType>(array, index);
    });
  });
#endif
  LOG(FATAL) << "IndexSelectCPUFromGPU requires CUDA.";
  // Should be unreachable
  return NDArray{};
}

void IndexScatterGPUToCPU(NDArray dest, IdArray index, NDArray source) {
#ifdef DGL_USE_CUDA
  CHECK(dest.IsPinned()) << "Destination array must be in pinned memory.";
  CHECK_EQ(index->ctx.device_type, kDGLCUDA) << "Index must be on the GPU.";
  CHECK_EQ(source->ctx.device_type, kDGLCUDA)
      << "Source array must be on the GPU.";
  CHECK_EQ(dest->dtype, source->dtype) << "Destination array and source "
                                          "array must have the same dtype.";
  CHECK_GE(dest->ndim, 1)
      << "Destination array must have at least 1 dimension.";
  CHECK_EQ(index->ndim, 1) << "Index must be a 1D array.";

  ATEN_DTYPE_BITS_ONLY_SWITCH(source->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      impl::IndexScatterGPUToCPU<DType, IdType>(dest, index, source);
    });
  });
#else
  LOG(FATAL) << "IndexScatterGPUToCPU requires CUDA.";
#endif
}

DGL_REGISTER_GLOBAL("ndarray.uvm._CAPI_DGLIndexSelectCPUFromGPU")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray array = args[0];
      IdArray index = args[1];
      *rv = IndexSelectCPUFromGPU(array, index);
    });

DGL_REGISTER_GLOBAL("ndarray.uvm._CAPI_DGLIndexScatterGPUToCPU")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray dest = args[0];
      IdArray index = args[1];
      NDArray source = args[2];
      IndexScatterGPUToCPU(dest, index, source);
    });

}  // namespace aten
}  // namespace dgl
