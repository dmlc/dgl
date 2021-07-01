/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/uvm_array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>
#include <sstream>
#include "../c_api_common.h"
#include "./uvm_array_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

NDArray IndexSelectCPUFromGPU(NDArray array, IdArray index) {
  NDArray ret;

  CHECK_EQ(array->ctx.device_type, kDLCPU) << "Only the CPU device type input "
                                           << "array supported";
  CHECK_EQ(index->ctx.device_type, kDLGPU) << "Only the GPU device type input "
                                           << "index supported";

  CHECK_GE(array->ndim, 1) << "Only support array with at least 1 dimension";
  CHECK_EQ(index->ndim, 1) << "Index array must be an 1D array.";
  ATEN_DTYPE_BITS_ONLY_SWITCH(array->dtype, DType, "values", {
    ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
      ret = impl::IndexSelectCPUFromGPU<DType, IdType>(array, index);
    });
  });
  return ret;
}

DGL_REGISTER_GLOBAL("ndarray.uvm._CAPI_DGLIndexSelectCPUFromGPU")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray array = args[0];
    IdArray index = args[1];
    *rv = IndexSelectCPUFromGPU(array, index);
  });

}  // namespace aten
}  // namespace dgl