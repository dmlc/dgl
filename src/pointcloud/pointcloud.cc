/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>

#include "../c_api_common.h"
#include "./pointcloud_op.h"
#include "cuda/pointcloud_fps.cuh"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

IdArray FPS_CPU(NDArray array, int64_t batch_size, int64_t sample_points) {
  IdArray ret;
  ATEN_FLOAT_TYPE_SWITCH(array->dtype, DType, "values", {
    ret = impl::_FPS_CPU<DType>(array, batch_size, sample_points, array->ctx);
  });
  return ret;
}

IdArray FPS_CUDA(NDArray array, int64_t batch_size, int64_t sample_points) {
  IdArray ret;
  ATEN_FLOAT_TYPE_SWITCH(array->dtype, DType, "values", {
    ret = cuda::_FPS_CUDA<DType>(array, batch_size, sample_points, array->ctx);
  });
  return ret;
}

DGL_REGISTER_GLOBAL("geometry._CAPI_FarthestPointSampler")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const NDArray data = args[0];
    const int64_t batch_size = args[1];
    const int64_t sample_points = args[2];
    const DLDeviceType XPU = data->ctx.device_type;
    IdArray result;
    if (XPU == kDLCPU) {
      result = FPS_CPU(data, batch_size, sample_points);
    } else if (XPU == kDLGPU) {
      result = FPS_CUDA(data, batch_size, sample_points);
    } else {
      LOG(FATAL) << "Incompatible array context. Currently only CPU/GPU are supported.";
    }
    *rv = result;
  });
}  // namespace aten
}  // namespace dgl
