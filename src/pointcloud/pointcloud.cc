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

using namespace dgl::runtime;

namespace dgl {
namespace aten {

IdArray FPS(NDArray array, int64_t batch_size, int64_t sample_points) {
  IdArray ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ret = impl::FPS<XPU, DType, int64_t>(array, batch_size, sample_points, array->ctx);
    });
  });
  return ret;
}

DGL_REGISTER_GLOBAL("pointcloud._CAPI_FarthestPointSampler")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const NDArray data = args[0];
    const int64_t batch_size = args[1];
    const int64_t sample_points = args[2];
    const auto result = FPS(data, batch_size, sample_points);
    *rv = result;
  });

}  // namespace aten
}  // namespace dgl
