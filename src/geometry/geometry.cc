/*!
 *  Copyright (c) 2019 by Contributors
 * \file geometry/geometry.cc
 * \brief DGL geometry utilities implementation
 */
#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>
#include "../c_api_common.h"
#include "./geometry_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace geometry {

void FarthestPointSampler(NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result) {

  CHECK_EQ(array->ctx, result->ctx) << "Array and the result should be on the same device.";
  CHECK_EQ(array->shape[0], dist->shape[0]) << "Shape of array and dist mismatch";
  CHECK_EQ(start_idx->shape[0], batch_size) << "Shape of start_idx and batch_size mismatch";
  CHECK_EQ(result->shape[0], batch_size * sample_points) << "Invalid shape of result";

  ATEN_FLOAT_TYPE_SWITCH(array->dtype, FloatType, "values", {
    ATEN_ID_TYPE_SWITCH(result->dtype, IdType, {
      ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "FarthestPointSampler", {
        impl::FarthestPointSampler<XPU, FloatType, IdType>(
            array, batch_size, sample_points, dist, start_idx, result);
      });
    });
  });
}

///////////////////////// C APIs /////////////////////////

DGL_REGISTER_GLOBAL("geometry._CAPI_FarthestPointSampler")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const NDArray data = args[0];
    const int64_t batch_size = args[1];
    const int64_t sample_points = args[2];
    NDArray dist = args[3];
    IdArray start_idx = args[4];
    IdArray result = args[5];

    FarthestPointSampler(data, batch_size, sample_points, dist, start_idx, result);
  });

}  // namespace geometry
}  // namespace dgl
