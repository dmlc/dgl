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

// Check whether the given arguments have the same context.
inline void CheckCtx(
    const DLContext& ctx,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (aten::IsNullArray(arrays[i]))
      continue;
    CHECK_EQ(ctx, arrays[i]->ctx)
      << "Expected device context " << ctx << ". But got "
      << arrays[i]->ctx << " for " << names[i] << ".";
  }
}

// Check whether input tensors are contiguous.
inline void CheckContiguous(
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (aten::IsNullArray(arrays[i]))
      continue;
    CHECK(arrays[i].IsContiguous())
      << "Expect " << names[i] << " to be a contiguous tensor";
  }
}

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

DGL_REGISTER_GLOBAL("geometry._CAPI_EdgeCoarsening")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const NDArray indptr = args[0];
    const NDArray indices = args[1];
    const NDArray weight = args[2];
    NDArray result = args[3];

    // check context and contiguous
    CheckCtx(indptr->ctx, {indices, weight, result},
      {"indices", "edge_weight", "result"});
    CheckContiguous({indptr, indices, weight, result},
      {"indptr", "indices", "edge_weight", "result"});

    // check shape
    CHECK_EQ(indptr->ndim, 1) << "indptr should be an 1D tensor.";
    CHECK_EQ(indices->ndim, 1) << "indices should be an 1D tensor.";
    CHECK_EQ(result->ndim, 1) << "result should be an 1D tensor.";
    CHECK_EQ(indptr->shape[0] - 1, result->shape[0])
      << "The number of nodes in CSR matrix should be the same as the result tensor.";

    if (!aten::IsNullArray(weight)) {
      CHECK_EQ(weight->ndim, 1) << "weight should be an 1D tensor.";
      CHECK_EQ(indices->shape[0], weight->shape[0])
        << "indices of CSR matrix should have the same shape "
        << "as the edge weight tensor.";
    }

    // call implementation
    if (!aten::IsNullArray(weight)) {
      ATEN_FLOAT_TYPE_SWITCH(weight->dtype, FloatType, "weight", {
        ATEN_ID_TYPE_SWITCH(indptr->dtype, IdType, {
          ATEN_XPU_SWITCH_CUDA(indptr->ctx.device_type, XPU, "EdgeCoarsening", {
            impl::EdgeCoarsening<XPU, FloatType, IdType>(
                indptr, indices, weight, result);
          });
        });
      });
    } else {
      ATEN_ID_TYPE_SWITCH(indptr->dtype, IdType, {
        ATEN_XPU_SWITCH_CUDA(indptr->ctx.device_type, XPU, "EdgeCoarsening", {
          impl::EdgeCoarsening<XPU, float, IdType>(
              indptr, indices, weight, result);
          });
        });
    }

  });

}  // namespace geometry
}  // namespace dgl
