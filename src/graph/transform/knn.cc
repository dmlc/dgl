/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/transform/knn.cc
 * @brief k-nearest-neighbor (KNN) interface
 */

#include "knn.h"

#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

#include "../../array/check.h"

using namespace dgl::runtime;
namespace dgl {
namespace transform {

DGL_REGISTER_GLOBAL("transform._CAPI_DGLKNN")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const NDArray data_points = args[0];
      const IdArray data_offsets = args[1];
      const NDArray query_points = args[2];
      const IdArray query_offsets = args[3];
      const int k = args[4];
      IdArray result = args[5];
      const std::string algorithm = args[6];

      aten::CheckContiguous(
          {data_points, data_offsets, query_points, query_offsets, result},
          {"data_points", "data_offsets", "query_points", "query_offsets",
           "result"});
      aten::CheckCtx(
          data_points->ctx, {data_offsets, query_points, query_offsets, result},
          {"data_offsets", "query_points", "query_offsets", "result"});

      ATEN_XPU_SWITCH_CUDA(data_points->ctx.device_type, XPU, "KNN", {
        ATEN_FLOAT_TYPE_SWITCH(data_points->dtype, FloatType, "data_points", {
          ATEN_ID_TYPE_SWITCH(result->dtype, IdType, {
            KNN<XPU, FloatType, IdType>(
                data_points, data_offsets, query_points, query_offsets, k,
                result, algorithm);
          });
        });
      });
    });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLNNDescent")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const NDArray points = args[0];
      const IdArray offsets = args[1];
      const IdArray result = args[2];
      const int k = args[3];
      const int num_iters = args[4];
      const int num_candidates = args[5];
      const double delta = args[6];

      aten::CheckContiguous(
          {points, offsets, result}, {"points", "offsets", "result"});
      aten::CheckCtx(
          points->ctx, {points, offsets, result},
          {"points", "offsets", "result"});

      ATEN_XPU_SWITCH_CUDA(points->ctx.device_type, XPU, "NNDescent", {
        ATEN_FLOAT_TYPE_SWITCH(points->dtype, FloatType, "points", {
          ATEN_ID_TYPE_SWITCH(result->dtype, IdType, {
            NNDescent<XPU, FloatType, IdType>(
                points, offsets, result, k, num_iters, num_candidates, delta);
          });
        });
      });
    });

}  // namespace transform
}  // namespace dgl
