/**
 *  Copyright (c) 2019 by Contributors
 * @file geometry/geometry.cc
 * @brief DGL geometry utilities implementation
 */
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/ndarray.h>

#include "../array/check.h"
#include "../c_api_common.h"
#include "./geometry_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace geometry {

void FarthestPointSampler(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    IdArray start_idx, IdArray result) {
  CHECK_EQ(array->ctx, result->ctx)
      << "Array and the result should be on the same device.";
  CHECK_EQ(array->shape[0], dist->shape[0])
      << "Shape of array and dist mismatch";
  CHECK_EQ(start_idx->shape[0], batch_size)
      << "Shape of start_idx and batch_size mismatch";
  CHECK_EQ(result->shape[0], batch_size * sample_points)
      << "Invalid shape of result";

  ATEN_FLOAT_TYPE_SWITCH(array->dtype, FloatType, "values", {
    ATEN_ID_TYPE_SWITCH(result->dtype, IdType, {
      ATEN_XPU_SWITCH_CUDA(
          array->ctx.device_type, XPU, "FarthestPointSampler", {
            impl::FarthestPointSampler<XPU, FloatType, IdType>(
                array, batch_size, sample_points, dist, start_idx, result);
          });
    });
  });
}

void NeighborMatching(
    HeteroGraphPtr graph, const NDArray weight, IdArray result) {
  if (!aten::IsNullArray(weight)) {
    ATEN_XPU_SWITCH_CUDA(
        graph->Context().device_type, XPU, "NeighborMatching", {
          ATEN_FLOAT_TYPE_SWITCH(weight->dtype, FloatType, "weight", {
            ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
              impl::WeightedNeighborMatching<XPU, FloatType, IdType>(
                  graph->GetCSRMatrix(0), weight, result);
            });
          });
        });
  } else {
    ATEN_XPU_SWITCH_CUDA(
        graph->Context().device_type, XPU, "NeighborMatching", {
          ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
            impl::NeighborMatching<XPU, IdType>(graph->GetCSRMatrix(0), result);
          });
        });
  }
}

///////////////////////// C APIs /////////////////////////

DGL_REGISTER_GLOBAL("geometry._CAPI_FarthestPointSampler")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const NDArray data = args[0];
      const int64_t batch_size = args[1];
      const int64_t sample_points = args[2];
      NDArray dist = args[3];
      IdArray start_idx = args[4];
      IdArray result = args[5];

      FarthestPointSampler(
          data, batch_size, sample_points, dist, start_idx, result);
    });

DGL_REGISTER_GLOBAL("geometry._CAPI_NeighborMatching")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef graph = args[0];
      const NDArray weight = args[1];
      IdArray result = args[2];

      // sanity check
      aten::CheckCtx(
          graph->Context(), {weight, result}, {"edge_weight, result"});
      aten::CheckContiguous({weight, result}, {"edge_weight", "result"});
      CHECK_EQ(graph->NumEdgeTypes(), 1)
          << "homogeneous graph has only one edge type";
      CHECK_EQ(result->ndim, 1) << "result should be an 1D tensor.";
      auto pair = graph->meta_graph()->FindEdge(0);
      const dgl_type_t node_type = pair.first;
      CHECK_EQ(graph->NumVertices(node_type), result->shape[0])
          << "The number of nodes should be the same as the length of result "
             "tensor.";
      if (!aten::IsNullArray(weight)) {
        CHECK_EQ(weight->ndim, 1) << "weight should be an 1D tensor.";
        CHECK_EQ(graph->NumEdges(0), weight->shape[0])
            << "number of edges in graph should be the same "
            << "as the length of edge weight tensor.";
      }

      // call implementation
      NeighborMatching(graph.sptr(), weight, result);
    });

}  // namespace geometry
}  // namespace dgl
