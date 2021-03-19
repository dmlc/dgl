/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/knn.cc
 * \brief k-nearest-neighbor (KNN) implementation
 */

#include "kdtree_ndarray_adaptor.h"
#include "knn.h"
#include <vector>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include "../../array/check.h"

using namespace dgl::runtime;
using namespace dgl::transform::knn_utils;
namespace dgl {
namespace transform {
namespace impl {

/*! \brief The kd-tree implementation of K-Nearest Neighbors */
template <typename FloatType, typename IdType>
void KdTreeKNN(const NDArray data_points, const IdArray data_offsets,
               const NDArray query_points, const IdArray query_offsets,
               const int k, IdArray result) {
  int64_t batch_size = data_offsets->shape[0] - 1;
  int64_t feature_size = data_points->shape[1];
  IdType * data_offsets_data = static_cast<IdType*>(data_offsets->data);
  IdType * query_offsets_data = static_cast<IdType*>(query_offsets->data);
  IdType * result_data = static_cast<IdType*>(result->data);
  FloatType * query_points_data = static_cast<FloatType*>(query_points->data);

  for (int64_t b = 0; b < batch_size; ++b) {
    auto d_offset = data_offsets_data[b];
    auto d_length = data_offsets_data[b + 1] - d_offset;
    auto q_offset = query_offsets_data[b];
    auto q_length = query_offsets_data[b + 1] - q_offset;
    auto out_offset = 2 * k * q_offset;

    // create view for each segment
    NDArray current_data_points = const_cast<NDArray*>(&data_points)->CreateView(
      {d_length, feature_size}, data_points->dtype, d_offset * feature_size * sizeof(FloatType));
    FloatType * current_query_pts_data = query_points_data + q_offset * feature_size;

    KDTreeNDArrayAdaptor<FloatType, IdType> kdtree(feature_size, current_data_points);

    // query
#pragma omp parallel for
    for (int64_t q = 0; q < q_length; ++q) {
      std::vector<IdType> out_buffer(k);
      std::vector<FloatType> out_dist_buffer(k);
      auto curr_out_offset = 2 * k * q + out_offset;
      FloatType * q_point = current_query_pts_data + q * feature_size;
      size_t num_matches = kdtree.GetIndex()->knnSearch(
        q_point, k, out_buffer.data(), out_dist_buffer.data());

      for (size_t i = 0; i < num_matches; ++i) {
        result_data[curr_out_offset] = q + q_offset; curr_out_offset++;
        result_data[curr_out_offset] = out_buffer[i] + d_offset; curr_out_offset++;
      }
    }
  }
}
}  // namespace impl

template <DLDeviceType XPU, typename FloatType, typename IdType>
void KNN(const NDArray data_points, const IdArray data_offsets,
         const NDArray query_points, const IdArray query_offsets,
         const int k, IdArray result) {
  impl::KdTreeKNN<FloatType, IdType>(
    data_points, data_offsets, query_points, query_offsets, k, result);
}

template void KNN<kDLCPU, float, int32_t>(
  const NDArray data_points, const IdArray data_offsets,
  const NDArray query_points, const IdArray query_offsets,
  const int k, IdArray result);
template void KNN<kDLCPU, float, int64_t>(
  const NDArray data_points, const IdArray data_offsets,
  const NDArray query_points, const IdArray query_offsets,
  const int k, IdArray result);
template void KNN<kDLCPU, double, int32_t>(
  const NDArray data_points, const IdArray data_offsets,
  const NDArray query_points, const IdArray query_offsets,
  const int k, IdArray result);
template void KNN<kDLCPU, double, int64_t>(
  const NDArray data_points, const IdArray data_offsets,
  const NDArray query_points, const IdArray query_offsets,
  const int k, IdArray result);

DGL_REGISTER_GLOBAL("transform._CAPI_DGLKNN")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    const NDArray data_points = args[0];
    const IdArray data_offsets = args[1];
    const NDArray query_points = args[2];
    const IdArray query_offsets = args[3];
    const int k = args[4];
    IdArray result = args[5];

    aten::CheckContiguous(
      {data_points, data_offsets, query_points, query_offsets, result},
      {"data_points", "data_offsets", "query_points", "query_offsets", "result"});
    aten::CheckCtx(
      data_points->ctx, {data_offsets, query_points, query_offsets, result},
      {"data_offsets", "query_points", "query_offsets", "result"});

    ATEN_XPU_SWITCH_CUDA(data_points->ctx.device_type, XPU, "KNN", {
      ATEN_FLOAT_TYPE_SWITCH(data_points->dtype, FloatType, "data_points", {
        ATEN_ID_TYPE_SWITCH(result->dtype, IdType, {
          KNN<XPU, FloatType, IdType>(
            data_points, data_offsets, query_points, query_offsets, k, result);
        });
      });
    });
  });

} // namespace transform
} // namespace dgl
