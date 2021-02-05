/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/geometry_op_impl.cc
 * \brief Geometry operator CPU implementation
 */
#include <dgl/array.h>
#include <numeric>
#include <vector>

namespace dgl {
using runtime::NDArray;
namespace geometry {
namespace impl {

/*!
 * \brief Farthest Point Sampler without the need to compute all pairs of distance.
 * 
 * The input array has shape (N, d), where N is the number of points, and d is the dimension.
 * It consists of a (flatten) batch of point clouds.
 *
 * In each batch, the algorithm starts with the sample index specified by ``start_idx``.
 * Then for each point, we maintain the minimum to-sample distance.
 * Finally, we pick the point with the maximum such distance.
 * This process will be repeated for ``sample_points`` - 1 times.
 */
template <DLDeviceType XPU, typename FloatType, typename IdType>
void FarthestPointSampler(NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result) {
  const FloatType* array_data = static_cast<FloatType*>(array->data);
  const int64_t point_in_batch = array->shape[0] / batch_size;
  const int64_t dim = array->shape[1];

  // distance
  FloatType* dist_data = static_cast<FloatType*>(dist->data);

  // sample for each cloud in the batch
  IdType* start_idx_data = static_cast<IdType*>(start_idx->data);

  // return value
  IdType* ret_data = static_cast<IdType*>(result->data);

  int64_t array_start = 0, ret_start = 0;
  // loop for each point cloud sample in this batch
  for (auto b = 0; b < batch_size; b++) {
    // random init start sample
    int64_t sample_idx = (int64_t)start_idx_data[b];
    ret_data[ret_start] = (IdType)(sample_idx);

    // sample the rest `sample_points - 1` points
    for (auto i = 0; i < sample_points - 1; i++) {
      // re-init distance and the argmax
      int64_t dist_argmax = 0;
      FloatType dist_max = -1;

      // update the distance
      for (auto j = 0; j < point_in_batch; j++) {
        // compute the distance on dimensions
        FloatType one_dist = 0;
        for (auto d = 0; d < dim; d++) {
          FloatType tmp = array_data[(array_start + j) * dim + d] -
              array_data[(array_start + sample_idx) * dim + d];
          one_dist += tmp * tmp;
        }

        // for each out-of-set point, keep its nearest to-the-set distance
        if (i == 0 || dist_data[j] > one_dist) {
          dist_data[j] = one_dist;
        }
        // look for the farthest sample
        if (dist_data[j] > dist_max) {
          dist_argmax = j;
          dist_max = dist_data[j];
        }
      }
      // sample the `dist_argmax`-th point
      sample_idx = dist_argmax;
      ret_data[ret_start + i + 1] = (IdType)(sample_idx);
    }

    array_start += point_in_batch;
    ret_start += sample_points;
  }
}

template <DLDeviceType XPU, typename FloatType, typename IdType>
void GraphMatching(const NDArray indptr, const NDArray indices,
                   const NDArray weight, const NDArray vis_order,
                   NDArray result) {
  const int64_t num_nodes = indptr->shape[0] - 1;
  const IdType* indptr_data = static_cast<IdType*>(indptr->data);
  const IdType* indices_data = static_cast<IdType*>(indices->data);
  const IdType* vis_order_data = static_cast<IdType*>(vis_order->data);
  IdType* result_data = static_cast<IdType*>(result->data);

  if (aten::IsNullArray(weight)) {
    for (int64_t n = 0; n < num_nodes; ++n) {
      auto u = vis_order_data[n];

      // if marked
      if (result_data[u] >= 0) continue;

      result_data[u] = u;
      IdType row_start = indptr_data[u], row_end = indptr_data[u + 1];

      // find one unmarked neighbor
      for (auto e = row_start; e < row_end; ++e) {
        auto v = indices_data[e];

        if (result_data[v] >= 0) continue;

        result_data[u] = std::min(u, v);
        result_data[v] = result_data[u];
        break;
      }
    }
  } else {
    FloatType* weight_data = static_cast<FloatType*>(weight->data);

    for (int64_t n = 0; n < num_nodes; ++n) {
      auto u = vis_order_data[n];
      if (result_data[u] >= 0) continue;
      auto v_max = u;
      FloatType weight_max = static_cast<FloatType>(0.);

      IdType row_start = indptr_data[u], row_end = indptr_data[u + 1];
      for (auto e = row_start; e < row_end; ++e) {
        auto v = indices_data[e];

        if (result_data[v] >= 0) continue;
        if (weight_data[e] >= weight_max) {
          v_max = v;
          weight_max = weight_data[e];
        }
      }

      result_data[u] = std::min(u, v_max);
      result_data[v_max] = result_data[u];
    }
  }
}


template void FarthestPointSampler<kDLCPU, float, int32_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDLCPU, float, int64_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDLCPU, double, int32_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDLCPU, double, int64_t>(
    NDArray array, int64_t batch_size, int64_t sample_points,
    NDArray dist, IdArray start_idx, IdArray result);

template void GraphMatching<kDLCPU, float, int32_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, const NDArray vis_order,
    NDArray result);
template void GraphMatching<kDLCPU, float, int64_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, const NDArray vis_order,
    NDArray result);
template void GraphMatching<kDLCPU, double, int32_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, const NDArray vis_order,
    NDArray result);
template void GraphMatching<kDLCPU, double, int64_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, const NDArray vis_order,
    NDArray result);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl
