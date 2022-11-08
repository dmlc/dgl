/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/geometry_op_impl.cc
 * @brief Geometry operator CPU implementation
 */
#include <dgl/random.h>

#include <numeric>
#include <utility>
#include <vector>

#include "../geometry_op.h"

namespace dgl {
using runtime::NDArray;
namespace geometry {
namespace impl {

/** @brief Knuth shuffle algorithm */
template <typename IdType>
void IndexShuffle(IdType *idxs, int64_t num_elems) {
  for (int64_t i = num_elems - 1; i > 0; --i) {
    int64_t j = dgl::RandomEngine::ThreadLocal()->RandInt(i);
    std::swap(idxs[i], idxs[j]);
  }
}
template void IndexShuffle<int32_t>(int32_t *idxs, int64_t num_elems);
template void IndexShuffle<int64_t>(int64_t *idxs, int64_t num_elems);

/** @brief Groupwise index shuffle algorithm. This function will perform shuffle
 * in subarrays indicated by group index. The group index is similar to indptr
 * in CSRMatrix.
 *
 * @param group_idxs group index array.
 * @param idxs index array for shuffle.
 * @param num_groups_idxs length of group_idxs
 * @param num_elems length of idxs
 */
template <typename IdType>
void GroupIndexShuffle(
    const IdType *group_idxs, IdType *idxs, int64_t num_groups_idxs,
    int64_t num_elems) {
  if (num_groups_idxs < 2) return;  // empty idxs array
  CHECK_LE(group_idxs[num_groups_idxs - 1], num_elems)
      << "group_idxs out of range";
  for (int64_t i = 0; i < num_groups_idxs - 1; ++i) {
    auto subarray_len = group_idxs[i + 1] - group_idxs[i];
    IndexShuffle(idxs + group_idxs[i], subarray_len);
  }
}
template void GroupIndexShuffle<int32_t>(
    const int32_t *group_idxs, int32_t *idxs, int64_t num_groups_idxs,
    int64_t num_elems);
template void GroupIndexShuffle<int64_t>(
    const int64_t *group_idxs, int64_t *idxs, int64_t num_groups_idxs,
    int64_t num_elems);

template <typename IdType>
IdArray RandomPerm(int64_t num_nodes) {
  IdArray perm =
      aten::NewIdArray(num_nodes, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
  IdType *perm_data = static_cast<IdType *>(perm->data);
  std::iota(perm_data, perm_data + num_nodes, 0);
  IndexShuffle(perm_data, num_nodes);
  return perm;
}

template <typename IdType>
IdArray GroupRandomPerm(
    const IdType *group_idxs, int64_t num_group_idxs, int64_t num_nodes) {
  IdArray perm =
      aten::NewIdArray(num_nodes, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
  IdType *perm_data = static_cast<IdType *>(perm->data);
  std::iota(perm_data, perm_data + num_nodes, 0);
  GroupIndexShuffle(group_idxs, perm_data, num_group_idxs, num_nodes);
  return perm;
}

/**
 * @brief Farthest Point Sampler without the need to compute all pairs of
 * distance.
 *
 * The input array has shape (N, d), where N is the number of points, and d is
 * the dimension. It consists of a (flatten) batch of point clouds.
 *
 * In each batch, the algorithm starts with the sample index specified by
 * ``start_idx``. Then for each point, we maintain the minimum to-sample
 * distance. Finally, we pick the point with the maximum such distance. This
 * process will be repeated for ``sample_points`` - 1 times.
 */
template <DGLDeviceType XPU, typename FloatType, typename IdType>
void FarthestPointSampler(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    IdArray start_idx, IdArray result) {
  const FloatType *array_data = static_cast<FloatType *>(array->data);
  const int64_t point_in_batch = array->shape[0] / batch_size;
  const int64_t dim = array->shape[1];

  // distance
  FloatType *dist_data = static_cast<FloatType *>(dist->data);

  // sample for each cloud in the batch
  IdType *start_idx_data = static_cast<IdType *>(start_idx->data);

  // return value
  IdType *ret_data = static_cast<IdType *>(result->data);

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
template void FarthestPointSampler<kDGLCPU, float, int32_t>(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDGLCPU, float, int64_t>(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDGLCPU, double, int32_t>(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    IdArray start_idx, IdArray result);
template void FarthestPointSampler<kDGLCPU, double, int64_t>(
    NDArray array, int64_t batch_size, int64_t sample_points, NDArray dist,
    IdArray start_idx, IdArray result);

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void WeightedNeighborMatching(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result) {
  const int64_t num_nodes = result->shape[0];
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  IdType *result_data = static_cast<IdType *>(result->data);
  FloatType *weight_data = static_cast<FloatType *>(weight->data);

  // build node visiting order
  IdArray vis_order = RandomPerm<IdType>(num_nodes);
  IdType *vis_order_data = static_cast<IdType *>(vis_order->data);

  for (int64_t n = 0; n < num_nodes; ++n) {
    auto u = vis_order_data[n];

    // if marked
    if (result_data[u] >= 0) continue;

    auto v_max = u;
    FloatType weight_max = 0;

    for (auto e = indptr_data[u]; e < indptr_data[u + 1]; ++e) {
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
template void WeightedNeighborMatching<kDGLCPU, float, int32_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);
template void WeightedNeighborMatching<kDGLCPU, float, int64_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);
template void WeightedNeighborMatching<kDGLCPU, double, int32_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);
template void WeightedNeighborMatching<kDGLCPU, double, int64_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);

template <DGLDeviceType XPU, typename IdType>
void NeighborMatching(const aten::CSRMatrix &csr, IdArray result) {
  const int64_t num_nodes = result->shape[0];
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  IdType *result_data = static_cast<IdType *>(result->data);

  // build vis order
  IdArray u_vis_order = RandomPerm<IdType>(num_nodes);
  IdType *u_vis_order_data = static_cast<IdType *>(u_vis_order->data);
  IdArray v_vis_order = GroupRandomPerm<IdType>(
      indptr_data, csr.indptr->shape[0], csr.indices->shape[0]);
  IdType *v_vis_order_data = static_cast<IdType *>(v_vis_order->data);

  for (int64_t n = 0; n < num_nodes; ++n) {
    auto u = u_vis_order_data[n];

    // if marked
    if (result_data[u] >= 0) continue;

    result_data[u] = u;

    for (auto e = indptr_data[u]; e < indptr_data[u + 1]; ++e) {
      auto v = indices_data[v_vis_order_data[e]];
      if (result_data[v] >= 0) continue;
      result_data[u] = std::min(u, v);
      result_data[v] = result_data[u];
      break;
    }
  }
}
template void NeighborMatching<kDGLCPU, int32_t>(
    const aten::CSRMatrix &csr, IdArray result);
template void NeighborMatching<kDGLCPU, int64_t>(
    const aten::CSRMatrix &csr, IdArray result);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl
