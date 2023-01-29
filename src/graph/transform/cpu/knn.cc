/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/transform/cpu/knn.cc
 * @brief k-nearest-neighbor (KNN) implementation
 */

#include "../knn.h"

#include <dgl/random.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/parallel_for.h>
#include <dmlc/omp.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

#include "kdtree_ndarray_adapter.h"

using namespace dgl::runtime;
using namespace dgl::transform::knn_utils;
namespace dgl {
namespace transform {
namespace impl {

// This value is directly from pynndescent
static constexpr int NN_DESCENT_BLOCK_SIZE = 16384;

/**
 * @brief Compute Euclidean distance between two vectors, return positive
 *  infinite value if the intermediate distance is greater than the worst
 *  distance.
 */
template <typename FloatType, typename IdType>
FloatType EuclideanDistWithCheck(
    const FloatType* vec1, const FloatType* vec2, int64_t dim,
    FloatType worst_dist = std::numeric_limits<FloatType>::max()) {
  FloatType dist = 0;
  bool early_stop = false;

  for (IdType idx = 0; idx < dim; ++idx) {
    dist += (vec1[idx] - vec2[idx]) * (vec1[idx] - vec2[idx]);
    if (dist > worst_dist) {
      early_stop = true;
      break;
    }
  }

  if (early_stop) {
    return std::numeric_limits<FloatType>::max();
  } else {
    return dist;
  }
}

/** @brief Compute Euclidean distance between two vectors */
template <typename FloatType, typename IdType>
FloatType EuclideanDist(
    const FloatType* vec1, const FloatType* vec2, int64_t dim) {
  FloatType dist = 0;

  for (IdType idx = 0; idx < dim; ++idx) {
    dist += (vec1[idx] - vec2[idx]) * (vec1[idx] - vec2[idx]);
  }

  return dist;
}

/** @brief Insert a new element into a heap */
template <typename FloatType, typename IdType>
void HeapInsert(
    IdType* out, FloatType* dist, IdType new_id, FloatType new_dist, int k,
    bool check_repeat = false) {
  if (new_dist > dist[0]) return;

  // check if we have it
  if (check_repeat) {
    for (IdType i = 0; i < k; ++i) {
      if (out[i] == new_id) return;
    }
  }

  IdType left_idx = 0, right_idx = 0, curr_idx = 0, swap_idx = 0;
  dist[0] = new_dist;
  out[0] = new_id;
  while (true) {
    left_idx = 2 * curr_idx + 1;
    right_idx = left_idx + 1;
    swap_idx = curr_idx;
    if (left_idx < k && dist[left_idx] > dist[swap_idx]) {
      swap_idx = left_idx;
    }
    if (right_idx < k && dist[right_idx] > dist[swap_idx]) {
      swap_idx = right_idx;
    }
    if (swap_idx != curr_idx) {
      std::swap(dist[curr_idx], dist[swap_idx]);
      std::swap(out[curr_idx], out[swap_idx]);
      curr_idx = swap_idx;
    } else {
      break;
    }
  }
}

/** @brief Insert a new element and its flag into heap, return 1 if insert
 * successfully */
template <typename FloatType, typename IdType>
int FlaggedHeapInsert(
    IdType* out, FloatType* dist, bool* flag, IdType new_id, FloatType new_dist,
    bool new_flag, int k, bool check_repeat = false) {
  if (new_dist > dist[0]) return 0;

  if (check_repeat) {
    for (IdType i = 0; i < k; ++i) {
      if (out[i] == new_id) return 0;
    }
  }

  IdType left_idx = 0, right_idx = 0, curr_idx = 0, swap_idx = 0;
  dist[0] = new_dist;
  out[0] = new_id;
  flag[0] = new_flag;
  while (true) {
    left_idx = 2 * curr_idx + 1;
    right_idx = left_idx + 1;
    swap_idx = curr_idx;
    if (left_idx < k && dist[left_idx] > dist[swap_idx]) {
      swap_idx = left_idx;
    }
    if (right_idx < k && dist[right_idx] > dist[swap_idx]) {
      swap_idx = right_idx;
    }
    if (swap_idx != curr_idx) {
      std::swap(dist[curr_idx], dist[swap_idx]);
      std::swap(out[curr_idx], out[swap_idx]);
      std::swap(flag[curr_idx], flag[swap_idx]);
      curr_idx = swap_idx;
    } else {
      break;
    }
  }
  return 1;
}

/** @brief Build heap for each point. Used by NN-descent */
template <typename FloatType, typename IdType>
void BuildHeap(IdType* index, FloatType* dist, int k) {
  for (int i = k / 2 - 1; i >= 0; --i) {
    IdType idx = i;
    while (true) {
      IdType largest = idx;
      IdType left = idx * 2 + 1;
      IdType right = left + 1;
      if (left < k && dist[left] > dist[largest]) {
        largest = left;
      }
      if (right < k && dist[right] > dist[largest]) {
        largest = right;
      }
      if (largest != idx) {
        std::swap(index[largest], index[idx]);
        std::swap(dist[largest], dist[idx]);
        idx = largest;
      } else {
        break;
      }
    }
  }
}

/**
 * @brief Neighbor update process in NN-descent. The distance between
 *  two points are computed. If this new distance is less than any worst
 *  distance of these two points, we update the neighborhood of that point.
 */
template <typename FloatType, typename IdType>
int UpdateNeighbors(
    IdType* neighbors, FloatType* dists, const FloatType* points, bool* flags,
    IdType c1, IdType c2, IdType point_start, int64_t feature_size, int k) {
  IdType c1_local = c1 - point_start, c2_local = c2 - point_start;
  FloatType worst_c1_dist = dists[c1_local * k];
  FloatType worst_c2_dist = dists[c2_local * k];
  FloatType new_dist = EuclideanDistWithCheck<FloatType, IdType>(
      points + c1 * feature_size, points + c2 * feature_size, feature_size,
      std::max(worst_c1_dist, worst_c2_dist));

  int num_updates = 0;
  if (new_dist < worst_c1_dist) {
    ++num_updates;
#pragma omp critical
    {
      FlaggedHeapInsert<FloatType, IdType>(
          neighbors + c1 * k, dists + c1_local * k, flags + c1_local * k, c2,
          new_dist, true, k, true);
    }
  }
  if (new_dist < worst_c2_dist) {
    ++num_updates;
#pragma omp critical
    {
      FlaggedHeapInsert<FloatType, IdType>(
          neighbors + c2 * k, dists + c2_local * k, flags + c2_local * k, c1,
          new_dist, true, k, true);
    }
  }
  return num_updates;
}

/** @brief The kd-tree implementation of K-Nearest Neighbors */
template <typename FloatType, typename IdType>
void KdTreeKNN(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result) {
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  for (int64_t b = 0; b < batch_size; ++b) {
    auto d_offset = data_offsets_data[b];
    auto d_length = data_offsets_data[b + 1] - d_offset;
    auto q_offset = query_offsets_data[b];
    auto q_length = query_offsets_data[b + 1] - q_offset;
    auto out_offset = k * q_offset;

    // create view for each segment
    const NDArray current_data_points =
        const_cast<NDArray*>(&data_points)
            ->CreateView(
                {d_length, feature_size}, data_points->dtype,
                d_offset * feature_size * sizeof(FloatType));
    const FloatType* current_query_pts_data =
        query_points_data + q_offset * feature_size;

    KDTreeNDArrayAdapter<FloatType, IdType> kdtree(
        feature_size, current_data_points);

    // query
    parallel_for(0, q_length, [&](IdType b, IdType e) {
      for (auto q = b; q < e; ++q) {
        std::vector<IdType> out_buffer(k);
        std::vector<FloatType> out_dist_buffer(k);

        auto curr_out_offset = k * q + out_offset;
        const FloatType* q_point = current_query_pts_data + q * feature_size;
        size_t num_matches = kdtree.GetIndex()->knnSearch(
            q_point, k, out_buffer.data(), out_dist_buffer.data());

        for (size_t i = 0; i < num_matches; ++i) {
          query_out[curr_out_offset] = q + q_offset;
          data_out[curr_out_offset] = out_buffer[i] + d_offset;
          curr_out_offset++;
        }
      }
    });
  }
}

template <typename FloatType, typename IdType>
void BruteForceKNN(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result) {
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* data_points_data = data_points.Ptr<FloatType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  for (int64_t b = 0; b < batch_size; ++b) {
    IdType d_start = data_offsets_data[b], d_end = data_offsets_data[b + 1];
    IdType q_start = query_offsets_data[b], q_end = query_offsets_data[b + 1];

    std::vector<FloatType> dist_buffer(k);

    parallel_for(q_start, q_end, [&](IdType b, IdType e) {
      for (auto q_idx = b; q_idx < e; ++q_idx) {
        std::vector<FloatType> dist_buffer(k);
        for (IdType k_idx = 0; k_idx < k; ++k_idx) {
          query_out[q_idx * k + k_idx] = q_idx;
          dist_buffer[k_idx] = std::numeric_limits<FloatType>::max();
        }
        FloatType worst_dist = std::numeric_limits<FloatType>::max();

        for (IdType d_idx = d_start; d_idx < d_end; ++d_idx) {
          FloatType tmp_dist = EuclideanDistWithCheck<FloatType, IdType>(
              query_points_data + q_idx * feature_size,
              data_points_data + d_idx * feature_size, feature_size,
              worst_dist);

          if (tmp_dist == std::numeric_limits<FloatType>::max()) {
            continue;
          }

          IdType out_offset = q_idx * k;
          HeapInsert<FloatType, IdType>(
              data_out + out_offset, dist_buffer.data(), d_idx, tmp_dist, k);
          worst_dist = dist_buffer[0];
        }
      }
    });
  }
}
}  // namespace impl

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void KNN(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm) {
  if (algorithm == std::string("kd-tree")) {
    impl::KdTreeKNN<FloatType, IdType>(
        data_points, data_offsets, query_points, query_offsets, k, result);
  } else if (algorithm == std::string("bruteforce")) {
    impl::BruteForceKNN<FloatType, IdType>(
        data_points, data_offsets, query_points, query_offsets, k, result);
  } else {
    LOG(FATAL) << "Algorithm " << algorithm << " is not supported on CPU";
  }
}

template <DGLDeviceType XPU, typename FloatType, typename IdType>
void NNDescent(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta) {
  using nnd_updates_t =
      std::vector<std::vector<std::tuple<IdType, IdType, FloatType>>>;
  const auto& ctx = points->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t num_nodes = points->shape[0];
  const int64_t batch_size = offsets->shape[0] - 1;
  const int64_t feature_size = points->shape[1];
  const IdType* offsets_data = offsets.Ptr<IdType>();
  const FloatType* points_data = points.Ptr<FloatType>();

  IdType* central_nodes = result.Ptr<IdType>();
  IdType* neighbors = central_nodes + k * num_nodes;
  int64_t max_segment_size = 0;

  // find max segment
  for (IdType b = 0; b < batch_size; ++b) {
    if (max_segment_size < offsets_data[b + 1] - offsets_data[b])
      max_segment_size = offsets_data[b + 1] - offsets_data[b];
  }

  // allocate memory for candidate, sampling pool, distance and flag
  IdType* new_candidates = static_cast<IdType*>(device->AllocWorkspace(
      ctx, max_segment_size * num_candidates * sizeof(IdType)));
  IdType* old_candidates = static_cast<IdType*>(device->AllocWorkspace(
      ctx, max_segment_size * num_candidates * sizeof(IdType)));
  FloatType* new_candidates_dists =
      static_cast<FloatType*>(device->AllocWorkspace(
          ctx, max_segment_size * num_candidates * sizeof(FloatType)));
  FloatType* old_candidates_dists =
      static_cast<FloatType*>(device->AllocWorkspace(
          ctx, max_segment_size * num_candidates * sizeof(FloatType)));
  FloatType* neighbors_dists = static_cast<FloatType*>(
      device->AllocWorkspace(ctx, max_segment_size * k * sizeof(FloatType)));
  bool* flags = static_cast<bool*>(
      device->AllocWorkspace(ctx, max_segment_size * k * sizeof(bool)));

  for (IdType b = 0; b < batch_size; ++b) {
    IdType point_idx_start = offsets_data[b],
           point_idx_end = offsets_data[b + 1];
    IdType segment_size = point_idx_end - point_idx_start;

    // random initialization
    runtime::parallel_for(
        point_idx_start, point_idx_end, [&](size_t b, size_t e) {
          for (auto i = b; i < e; ++i) {
            IdType local_idx = i - point_idx_start;

            dgl::RandomEngine::ThreadLocal()->UniformChoice<IdType>(
                k, segment_size, neighbors + i * k, false);

            for (IdType n = 0; n < k; ++n) {
              central_nodes[i * k + n] = i;
              neighbors[i * k + n] += point_idx_start;
              flags[local_idx * k + n] = true;
              neighbors_dists[local_idx * k + n] =
                  impl::EuclideanDist<FloatType, IdType>(
                      points_data + i * feature_size,
                      points_data + neighbors[i * k + n] * feature_size,
                      feature_size);
            }
            impl::BuildHeap<FloatType, IdType>(
                neighbors + i * k, neighbors_dists + local_idx * k, k);
          }
        });

    size_t num_updates = 0;
    for (int iter = 0; iter < num_iters; ++iter) {
      num_updates = 0;

      // initialize candidates array as empty value
      runtime::parallel_for(
          point_idx_start, point_idx_end, [&](size_t b, size_t e) {
            for (auto i = b; i < e; ++i) {
              IdType local_idx = i - point_idx_start;
              for (IdType c = 0; c < num_candidates; ++c) {
                new_candidates[local_idx * num_candidates + c] = num_nodes;
                old_candidates[local_idx * num_candidates + c] = num_nodes;
                new_candidates_dists[local_idx * num_candidates + c] =
                    std::numeric_limits<FloatType>::max();
                old_candidates_dists[local_idx * num_candidates + c] =
                    std::numeric_limits<FloatType>::max();
              }
            }
          });

      // randomly select neighbors as candidates
      int num_threads = omp_get_max_threads();
      runtime::parallel_for(0, num_threads, [&](IdType b, IdType e) {
        for (auto tid = b; tid < e; ++tid) {
          for (IdType i = point_idx_start; i < point_idx_end; ++i) {
            IdType local_idx = i - point_idx_start;
            for (IdType n = 0; n < k; ++n) {
              IdType neighbor_idx = neighbors[i * k + n];
              bool is_new = flags[local_idx * k + n];
              IdType local_neighbor_idx = neighbor_idx - point_idx_start;
              FloatType random_dist =
                  dgl::RandomEngine::ThreadLocal()->Uniform<FloatType>();

              if (is_new) {
                if (local_idx % num_threads == tid) {
                  impl::HeapInsert<FloatType, IdType>(
                      new_candidates + local_idx * num_candidates,
                      new_candidates_dists + local_idx * num_candidates,
                      neighbor_idx, random_dist, num_candidates, true);
                }
                if (local_neighbor_idx % num_threads == tid) {
                  impl::HeapInsert<FloatType, IdType>(
                      new_candidates + local_neighbor_idx * num_candidates,
                      new_candidates_dists +
                          local_neighbor_idx * num_candidates,
                      i, random_dist, num_candidates, true);
                }
              } else {
                if (local_idx % num_threads == tid) {
                  impl::HeapInsert<FloatType, IdType>(
                      old_candidates + local_idx * num_candidates,
                      old_candidates_dists + local_idx * num_candidates,
                      neighbor_idx, random_dist, num_candidates, true);
                }
                if (local_neighbor_idx % num_threads == tid) {
                  impl::HeapInsert<FloatType, IdType>(
                      old_candidates + local_neighbor_idx * num_candidates,
                      old_candidates_dists +
                          local_neighbor_idx * num_candidates,
                      i, random_dist, num_candidates, true);
                }
              }
            }
          }
        }
      });

      // mark all elements in new_candidates as false
      runtime::parallel_for(
          point_idx_start, point_idx_end, [&](size_t b, size_t e) {
            for (auto i = b; i < e; ++i) {
              IdType local_idx = i - point_idx_start;
              for (IdType n = 0; n < k; ++n) {
                IdType n_idx = neighbors[i * k + n];

                for (IdType c = 0; c < num_candidates; ++c) {
                  if (new_candidates[local_idx * num_candidates + c] == n_idx) {
                    flags[local_idx * k + n] = false;
                    break;
                  }
                }
              }
            }
          });

      // update neighbors block by block
      for (IdType block_start = point_idx_start; block_start < point_idx_end;
           block_start += impl::NN_DESCENT_BLOCK_SIZE) {
        IdType block_end =
            std::min(point_idx_end, block_start + impl::NN_DESCENT_BLOCK_SIZE);
        IdType block_size = block_end - block_start;
        nnd_updates_t updates(block_size);

        // generate updates
        runtime::parallel_for(block_start, block_end, [&](size_t b, size_t e) {
          for (auto i = b; i < e; ++i) {
            IdType local_idx = i - point_idx_start;

            for (IdType c1 = 0; c1 < num_candidates; ++c1) {
              IdType new_c1 = new_candidates[local_idx * num_candidates + c1];
              if (new_c1 == num_nodes) continue;
              IdType c1_local = new_c1 - point_idx_start;

              // new-new
              for (IdType c2 = c1; c2 < num_candidates; ++c2) {
                IdType new_c2 = new_candidates[local_idx * num_candidates + c2];
                if (new_c2 == num_nodes) continue;
                IdType c2_local = new_c2 - point_idx_start;

                FloatType worst_c1_dist = neighbors_dists[c1_local * k];
                FloatType worst_c2_dist = neighbors_dists[c2_local * k];
                FloatType new_dist =
                    impl::EuclideanDistWithCheck<FloatType, IdType>(
                        points_data + new_c1 * feature_size,
                        points_data + new_c2 * feature_size, feature_size,
                        std::max(worst_c1_dist, worst_c2_dist));

                if (new_dist < worst_c1_dist || new_dist < worst_c2_dist) {
                  updates[i - block_start].push_back(
                      std::make_tuple(new_c1, new_c2, new_dist));
                }
              }

              // new-old
              for (IdType c2 = 0; c2 < num_candidates; ++c2) {
                IdType old_c2 = old_candidates[local_idx * num_candidates + c2];
                if (old_c2 == num_nodes) continue;
                IdType c2_local = old_c2 - point_idx_start;

                FloatType worst_c1_dist = neighbors_dists[c1_local * k];
                FloatType worst_c2_dist = neighbors_dists[c2_local * k];
                FloatType new_dist =
                    impl::EuclideanDistWithCheck<FloatType, IdType>(
                        points_data + new_c1 * feature_size,
                        points_data + old_c2 * feature_size, feature_size,
                        std::max(worst_c1_dist, worst_c2_dist));

                if (new_dist < worst_c1_dist || new_dist < worst_c2_dist) {
                  updates[i - block_start].push_back(
                      std::make_tuple(new_c1, old_c2, new_dist));
                }
              }
            }
          }
        });

        int tid;
#pragma omp parallel private(tid, num_threads) reduction(+ : num_updates)
        {
          tid = omp_get_thread_num();
          num_threads = omp_get_num_threads();
          for (IdType i = 0; i < block_size; ++i) {
            for (const auto& u : updates[i]) {
              IdType p1, p2;
              FloatType d;
              std::tie(p1, p2, d) = u;
              IdType p1_local = p1 - point_idx_start;
              IdType p2_local = p2 - point_idx_start;

              if (p1 % num_threads == tid) {
                num_updates += impl::FlaggedHeapInsert<FloatType, IdType>(
                    neighbors + p1 * k, neighbors_dists + p1_local * k,
                    flags + p1_local * k, p2, d, true, k, true);
              }
              if (p2 % num_threads == tid) {
                num_updates += impl::FlaggedHeapInsert<FloatType, IdType>(
                    neighbors + p2 * k, neighbors_dists + p2_local * k,
                    flags + p2_local * k, p1, d, true, k, true);
              }
            }
          }
        }
      }

      // early abort
      if (num_updates <= static_cast<size_t>(delta * k * segment_size)) {
        break;
      }
    }
  }

  device->FreeWorkspace(ctx, new_candidates);
  device->FreeWorkspace(ctx, old_candidates);
  device->FreeWorkspace(ctx, new_candidates_dists);
  device->FreeWorkspace(ctx, old_candidates_dists);
  device->FreeWorkspace(ctx, neighbors_dists);
  device->FreeWorkspace(ctx, flags);
}

template void KNN<kDGLCPU, float, int32_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);
template void KNN<kDGLCPU, float, int64_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);
template void KNN<kDGLCPU, double, int32_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);
template void KNN<kDGLCPU, double, int64_t>(
    const NDArray& data_points, const IdArray& data_offsets,
    const NDArray& query_points, const IdArray& query_offsets, const int k,
    IdArray result, const std::string& algorithm);

template void NNDescent<kDGLCPU, float, int32_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);
template void NNDescent<kDGLCPU, float, int64_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);
template void NNDescent<kDGLCPU, double, int32_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);
template void NNDescent<kDGLCPU, double, int64_t>(
    const NDArray& points, const IdArray& offsets, IdArray result, const int k,
    const int num_iters, const int num_candidates, const double delta);
}  // namespace transform
}  // namespace dgl
