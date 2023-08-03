/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.cc
 * @brief Unique and compact op.
 */
#ifndef GRAPHBOLT_UNIQUE_AND_COMPACT_H_
#define GRAPHBOLT_UNIQUE_AND_COMPACT_H_

#include "./unique_and_compact.h"

#include "./concurrent_id_hash_map.h"

namespace graphbolt {
namespace sampling {
std::vector<torch::Tensor> Unique_and_compact(
    const torch::Tensor& ids, int64_t num_seeds,
    const std::vector<torch::Tensor>& id_to_compact) {
  std::vector<torch::Tensor> compacted_ids;
  AT_DISPATCH_INTEGRAL_TYPES(
      ids.scalar_type(), "unique_and_compact", ([&] {
        ConcurrentIdHashMap<scalar_t> id_map;
        auto unique_ids = id_map.Init(ids, num_seeds);
        compacted_ids.push_back(unique_ids);
        for (const torch::Tensor& original_id : id_to_compact) {
          compacted_ids.emplace_back(id_map.MapIds(original_id));
        }
      }));
  return compacted_ids;
}
}  // namespace sampling
}  // namespace graphbolt
#endif  // GRAPHBOLT_UNIQUE_AND_COMPACT_H_