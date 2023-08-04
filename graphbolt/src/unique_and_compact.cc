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
std::tuple<torch::Tensor, TensorList, TensorList> Unique_and_compact(
    const TensorList& src_ids, const TensorList& dst_ids,
    const torch::Tensor unique_dst_ids) {
  TensorList compacted_src_ids;
  TensorList compacted_dst_ids;
  torch::Tensor unique_ids;
  auto num_dst = unique_dst_ids.size(0);
  std::vector<torch::Tensor> tensors_to_concat = {unique_dst_ids};
  tensors_to_concat.insert(
      tensors_to_concat.end(), src_ids.begin(), src_ids.end());
  torch::Tensor ids = torch::cat(tensors_to_concat);
  AT_DISPATCH_INTEGRAL_TYPES(
      ids.scalar_type(), "unique_and_compact", ([&] {
        ConcurrentIdHashMap<scalar_t> id_map;
        unique_ids = id_map.Init(ids, num_dst);
        for (auto id : src_ids) {
          compacted_src_ids.emplace_back(id_map.MapIds(id));
        }
        for (auto id : dst_ids) {
          compacted_dst_ids.emplace_back(id_map.MapIds(id));
        }
      }));
  return std::tuple(unique_ids, compacted_src_ids, compacted_dst_ids);
}
}  // namespace sampling
}  // namespace graphbolt
#endif  // GRAPHBOLT_UNIQUE_AND_COMPACT_H_