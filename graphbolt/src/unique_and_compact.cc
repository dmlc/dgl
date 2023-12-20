/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.cc
 * @brief Unique and compact op.
 */

#include <graphbolt/unique_and_compact.h>

#include "./concurrent_id_hash_map.h"
#include "./utils.h"

namespace graphbolt {
namespace sampling {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor& src_ids, const torch::Tensor& dst_ids,
    const torch::Tensor unique_dst_ids) {
  torch::Tensor compacted_src_ids;
  torch::Tensor compacted_dst_ids;
  torch::Tensor unique_ids;
  auto num_dst = unique_dst_ids.size(0);
  // Workaround until #6738 is merged.
  auto src_ids_cpu = src_ids.cpu();
  auto dst_ids_cpu = dst_ids.cpu();
  auto unique_dst_ids_cpu = unique_dst_ids.cpu();
  torch::Tensor ids = torch::cat({unique_dst_ids_cpu, src_ids_cpu});
  AT_DISPATCH_INTEGRAL_TYPES(ids.scalar_type(), "unique_and_compact", ([&] {
                               ConcurrentIdHashMap<scalar_t> id_map;
                               unique_ids = id_map.Init(ids, num_dst);
                               compacted_src_ids = id_map.MapIds(src_ids_cpu);
                               compacted_dst_ids = id_map.MapIds(dst_ids_cpu);
                             }));
  if (utils::is_accessible_from_gpu(src_ids) ||
      utils::is_accessible_from_gpu(unique_dst_ids))
    unique_ids = unique_ids.cuda();
  if (utils::is_accessible_from_gpu(src_ids))
    compacted_src_ids = compacted_src_ids.cuda();
  if (utils::is_accessible_from_gpu(dst_ids))
    compacted_dst_ids = compacted_dst_ids.cuda();
  return std::tuple(unique_ids, compacted_src_ids, compacted_dst_ids);
}
}  // namespace sampling
}  // namespace graphbolt
