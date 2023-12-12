/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.cc
 * @brief Unique and compact op.
 */

#include <graphbolt/unique_and_compact.h>
#include <graphbolt/cuda_ops.h>

#include "./concurrent_id_hash_map.h"
#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace sampling {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor& src_ids, const torch::Tensor& dst_ids,
    const torch::Tensor unique_dst_ids) {
  if (utils::is_accessible_from_gpu(src_ids) &&
      utils::is_accessible_from_gpu(dst_ids) && utils::is_accessible_from_gpu(unique_dst_ids)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "unique_and_compact",
        { return ops::UniqueAndCompact(src_ids, dst_ids, unique_dst_ids); });
  }
  torch::Tensor compacted_src_ids;
  torch::Tensor compacted_dst_ids;
  torch::Tensor unique_ids;
  auto num_dst = unique_dst_ids.size(0);
  torch::Tensor ids = torch::cat({unique_dst_ids, src_ids});
  AT_DISPATCH_INTEGRAL_TYPES(ids.scalar_type(), "unique_and_compact", ([&] {
                               ConcurrentIdHashMap<scalar_t> id_map;
                               unique_ids = id_map.Init(ids, num_dst);
                               compacted_src_ids = id_map.MapIds(src_ids);
                               compacted_dst_ids = id_map.MapIds(dst_ids);
                             }));
  return std::tuple(unique_ids, compacted_src_ids, compacted_dst_ids);
}
}  // namespace sampling
}  // namespace graphbolt
