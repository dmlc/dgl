/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.cc
 * @brief Unique and compact op.
 */

#include <graphbolt/cuda_ops.h>
#include <graphbolt/unique_and_compact.h>

#include "./concurrent_id_hash_map.h"
#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace sampling {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
UniqueAndCompact(
    const torch::Tensor& src_ids, const torch::Tensor& dst_ids,
    const torch::Tensor unique_dst_ids, const int64_t rank,
    const int64_t world_size) {
  if (utils::is_on_gpu(src_ids) && utils::is_on_gpu(dst_ids) &&
      utils::is_on_gpu(unique_dst_ids)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "unique_and_compact", {
          return ops::UniqueAndCompact(
              src_ids, dst_ids, unique_dst_ids, rank, world_size);
        });
  }
  TORCH_CHECK(
      world_size <= 1,
      "Cooperative Minibatching (arXiv:2310.12403) is supported only on GPUs.");
  auto num_dst = unique_dst_ids.size(0);
  torch::Tensor ids = torch::cat({unique_dst_ids, src_ids});
  auto [unique_ids, compacted_src, compacted_dst] = AT_DISPATCH_INDEX_TYPES(
      ids.scalar_type(), "unique_and_compact", ([&] {
        ConcurrentIdHashMap<index_t> id_map(ids, num_dst);
        return std::make_tuple(
            id_map.GetUniqueIds(), id_map.MapIds(src_ids),
            id_map.MapIds(dst_ids));
      }));
  auto offsets = torch::zeros(2, c10::TensorOptions().dtype(torch::kInt64));
  offsets.data_ptr<int64_t>()[1] = unique_ids.size(0);
  return {unique_ids, compacted_src, compacted_dst, offsets};
}

std::vector<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatched(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor> unique_dst_ids, const int64_t rank,
    const int64_t world_size) {
  TORCH_CHECK(
      src_ids.size() == dst_ids.size() &&
          dst_ids.size() == unique_dst_ids.size(),
      "The batch dimension of the parameters need to be identical.");
  bool all_on_gpu = true;
  for (std::size_t i = 0; i < src_ids.size(); i++) {
    all_on_gpu = all_on_gpu && utils::is_on_gpu(src_ids[i]) &&
                 utils::is_on_gpu(dst_ids[i]) &&
                 utils::is_on_gpu(unique_dst_ids[i]);
    if (!all_on_gpu) break;
  }
  if (all_on_gpu) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "unique_and_compact", {
          return ops::UniqueAndCompactBatched(
              src_ids, dst_ids, unique_dst_ids, rank, world_size);
        });
  }
  std::vector<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
      results;
  results.reserve(src_ids.size());
  for (std::size_t i = 0; i < src_ids.size(); i++) {
    results.emplace_back(UniqueAndCompact(
        src_ids[i], dst_ids[i], unique_dst_ids[i], rank, world_size));
  }
  return results;
}

c10::intrusive_ptr<Future<std::vector<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>>>
UniqueAndCompactBatchedAsync(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor> unique_dst_ids, const int64_t rank,
    const int64_t world_size) {
  return async(
      [=] {
        return UniqueAndCompactBatched(
            src_ids, dst_ids, unique_dst_ids, rank, world_size);
      },
      utils::is_on_gpu(src_ids.at(0)));
}

}  // namespace sampling
}  // namespace graphbolt
