/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file unique_and_compact.cc
 * @brief Unique and compact op.
 */

#include <graphbolt/cuda_ops.h>
#include <graphbolt/unique_and_compact.h>

#include <unordered_map>

#include "./concurrent_id_hash_map.h"
#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace sampling {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor& src_ids, const torch::Tensor& dst_ids,
    const torch::Tensor unique_dst_ids) {
  if (utils::is_on_gpu(src_ids) && utils::is_on_gpu(dst_ids) &&
      utils::is_on_gpu(unique_dst_ids)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "unique_and_compact",
        { return ops::UniqueAndCompact(src_ids, dst_ids, unique_dst_ids); });
  }
  torch::Tensor compacted_src_ids;
  torch::Tensor compacted_dst_ids;
  torch::Tensor unique_ids;
  auto num_dst = unique_dst_ids.size(0);
  torch::Tensor ids = torch::cat({unique_dst_ids, src_ids});
// TODO: Remove this after windows concurrent bug being fixed.
#ifdef _MSC_VER
  AT_DISPATCH_INTEGRAL_TYPES(
      ids.scalar_type(), "unique_and_compact", ([&] {
        std::unordered_map<scalar_t, scalar_t> id_map;
        unique_ids = torch::empty_like(ids);
        auto unique_ids_data = unique_ids.data_ptr<scalar_t>();
        auto ids_data = ids.data_ptr<scalar_t>();
        auto num_ids = ids.size(0);
        scalar_t index = 0;
        for (auto i = 0; i < num_ids; i++) {
          auto id = ids_data[i];
          if (id_map.count(id) == 0) {
            unique_ids_data[index] = id;
            id_map[id] = index++;
          }
        }
        unique_ids = unique_ids.slice(0, 0, index);
        compacted_src_ids = torch::empty_like(src_ids);
        compacted_dst_ids = torch::empty_like(dst_ids);
        num_ids = compacted_src_ids.size(0);
        auto src_ids_data = src_ids.data_ptr<scalar_t>();
        auto dst_ids_data = dst_ids.data_ptr<scalar_t>();
        auto compacted_src_ids_data = compacted_src_ids.data_ptr<scalar_t>();
        auto compacted_dst_ids_data = compacted_dst_ids.data_ptr<scalar_t>();
        torch::parallel_for(0, num_ids, 256, [&](int64_t s, int64_t e) {
          for (int64_t i = s; i < e; i++) {
            auto it = id_map.find(src_ids_data[i]);
            if (it == id_map.end())
              throw std::out_of_range(
                  "Id not found: " + std::to_string(src_ids_data[i]));
            compacted_src_ids_data[i] = it->second;
          }
        });
        num_ids = compacted_dst_ids.size(0);
        torch::parallel_for(0, num_ids, 256, [&](int64_t s, int64_t e) {
          for (int64_t i = s; i < e; i++) {
            auto it = id_map.find(dst_ids_data[i]);
            if (it == id_map.end())
              throw std::out_of_range(
                  "Id not found: " + std::to_string(dst_ids_data[i]));
            compacted_dst_ids_data[i] = it->second;
          }
        });
      }));
#else
  AT_DISPATCH_INTEGRAL_TYPES(ids.scalar_type(), "unique_and_compact", ([&] {
                               ConcurrentIdHashMap<scalar_t> id_map;
                               unique_ids = id_map.Init(ids, num_dst);
                               compacted_src_ids = id_map.MapIds(src_ids);
                               compacted_dst_ids = id_map.MapIds(dst_ids);
                             }));
#endif
  return std::tuple(unique_ids, compacted_src_ids, compacted_dst_ids);
}
}  // namespace sampling
}  // namespace graphbolt
