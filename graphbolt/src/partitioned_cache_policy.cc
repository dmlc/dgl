/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file partitioned_cache_policy.cc
 * @brief Partitioned cache policy implementation on the CPU.
 */
#include "./partitioned_cache_policy.h"

#include <numeric>

#include "./utils.h"

namespace graphbolt {
namespace storage {

constexpr int kIntGrainSize = 64;

template <typename CachePolicy>
PartitionedCachePolicy::PartitionedCachePolicy(
    CachePolicy, int64_t capacity, int64_t num_partitions)
    : capacity_(capacity) {
  TORCH_CHECK(num_partitions >= 1, "# partitions need to be positive.");
  for (int64_t i = 0; i < num_partitions; i++) {
    const auto begin = i * capacity / num_partitions;
    const auto end = (i + 1) * capacity / num_partitions;
    policies_.emplace_back(std::make_unique<CachePolicy>(end - begin));
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PartitionedCachePolicy::Partition(torch::Tensor keys) {
  const int64_t num_parts = policies_.size();
  torch::Tensor offsets = torch::zeros(
      num_parts * num_parts + 1, keys.options().dtype(torch::kInt64));
  auto indices = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  auto part_id = torch::empty_like(keys, keys.options().dtype(torch::kInt32));
  const auto num_keys = keys.size(0);
  auto part_id_ptr = part_id.data_ptr<int32_t>();
  auto offsets_ptr = offsets.data_ptr<int64_t>();
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "PartitionedCachePolicy::partition", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        torch::parallel_for(0, num_parts, 1, [&](int64_t begin, int64_t end) {
          if (begin == end) return;
          TORCH_CHECK(end - begin == 1);
          const auto tid = begin;
          begin = tid * num_keys / num_parts;
          end = (tid + 1) * num_keys / num_parts;
          for (int64_t i = begin; i < end; i++) {
            const auto part_id = PartAssignment(keys_ptr[i]);
            offsets_ptr[tid * num_parts + part_id]++;
            part_id_ptr[i] = part_id;
          }
        });
      }));

  // Transpose the offsets tensor, take cumsum and transpose back.
  auto offsets_permuted = torch::empty_like(offsets);
  auto offsets_permuted_ptr = offsets_permuted.data_ptr<int64_t>();
  torch::parallel_for(
      0, num_parts * num_parts, kIntGrainSize, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          auto part_id = i % num_parts;
          auto tid = i / num_parts;
          // + 1 so that we have exclusive_scan after torch.cumsum().
          offsets_permuted_ptr[part_id * num_parts + tid + 1] = offsets_ptr[i];
        }
      });
  offsets_permuted_ptr[0] = 0;
  offsets = offsets_permuted.cumsum(0);
  offsets_ptr = offsets.data_ptr<int64_t>();
  torch::parallel_for(
      0, num_parts * num_parts, kIntGrainSize, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          auto part_id = i % num_parts;
          auto tid = i / num_parts;
          offsets_permuted_ptr[i] = offsets_ptr[part_id * num_parts + tid];
        }
      });
  auto indices_ptr = indices.data_ptr<int64_t>();
  auto permuted_keys = torch::empty_like(keys);
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "PartitionedCachePolicy::partition", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto permuted_keys_ptr = permuted_keys.data_ptr<index_t>();
        torch::parallel_for(0, num_parts, 1, [&](int64_t begin, int64_t end) {
          if (begin == end) return;
          const auto tid = begin;
          begin = tid * num_keys / num_parts;
          end = (tid + 1) * num_keys / num_parts;
          for (int64_t i = begin; i < end; i++) {
            const auto part_id = part_id_ptr[i];
            auto& offset = offsets_permuted_ptr[tid * num_parts + part_id];
            indices_ptr[offset] = i;
            permuted_keys_ptr[offset++] = keys_ptr[i];
          }
        });
      }));
  return {
      offsets.slice(0, 0, offsets.size(0), num_parts).contiguous(), indices,
      permuted_keys};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
PartitionedCachePolicy::Query(torch::Tensor keys) {
  if (policies_.size() == 1) {
    std::lock_guard lock(mtx_);
    return policies_[0]->Query(keys);
  };
  torch::Tensor offsets, indices, permuted_keys;
  std::tie(offsets, indices, permuted_keys) = Partition(keys);
  auto offsets_ptr = offsets.data_ptr<int64_t>();
  auto indices_ptr = indices.data_ptr<int64_t>();
  std::vector<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
      results(policies_.size());
  torch::Tensor result_offsets_tensor =
      torch::empty(policies_.size() * 2 + 1, offsets.options());
  auto result_offsets = result_offsets_tensor.data_ptr<int64_t>();
  {
    std::lock_guard lock(mtx_);
    torch::parallel_for(
        0, policies_.size(), 1, [&](int64_t begin, int64_t end) {
          if (begin == end) return;
          TORCH_CHECK(end - begin == 1);
          const auto tid = begin;
          begin = offsets_ptr[tid];
          end = offsets_ptr[tid + 1];
          results[tid] =
              policies_.at(tid)->Query(permuted_keys.slice(0, begin, end));
          result_offsets[tid] = std::get<0>(results[tid]).size(0);
          result_offsets[tid + policies_.size()] =
              std::get<2>(results[tid]).size(0);
        });
  }
  std::exclusive_scan(
      result_offsets, result_offsets + result_offsets_tensor.size(0),
      result_offsets, 0);
  torch::Tensor positions = torch::empty(
      result_offsets[policies_.size()],
      std::get<0>(results[0]).options().pinned_memory(utils::is_pinned(keys)));
  torch::Tensor output_indices = torch::empty_like(
      indices, indices.options().pinned_memory(utils::is_pinned(keys)));
  torch::Tensor missing_keys = torch::empty(
      indices.size(0) - positions.size(0),
      std::get<2>(results[0]).options().pinned_memory(utils::is_pinned(keys)));
  torch::Tensor found_keys = torch::empty(
      positions.size(0),
      std::get<3>(results[0]).options().pinned_memory(utils::is_pinned(keys)));
  auto output_indices_ptr = output_indices.data_ptr<int64_t>();
  torch::parallel_for(0, policies_.size(), 1, [&](int64_t begin, int64_t end) {
    if (begin == end) return;
    const auto tid = begin;
    auto out_index_ptr = indices_ptr + offsets_ptr[tid];
    begin = result_offsets[tid];
    end = result_offsets[tid + 1];
    const auto num_selected = end - begin;
    auto indices_ptr = std::get<1>(results[tid]).data_ptr<int64_t>();
    for (int64_t i = 0; i < num_selected; i++) {
      output_indices_ptr[begin + i] = out_index_ptr[indices_ptr[i]];
    }
    auto selected_positions_ptr = std::get<0>(results[tid]).data_ptr<int64_t>();
    std::transform(
        selected_positions_ptr, selected_positions_ptr + num_selected,
        positions.data_ptr<int64_t>() + begin,
        [off = tid * capacity_ / policies_.size()](auto x) { return x + off; });
    std::memcpy(
        reinterpret_cast<std::byte*>(found_keys.data_ptr()) +
            begin * found_keys.element_size(),
        std::get<3>(results[tid]).data_ptr(),
        num_selected * found_keys.element_size());
    begin = result_offsets[policies_.size() + tid];
    end = result_offsets[policies_.size() + tid + 1];
    const auto num_missing = end - begin;
    for (int64_t i = 0; i < num_missing; i++) {
      output_indices_ptr[begin + i] =
          out_index_ptr[indices_ptr[i + num_selected]];
    }
    std::memcpy(
        reinterpret_cast<std::byte*>(missing_keys.data_ptr()) +
            (begin - positions.size(0)) * missing_keys.element_size(),
        std::get<2>(results[tid]).data_ptr(),
        num_missing * missing_keys.element_size());
  });
  return std::make_tuple(positions, output_indices, missing_keys, found_keys);
}

c10::intrusive_ptr<Future<std::vector<torch::Tensor>>>
PartitionedCachePolicy::QueryAsync(torch::Tensor keys) {
  return async([=] {
    auto [positions, output_indices, missing_keys, found_keys] = Query(keys);
    return std::vector{positions, output_indices, missing_keys, found_keys};
  });
}

torch::Tensor PartitionedCachePolicy::Replace(torch::Tensor keys) {
  if (policies_.size() == 1) {
    std::lock_guard lock(mtx_);
    return policies_[0]->Replace(keys);
  }
  torch::Tensor offsets, indices, permuted_keys;
  std::tie(offsets, indices, permuted_keys) = Partition(keys);
  auto output_positions = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto offsets_ptr = offsets.data_ptr<int64_t>();
  auto indices_ptr = indices.data_ptr<int64_t>();
  auto output_positions_ptr = output_positions.data_ptr<int64_t>();
  std::lock_guard lock(mtx_);
  torch::parallel_for(0, policies_.size(), 1, [&](int64_t begin, int64_t end) {
    if (begin == end) return;
    const auto tid = begin;
    begin = offsets_ptr[tid];
    end = offsets_ptr[tid + 1];
    torch::Tensor positions =
        policies_.at(tid)->Replace(permuted_keys.slice(0, begin, end));
    auto positions_ptr = positions.data_ptr<int64_t>();
    const auto off = tid * capacity_ / policies_.size();
    for (int64_t i = 0; i < positions.size(0); i++) {
      output_positions_ptr[indices_ptr[begin + i]] = positions_ptr[i] + off;
    }
  });
  return output_positions;
}

c10::intrusive_ptr<Future<torch::Tensor>> PartitionedCachePolicy::ReplaceAsync(
    torch::Tensor keys) {
  return async([=] { return Replace(keys); });
}

template <bool write>
void PartitionedCachePolicy::ReadingWritingCompletedImpl(torch::Tensor keys) {
  if (policies_.size() == 1) {
    std::lock_guard lock(mtx_);
    if constexpr (write)
      policies_[0]->WritingCompleted(keys);
    else
      policies_[0]->ReadingCompleted(keys);
    return;
  }
  torch::Tensor offsets, indices, permuted_keys;
  std::tie(offsets, indices, permuted_keys) = Partition(keys);
  auto offsets_ptr = offsets.data_ptr<int64_t>();
  std::lock_guard lock(mtx_);
  torch::parallel_for(0, policies_.size(), 1, [&](int64_t begin, int64_t end) {
    if (begin == end) return;
    const auto tid = begin;
    begin = offsets_ptr[tid];
    end = offsets_ptr[tid + 1];
    if constexpr (write)
      policies_.at(tid)->WritingCompleted(permuted_keys.slice(0, begin, end));
    else
      policies_.at(tid)->ReadingCompleted(permuted_keys.slice(0, begin, end));
  });
}

void PartitionedCachePolicy::ReadingCompleted(torch::Tensor keys) {
  ReadingWritingCompletedImpl<false>(keys);
}

void PartitionedCachePolicy::WritingCompleted(torch::Tensor keys) {
  ReadingWritingCompletedImpl<true>(keys);
}

c10::intrusive_ptr<Future<void>> PartitionedCachePolicy::ReadingCompletedAsync(
    torch::Tensor keys) {
  return async([=] { return ReadingCompleted(keys); });
}

c10::intrusive_ptr<Future<void>> PartitionedCachePolicy::WritingCompletedAsync(
    torch::Tensor keys) {
  return async([=] { return WritingCompleted(keys); });
}

template <typename CachePolicy>
c10::intrusive_ptr<PartitionedCachePolicy> PartitionedCachePolicy::Create(
    int64_t capacity, int64_t num_partitions) {
  static_assert(std::is_base_of_v<BaseCachePolicy, CachePolicy>);
  return c10::make_intrusive<PartitionedCachePolicy>(
      CachePolicy(), capacity, num_partitions);
}

template c10::intrusive_ptr<PartitionedCachePolicy>
    PartitionedCachePolicy::Create<S3FifoCachePolicy>(int64_t, int64_t);
template c10::intrusive_ptr<PartitionedCachePolicy>
    PartitionedCachePolicy::Create<SieveCachePolicy>(int64_t, int64_t);
template c10::intrusive_ptr<PartitionedCachePolicy>
    PartitionedCachePolicy::Create<LruCachePolicy>(int64_t, int64_t);
template c10::intrusive_ptr<PartitionedCachePolicy>
    PartitionedCachePolicy::Create<ClockCachePolicy>(int64_t, int64_t);

}  // namespace storage
}  // namespace graphbolt
