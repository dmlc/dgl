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

#include <algorithm>
#include <limits>
#include <numeric>

#include "./utils.h"

namespace graphbolt {
namespace storage {

constexpr int kIntGrainSize = 256;

torch::Tensor AddOffset(torch::Tensor keys, int64_t offset) {
  if (offset == 0) return keys;
  auto output = torch::empty_like(
      keys, keys.options().pinned_memory(utils::is_pinned(keys)));
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "AddOffset", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto output_ptr = output.data_ptr<index_t>();
        graphbolt::parallel_for_each(
            0, keys.numel(), kIntGrainSize, [&](int64_t i) {
              const auto result = keys_ptr[i] + offset;
              if constexpr (!std::is_same_v<index_t, int64_t>) {
                TORCH_CHECK(
                    std::numeric_limits<index_t>::min() <= result &&
                    result <= std::numeric_limits<index_t>::max());
              }
              output_ptr[i] = static_cast<index_t>(result);
            });
      }));
  return output;
}

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
  torch::Tensor offsets = torch::empty(
      num_parts * num_parts + 1, keys.options().dtype(torch::kInt64));
  auto offsets_ptr = offsets.data_ptr<int64_t>();
  std::fill_n(offsets_ptr, offsets.size(0), int64_t{});
  auto indices = torch::empty_like(keys, keys.options().dtype(torch::kInt64));
  auto part_id = torch::empty_like(keys, keys.options().dtype(torch::kInt32));
  const auto num_keys = keys.size(0);
  auto part_id_ptr = part_id.data_ptr<int32_t>();
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "PartitionedCachePolicy::partition", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        namespace gb = graphbolt;
        gb::parallel_for_each(0, num_parts, 1, [&](int64_t tid) {
          const auto begin = tid * num_keys / num_parts;
          const auto end = (tid + 1) * num_keys / num_parts;
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
  graphbolt::parallel_for_each(
      0, num_parts * num_parts, kIntGrainSize, [&](int64_t i) {
        const auto part_id = i % num_parts;
        const auto tid = i / num_parts;
        // + 1 so that we have exclusive_scan after torch.cumsum().
        offsets_permuted_ptr[part_id * num_parts + tid + 1] = offsets_ptr[i];
      });
  offsets_permuted_ptr[0] = 0;
  // offsets = offsets_permuted.cumsum(0); @TODO implement this in parallel.
  std::inclusive_scan(
      offsets_permuted_ptr, offsets_permuted_ptr + num_parts * num_parts + 1,
      offsets_ptr);
  offsets_ptr = offsets.data_ptr<int64_t>();
  graphbolt::parallel_for_each(
      0, num_parts * num_parts, kIntGrainSize, [&](int64_t i) {
        const auto part_id = i % num_parts;
        const auto tid = i / num_parts;
        offsets_permuted_ptr[i] = offsets_ptr[part_id * num_parts + tid];
      });
  auto indices_ptr = indices.data_ptr<int64_t>();
  auto permuted_keys = torch::empty_like(keys);
  auto offsets_sliced = torch::empty(num_parts + 1, offsets.options());
  auto offsets_sliced_ptr = offsets_sliced.data_ptr<int64_t>();
  offsets_sliced_ptr[0] = 0;
  AT_DISPATCH_INDEX_TYPES(
      keys.scalar_type(), "PartitionedCachePolicy::partition", ([&] {
        auto keys_ptr = keys.data_ptr<index_t>();
        auto permuted_keys_ptr = permuted_keys.data_ptr<index_t>();
        namespace gb = graphbolt;
        gb::parallel_for_each(0, num_parts, 1, [&](int64_t tid) {
          const auto begin = tid * num_keys / num_parts;
          const auto end = (tid + 1) * num_keys / num_parts;
          for (int64_t i = begin; i < end; i++) {
            const auto part_id = part_id_ptr[i];
            auto& offset = offsets_permuted_ptr[tid * num_parts + part_id];
            indices_ptr[offset] = i;
            permuted_keys_ptr[offset++] = keys_ptr[i];
          }
          offsets_sliced_ptr[tid + 1] = offsets_ptr[(tid + 1) * num_parts];
        });
      }));
  return {offsets_sliced, indices, permuted_keys};
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor>
PartitionedCachePolicy::Query(torch::Tensor keys, const int64_t offset) {
  keys = AddOffset(keys, offset);
  if (policies_.size() == 1) {
    std::lock_guard lock(mtx_);
    auto [positions, output_indices, missing_keys, found_pointers] =
        policies_[0]->Query(keys);
    auto found_and_missing_offsets = torch::empty(4, found_pointers.options());
    auto found_and_missing_offsets_ptr =
        found_and_missing_offsets.data_ptr<int64_t>();
    // Found offsets part.
    found_and_missing_offsets_ptr[0] = 0;
    found_and_missing_offsets_ptr[1] = found_pointers.size(0);
    // Missing offsets part.
    found_and_missing_offsets_ptr[2] = 0;
    found_and_missing_offsets_ptr[3] = missing_keys.size(0);
    auto found_offsets = found_and_missing_offsets.slice(0, 0, 2);
    auto missing_offsets = found_and_missing_offsets.slice(0, 2);
    missing_keys = AddOffset(missing_keys, -offset);
    return {positions,      output_indices, missing_keys,
            found_pointers, found_offsets,  missing_offsets};
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
  namespace gb = graphbolt;
  {
    std::lock_guard lock(mtx_);
    gb::parallel_for_each(0, policies_.size(), 1, [&](int64_t tid) {
      const auto begin = offsets_ptr[tid];
      const auto end = offsets_ptr[tid + 1];
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
  torch::Tensor found_pointers = torch::empty(
      positions.size(0),
      std::get<3>(results[0]).options().pinned_memory(utils::is_pinned(keys)));
  auto missing_offsets =
      torch::empty(policies_.size() + 1, result_offsets_tensor.options());
  auto output_indices_ptr = output_indices.data_ptr<int64_t>();
  auto missing_offsets_ptr = missing_offsets.data_ptr<int64_t>();
  missing_offsets_ptr[0] = 0;
  gb::parallel_for_each(0, policies_.size(), 1, [&](int64_t tid) {
    auto out_index_ptr = indices_ptr + offsets_ptr[tid];
    auto begin = result_offsets[tid];
    auto end = result_offsets[tid + 1];
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
    auto selected_pointers_ptr = std::get<3>(results[tid]).data_ptr<int64_t>();
    std::copy(
        selected_pointers_ptr, selected_pointers_ptr + num_selected,
        found_pointers.data_ptr<int64_t>() + begin);
    begin = result_offsets[policies_.size() + tid];
    end = result_offsets[policies_.size() + tid + 1];
    missing_offsets[tid + 1] = end - result_offsets[policies_.size()];
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
  auto found_offsets = result_offsets_tensor.slice(0, 0, policies_.size() + 1);
  missing_keys = AddOffset(missing_keys, -offset);
  return std::make_tuple(
      positions, output_indices, missing_keys, found_pointers, found_offsets,
      missing_offsets);
}

c10::intrusive_ptr<Future<std::vector<torch::Tensor>>>
PartitionedCachePolicy::QueryAsync(torch::Tensor keys, const int64_t offset) {
  return async([=] {
    auto
        [positions, output_indices, missing_keys, found_pointers, found_offsets,
         missing_offsets] = Query(keys, offset);
    return std::vector{positions,      output_indices, missing_keys,
                       found_pointers, found_offsets,  missing_offsets};
  });
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor>
PartitionedCachePolicy::QueryAndReplace(
    torch::Tensor keys, const int64_t offset) {
  keys = AddOffset(keys, offset);
  if (policies_.size() == 1) {
    std::lock_guard lock(mtx_);
    auto [positions, output_indices, pointers, missing_keys] =
        policies_[0]->QueryAndReplace(keys);
    auto found_and_missing_offsets = torch::empty(4, pointers.options());
    auto found_and_missing_offsets_ptr =
        found_and_missing_offsets.data_ptr<int64_t>();
    // Found offsets part.
    found_and_missing_offsets_ptr[0] = 0;
    found_and_missing_offsets_ptr[1] = keys.size(0) - missing_keys.size(0);
    // Missing offsets part.
    found_and_missing_offsets_ptr[2] = 0;
    found_and_missing_offsets_ptr[3] = missing_keys.size(0);
    auto found_offsets = found_and_missing_offsets.slice(0, 0, 2);
    auto missing_offsets = found_and_missing_offsets.slice(0, 2);
    missing_keys = AddOffset(missing_keys, -offset);
    return {positions,    output_indices, pointers,
            missing_keys, found_offsets,  missing_offsets};
  }
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
  namespace gb = graphbolt;
  {
    std::lock_guard lock(mtx_);
    gb::parallel_for_each(0, policies_.size(), 1, [&](int64_t tid) {
      const auto begin = offsets_ptr[tid];
      const auto end = offsets_ptr[tid + 1];
      results[tid] = policies_.at(tid)->QueryAndReplace(
          permuted_keys.slice(0, begin, end));
      const auto missing_cnt = std::get<3>(results[tid]).size(0);
      result_offsets[tid] = end - begin - missing_cnt;
      result_offsets[tid + policies_.size()] = missing_cnt;
    });
  }
  std::exclusive_scan(
      result_offsets, result_offsets + result_offsets_tensor.size(0),
      result_offsets, 0);
  torch::Tensor positions = torch::empty(
      keys.size(0),
      std::get<0>(results[0]).options().pinned_memory(utils::is_pinned(keys)));
  torch::Tensor output_indices = torch::empty_like(
      indices, indices.options().pinned_memory(utils::is_pinned(keys)));
  torch::Tensor pointers = torch::empty(
      keys.size(0),
      std::get<2>(results[0]).options().pinned_memory(utils::is_pinned(keys)));
  torch::Tensor missing_keys = torch::empty(
      result_offsets[2 * policies_.size()] - result_offsets[policies_.size()],
      std::get<3>(results[0]).options().pinned_memory(utils::is_pinned(keys)));
  auto missing_offsets =
      torch::empty(policies_.size() + 1, result_offsets_tensor.options());
  auto positions_ptr = positions.data_ptr<int64_t>();
  auto output_indices_ptr = output_indices.data_ptr<int64_t>();
  auto pointers_ptr = pointers.data_ptr<int64_t>();
  auto missing_offsets_ptr = missing_offsets.data_ptr<int64_t>();
  missing_offsets_ptr[0] = 0;
  gb::parallel_for_each(0, policies_.size(), 1, [&](int64_t tid) {
    auto out_index_ptr = indices_ptr + offsets_ptr[tid];
    auto begin = result_offsets[tid];
    auto end = result_offsets[tid + 1];
    const auto num_selected = end - begin;
    auto indices_ptr = std::get<1>(results[tid]).data_ptr<int64_t>();
    for (int64_t i = 0; i < num_selected; i++) {
      output_indices_ptr[begin + i] = out_index_ptr[indices_ptr[i]];
    }
    auto selected_positions_ptr = std::get<0>(results[tid]).data_ptr<int64_t>();
    std::transform(
        selected_positions_ptr, selected_positions_ptr + num_selected,
        positions_ptr + begin,
        [off = tid * capacity_ / policies_.size()](auto x) { return x + off; });
    auto selected_pointers_ptr = std::get<2>(results[tid]).data_ptr<int64_t>();
    std::copy(
        selected_pointers_ptr, selected_pointers_ptr + num_selected,
        pointers_ptr + begin);
    begin = result_offsets[policies_.size() + tid];
    end = result_offsets[policies_.size() + tid + 1];
    missing_offsets[tid + 1] = end - result_offsets[policies_.size()];
    const auto num_missing = end - begin;
    for (int64_t i = 0; i < num_missing; i++) {
      output_indices_ptr[begin + i] =
          out_index_ptr[indices_ptr[i + num_selected]];
    }
    auto missing_positions_ptr = selected_positions_ptr + num_selected;
    std::transform(
        missing_positions_ptr, missing_positions_ptr + num_missing,
        positions_ptr + begin,
        [off = tid * capacity_ / policies_.size()](auto x) { return x + off; });
    auto missing_pointers_ptr = selected_pointers_ptr + num_selected;
    std::copy(
        missing_pointers_ptr, missing_pointers_ptr + num_missing,
        pointers_ptr + begin);
    std::memcpy(
        reinterpret_cast<std::byte*>(missing_keys.data_ptr()) +
            (begin - result_offsets[policies_.size()]) *
                missing_keys.element_size(),
        std::get<3>(results[tid]).data_ptr(),
        num_missing * missing_keys.element_size());
  });
  auto found_offsets = result_offsets_tensor.slice(0, 0, policies_.size() + 1);
  missing_keys = AddOffset(missing_keys, -offset);
  return std::make_tuple(
      positions, output_indices, pointers, missing_keys, found_offsets,
      missing_offsets);
}

c10::intrusive_ptr<Future<std::vector<torch::Tensor>>>
PartitionedCachePolicy::QueryAndReplaceAsync(
    torch::Tensor keys, const int64_t offset) {
  return async([=] {
    auto
        [positions, output_indices, pointers, missing_keys, found_offsets,
         missing_offsets] = QueryAndReplace(keys, offset);
    return std::vector{positions,    output_indices, pointers,
                       missing_keys, found_offsets,  missing_offsets};
  });
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PartitionedCachePolicy::Replace(
    torch::Tensor keys, torch::optional<torch::Tensor> offsets,
    const int64_t offset) {
  keys = AddOffset(keys, offset);
  if (policies_.size() == 1) {
    std::lock_guard lock(mtx_);
    auto [positions, pointers] = policies_[0]->Replace(keys);
    if (!offsets.has_value()) {
      offsets = torch::empty(2, pointers.options());
      auto offsets_ptr = offsets->data_ptr<int64_t>();
      offsets_ptr[0] = 0;
      offsets_ptr[1] = pointers.size(0);
    }
    return {positions, pointers, *offsets};
  }
  const auto offsets_provided = offsets.has_value();
  torch::Tensor indices, permuted_keys;
  if (!offsets_provided) {
    std::tie(offsets, indices, permuted_keys) = Partition(keys);
  } else {
    permuted_keys = keys;
  }
  auto output_positions = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto output_pointers = torch::empty_like(
      keys, keys.options()
                .dtype(torch::kInt64)
                .pinned_memory(utils::is_pinned(keys)));
  auto offsets_ptr = offsets->data_ptr<int64_t>();
  auto indices_ptr = offsets_provided ? nullptr : indices.data_ptr<int64_t>();
  auto output_positions_ptr = output_positions.data_ptr<int64_t>();
  auto output_pointers_ptr = output_pointers.data_ptr<int64_t>();
  namespace gb = graphbolt;
  std::unique_lock lock(mtx_);
  std::atomic<size_t> semaphore = policies_.size();
  gb::parallel_for_each(0, policies_.size(), 1, [&](int64_t tid) {
    const auto begin = offsets_ptr[tid];
    const auto end = offsets_ptr[tid + 1];
    auto [positions, pointers] =
        policies_.at(tid)->Replace(permuted_keys.slice(0, begin, end));
    const auto ticket = semaphore.fetch_add(-1, std::memory_order_release) - 1;
    if (ticket == 0) {
      // This thread was the last thread in the critical region.
      lock.unlock();
    }
    auto positions_ptr = positions.data_ptr<int64_t>();
    const auto off = tid * capacity_ / policies_.size();
    if (indices_ptr) {
      for (int64_t i = 0; i < positions.size(0); i++) {
        output_positions_ptr[indices_ptr[begin + i]] = positions_ptr[i] + off;
      }
    } else {
      std::transform(
          positions_ptr, positions_ptr + positions.size(0),
          output_positions_ptr + begin, [off](auto x) { return x + off; });
    }
    auto pointers_ptr = pointers.data_ptr<int64_t>();
    std::copy(
        pointers_ptr, pointers_ptr + pointers.size(0),
        output_pointers_ptr + begin);
  });
  return {output_positions, output_pointers, *offsets};
}

c10::intrusive_ptr<Future<std::vector<torch::Tensor>>>
PartitionedCachePolicy::ReplaceAsync(
    torch::Tensor keys, torch::optional<torch::Tensor> offsets,
    const int64_t offset) {
  return async([=] {
    auto [positions, pointers, offsets_out] = Replace(keys, offsets, offset);
    return std::vector{positions, pointers, offsets_out};
  });
}

template <bool write>
void PartitionedCachePolicy::ReadingWritingCompletedImpl(
    torch::Tensor pointers, torch::Tensor offsets) {
  if (policies_.size() == 1) {
    if constexpr (write)
      policies_[0]->WritingCompleted(pointers);
    else
      policies_[0]->ReadingCompleted(pointers);
    return;
  }
  auto offsets_ptr = offsets.data_ptr<int64_t>();
  namespace gb = graphbolt;
  gb::parallel_for_each(0, policies_.size(), 1, [&](int64_t tid) {
    const auto begin = offsets_ptr[tid];
    const auto end = offsets_ptr[tid + 1];
    if constexpr (write)
      policies_.at(tid)->WritingCompleted(pointers.slice(0, begin, end));
    else
      policies_.at(tid)->ReadingCompleted(pointers.slice(0, begin, end));
  });
}

void PartitionedCachePolicy::ReadingCompleted(
    torch::Tensor pointers, torch::Tensor offsets) {
  ReadingWritingCompletedImpl<false>(pointers, offsets);
}

void PartitionedCachePolicy::WritingCompleted(
    torch::Tensor pointers, torch::Tensor offsets) {
  ReadingWritingCompletedImpl<true>(pointers, offsets);
}

c10::intrusive_ptr<Future<void>> PartitionedCachePolicy::ReadingCompletedAsync(
    torch::Tensor pointers, torch::Tensor offsets) {
  return async([=] { return ReadingCompleted(pointers, offsets); });
}

c10::intrusive_ptr<Future<void>> PartitionedCachePolicy::WritingCompletedAsync(
    torch::Tensor pointers, torch::Tensor offsets) {
  return async([=] { return WritingCompleted(pointers, offsets); });
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
