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
 * @file cuda/unique_and_compact_impl.cu
 * @brief Unique and compact operator implementation on CUDA.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/logical.h>

#include <cub/cub.cuh>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "./common.h"
#include "./extension/unique_and_compact.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

template <typename scalar_t>
struct EqualityFunc {
  const scalar_t* sorted_order;
  const scalar_t* found_locations;
  const scalar_t* searched_items;
  __host__ __device__ auto operator()(int64_t i) {
    return sorted_order[found_locations[i]] == searched_items[i];
  }
};

#define DefineCubReductionFunction(cub_reduce_fn, name)           \
  template <typename scalar_iterator_t>                           \
  auto name(const scalar_iterator_t input, int64_t size) {        \
    using scalar_t = std::remove_reference_t<decltype(input[0])>; \
    cuda::CopyScalar<scalar_t> result;                            \
    CUB_CALL(cub_reduce_fn, input, result.get(), size);           \
    return result;                                                \
  }

DefineCubReductionFunction(DeviceReduce::Max, Max);
DefineCubReductionFunction(DeviceReduce::Min, Min);

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatchedSortBased(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids, int num_bits = 0) {
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  auto scalar_type = src_ids.at(0).scalar_type();
  return AT_DISPATCH_INDEX_TYPES(
      scalar_type, "unique_and_compact", ([&] {
        std::vector<index_t*> src_ids_ptr, dst_ids_ptr, unique_dst_ids_ptr;
        for (std::size_t i = 0; i < src_ids.size(); i++) {
          src_ids_ptr.emplace_back(src_ids[i].data_ptr<index_t>());
          dst_ids_ptr.emplace_back(dst_ids[i].data_ptr<index_t>());
          unique_dst_ids_ptr.emplace_back(
              unique_dst_ids[i].data_ptr<index_t>());
        }

        // If num_bits is not given, compute maximum vertex ids to compute
        // num_bits later to speedup the expensive sort operations.
        std::vector<cuda::CopyScalar<index_t>> max_id_src;
        std::vector<cuda::CopyScalar<index_t>> max_id_dst;
        for (std::size_t i = 0; num_bits == 0 && i < src_ids.size(); i++) {
          max_id_src.emplace_back(Max(src_ids_ptr[i], src_ids[i].size(0)));
          max_id_dst.emplace_back(
              Max(unique_dst_ids_ptr[i], unique_dst_ids[i].size(0)));
        }

        // Sort the unique_dst_ids tensor.
        std::vector<torch::Tensor> sorted_unique_dst_ids;
        std::vector<index_t*> sorted_unique_dst_ids_ptr;
        for (std::size_t i = 0; i < unique_dst_ids.size(); i++) {
          sorted_unique_dst_ids.emplace_back(Sort<false>(
              unique_dst_ids_ptr[i], unique_dst_ids[i].size(0), num_bits));
          sorted_unique_dst_ids_ptr.emplace_back(
              sorted_unique_dst_ids[i].data_ptr<index_t>());
        }

        // Mark dst nodes in the src_ids tensor.
        std::vector<decltype(allocator.AllocateStorage<bool>(0))> is_dst;
        for (std::size_t i = 0; i < src_ids.size(); i++) {
          is_dst.emplace_back(
              allocator.AllocateStorage<bool>(src_ids[i].size(0)));
          THRUST_CALL(
              binary_search, sorted_unique_dst_ids_ptr[i],
              sorted_unique_dst_ids_ptr[i] + unique_dst_ids[i].size(0),
              src_ids_ptr[i], src_ids_ptr[i] + src_ids[i].size(0),
              is_dst[i].get());
        }

        // Filter the non-dst nodes in the src_ids tensor, hence only_src.
        std::vector<torch::Tensor> only_src;
        {
          std::vector<cuda::CopyScalar<int64_t>> only_src_size;
          for (std::size_t i = 0; i < src_ids.size(); i++) {
            only_src.emplace_back(torch::empty(
                src_ids[i].size(0), sorted_unique_dst_ids[i].options()));
            auto is_src = thrust::make_transform_iterator(
                is_dst[i].get(), thrust::logical_not<bool>{});
            only_src_size.emplace_back(cuda::CopyScalar<int64_t>{});
            CUB_CALL(
                DeviceSelect::Flagged, src_ids_ptr[i], is_src,
                only_src[i].data_ptr<index_t>(), only_src_size[i].get(),
                src_ids[i].size(0));
          }
          stream.synchronize();
          for (std::size_t i = 0; i < only_src.size(); i++) {
            only_src[i] =
                only_src[i].slice(0, 0, static_cast<int64_t>(only_src_size[i]));
          }
        }

        // The code block above synchronizes, ensuring safe access to
        // max_id_src and max_id_dst.
        if (num_bits == 0) {
          index_t max_id = 0;
          for (std::size_t i = 0; i < max_id_src.size(); i++) {
            max_id = std::max(max_id, static_cast<index_t>(max_id_src[i]));
            max_id = std::max(max_id, static_cast<index_t>(max_id_dst[i]));
          }
          num_bits = cuda::NumberOfBits(1ll + max_id);
        }

        // Sort the only_src tensor so that we can unique it later.
        std::vector<torch::Tensor> sorted_only_src;
        for (auto& only_src_i : only_src) {
          sorted_only_src.emplace_back(Sort<false>(
              only_src_i.data_ptr<index_t>(), only_src_i.size(0), num_bits));
        }

        std::vector<torch::Tensor> unique_only_src;
        std::vector<index_t*> unique_only_src_ptr;

        std::vector<cuda::CopyScalar<int64_t>> unique_only_src_size;
        for (std::size_t i = 0; i < src_ids.size(); i++) {
          // Compute the unique operation on the only_src tensor.
          unique_only_src.emplace_back(
              torch::empty(only_src[i].size(0), src_ids[i].options()));
          unique_only_src_ptr.emplace_back(
              unique_only_src[i].data_ptr<index_t>());
          unique_only_src_size.emplace_back(cuda::CopyScalar<int64_t>{});
          CUB_CALL(
              DeviceSelect::Unique, sorted_only_src[i].data_ptr<index_t>(),
              unique_only_src_ptr[i], unique_only_src_size[i].get(),
              only_src[i].size(0));
        }
        stream.synchronize();
        for (std::size_t i = 0; i < unique_only_src.size(); i++) {
          unique_only_src[i] = unique_only_src[i].slice(
              0, 0, static_cast<int64_t>(unique_only_src_size[i]));
        }

        std::vector<torch::Tensor> real_order;
        for (std::size_t i = 0; i < unique_dst_ids.size(); i++) {
          real_order.emplace_back(
              torch::cat({unique_dst_ids[i], unique_only_src[i]}));
        }
        // Sort here so that binary search can be used to lookup new_ids.
        std::vector<torch::Tensor> sorted_order, new_ids;
        std::vector<index_t*> sorted_order_ptr;
        std::vector<int64_t*> new_ids_ptr;
        for (std::size_t i = 0; i < real_order.size(); i++) {
          auto [sorted_order_i, new_ids_i] = Sort(real_order[i], num_bits);
          sorted_order_ptr.emplace_back(sorted_order_i.data_ptr<index_t>());
          new_ids_ptr.emplace_back(new_ids_i.data_ptr<int64_t>());
          sorted_order.emplace_back(std::move(sorted_order_i));
          new_ids.emplace_back(std::move(new_ids_i));
        }
        // Holds the found locations of the src and dst ids in the
        // sorted_order. Later is used to lookup the new ids of the src_ids
        // and dst_ids tensors.
        std::vector<decltype(allocator.AllocateStorage<index_t>(0))>
            new_dst_ids_loc;
        for (std::size_t i = 0; i < sorted_order.size(); i++) {
          new_dst_ids_loc.emplace_back(
              allocator.AllocateStorage<index_t>(dst_ids[i].size(0)));
          THRUST_CALL(
              lower_bound, sorted_order_ptr[i],
              sorted_order_ptr[i] + sorted_order[i].size(0), dst_ids_ptr[i],
              dst_ids_ptr[i] + dst_ids[i].size(0), new_dst_ids_loc[i].get());
        }

        std::vector<cuda::CopyScalar<bool>> all_exist;
        at::cuda::CUDAEvent all_exist_event;
        bool should_record = false;
        // Check if unique_dst_ids includes all dst_ids.
        for (std::size_t i = 0; i < dst_ids.size(); i++) {
          if (dst_ids[i].size(0) > 0) {
            thrust::counting_iterator<int64_t> iota(0);
            auto equal_it = thrust::make_transform_iterator(
                iota, EqualityFunc<index_t>{
                          sorted_order_ptr[i], new_dst_ids_loc[i].get(),
                          dst_ids_ptr[i]});
            all_exist.emplace_back(Min(equal_it, dst_ids[i].size(0)));
            should_record = true;
          } else {
            all_exist.emplace_back(cuda::CopyScalar<bool>{});
          }
        }
        if (should_record) all_exist_event.record();

        std::vector<decltype(allocator.AllocateStorage<index_t>(0))>
            new_src_ids_loc;
        for (std::size_t i = 0; i < sorted_order.size(); i++) {
          new_src_ids_loc.emplace_back(
              allocator.AllocateStorage<index_t>(src_ids[i].size(0)));
          THRUST_CALL(
              lower_bound, sorted_order_ptr[i],
              sorted_order_ptr[i] + sorted_order[i].size(0), src_ids_ptr[i],
              src_ids_ptr[i] + src_ids[i].size(0), new_src_ids_loc[i].get());
        }

        // Finally, lookup the new compact ids of the src and dst tensors
        // via gather operations.
        std::vector<torch::Tensor> new_src_ids;
        for (std::size_t i = 0; i < src_ids.size(); i++) {
          new_src_ids.emplace_back(torch::empty_like(src_ids[i]));
          THRUST_CALL(
              gather, new_src_ids_loc[i].get(),
              new_src_ids_loc[i].get() + src_ids[i].size(0),
              new_ids[i].data_ptr<int64_t>(),
              new_src_ids[i].data_ptr<index_t>());
        }
        // Perform check before we gather for the dst indices.
        for (std::size_t i = 0; i < dst_ids.size(); i++) {
          if (dst_ids[i].size(0) > 0) {
            if (should_record) {
              all_exist_event.synchronize();
              should_record = false;
            }
            if (!static_cast<bool>(all_exist[i])) {
              throw std::out_of_range("Some ids not found.");
            }
          }
        }
        std::vector<torch::Tensor> new_dst_ids;
        for (std::size_t i = 0; i < dst_ids.size(); i++) {
          new_dst_ids.emplace_back(torch::empty_like(dst_ids[i]));
          THRUST_CALL(
              gather, new_dst_ids_loc[i].get(),
              new_dst_ids_loc[i].get() + dst_ids[i].size(0),
              new_ids[i].data_ptr<int64_t>(),
              new_dst_ids[i].data_ptr<index_t>());
        }
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
            results;
        for (std::size_t i = 0; i < src_ids.size(); i++) {
          results.emplace_back(
              std::move(real_order[i]), std::move(new_src_ids[i]),
              std::move(new_dst_ids[i]));
        }
        return results;
      }));
}

std::vector<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatched(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor>& unique_dst_ids, const int64_t rank,
    const int64_t world_size) {
  if (cuda::compute_capability() >= 70) {
    // Utilizes a hash table based implementation, the mapped id of a vertex
    // will be monotonically increasing as the first occurrence index of it in
    // torch.cat([unique_dst_ids, src_ids]). Thus, it is deterministic.
    return UniqueAndCompactBatchedHashMapBased(
        src_ids, dst_ids, unique_dst_ids, rank, world_size);
  }
  TORCH_CHECK(
      world_size <= 1,
      "Cooperative Minibatching (arXiv:2310.12403) is not supported on "
      "pre-Volta generation GPUs.");
  // Utilizes a sort based algorithm, the mapped id of a vertex part of the
  // src_ids but not part of the unique_dst_ids will be monotonically increasing
  // as the actual vertex id increases. Thus, it is deterministic.
  auto results3 =
      UniqueAndCompactBatchedSortBased(src_ids, dst_ids, unique_dst_ids);
  std::vector<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
      results4;
  auto offsets = torch::zeros(
      2 * results3.size(),
      c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
  for (const auto& [a, b, c] : results3) {
    auto d = offsets.slice(0, 0, 2);
    d.data_ptr<int64_t>()[1] = a.size(0);
    results4.emplace_back(a, b, c, d);
    offsets = offsets.slice(0, 2);
  }
  return results4;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, const int64_t rank,
    const int64_t world_size) {
  return UniqueAndCompactBatched(
      {src_ids}, {dst_ids}, {unique_dst_ids}, rank, world_size)[0];
}

}  // namespace ops
}  // namespace graphbolt
