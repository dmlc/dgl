/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/unique_and_compact_impl.cu
 * @brief Unique and compact operator implementation on CUDA.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/logical.h>

#include <cub/cub.cuh>
#include <cuco/static_map.cuh>
#include <cuda/std/atomic>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <unordered_map>

#include "./common.h"
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

template <typename index_t, typename map_t>
__global__ void _InsertAndSetMinBatched(
    const int64_t num_edges, const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const int64_t tensor_index = indexes[i];
    const auto tensor_offset = i - offsets[tensor_index];
    const int64_t node_id = pointers[tensor_index][tensor_offset];
    const auto batch_index = tensor_index / 2;
    const int64_t key = node_id | (batch_index << 48);

    auto [slot, is_new_key] = map.insert_and_find(cuco::pair{key, i});

    if (!is_new_key) {
      auto ref = ::cuda::atomic_ref<int64_t, ::cuda::thread_scope_device>{
          slot->second};
      ref.fetch_min(i, ::cuda::memory_order_relaxed);
    }

    i += stride;
  }
}

template <typename index_t, typename map_t>
__global__ void _IsInsertedBatched(
    const int64_t num_edges, const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map, int64_t* valid) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const int64_t tensor_index = indexes[i];
    const auto tensor_offset = i - offsets[tensor_index];
    const int64_t node_id = pointers[tensor_index][tensor_offset];
    const auto batch_index = tensor_index / 2;
    const int64_t key = node_id | (batch_index << 48);

    auto slot = map.find(key);
    valid[i] = slot->second == i;

    i += stride;
  }
}

template <typename index_t, typename map_t>
__global__ void _GetInsertedBatched(
    const int64_t num_edges, const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map, const int64_t* const valid,
    index_t* unique_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const auto valid_i = valid[i];

    if (valid_i + 1 == valid[i + 1]) {
      const int64_t tensor_index = indexes[i];
      const auto tensor_offset = i - offsets[tensor_index];
      const int64_t node_id = pointers[tensor_index][tensor_offset];
      const auto batch_index = tensor_index / 2;
      const int64_t key = node_id | (batch_index << 48);

      auto slot = map.find(key);
      const auto batch_offset = offsets[batch_index * 2];
      const auto new_id = valid_i - valid[batch_offset];
      unique_ids[valid_i] = node_id;
      slot->second = new_id;
    }

    i += stride;
  }
}

template <typename index_t, typename map_t>
__global__ void _MapIdsBatched(
    const int num_batches, const int64_t num_edges,
    const int32_t* const indexes, index_t** pointers,
    const int64_t* const offsets, map_t map, index_t* mapped_ids) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  while (i < num_edges) {
    const int64_t tensor_index = indexes[i];
    int64_t batch_index;

    if (tensor_index >= 2 * num_batches) {
      batch_index = tensor_index - 2 * num_batches;
    } else if (tensor_index & 1) {
      batch_index = tensor_index / 2;
    } else {
      batch_index = -1;
    }

    // Only map src or dst ids.
    if (batch_index >= 0) {
      const auto tensor_offset = i - offsets[tensor_index];
      const int64_t node_id = pointers[tensor_index][tensor_offset];
      const int64_t key = node_id | (batch_index << 48);

      auto slot = map.find(key);
      mapped_ids[i] = slot->second;
    }

    i += stride;
  }
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatchedSort(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor> unique_dst_ids, int num_bits) {
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

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatchedMap(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor> unique_dst_ids) {
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  auto scalar_type = src_ids.at(0).scalar_type();
  constexpr int BLOCK_SIZE = 512;
  const auto num_batches = src_ids.size();
  static_assert(
      sizeof(std::ptrdiff_t) == sizeof(int64_t),
      "Need to be compiled on a 64-bit system.");
  TORCH_CHECK(
      num_batches <= (1 << 15),
      "UniqueAndCompactBatched supports a batch size of up to 32768");
  return AT_DISPATCH_INDEX_TYPES(
      scalar_type, "unique_and_compact", ([&] {
        auto pointers_and_offsets = torch::empty(
            6 * num_batches + 1,
            c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        auto pointers_ptr =
            reinterpret_cast<index_t**>(pointers_and_offsets.data_ptr());
        auto offsets_ptr =
            pointers_and_offsets.data_ptr<int64_t>() + 3 * num_batches;
        for (std::size_t i = 0; i < num_batches; i++) {
          pointers_ptr[2 * i] = unique_dst_ids[i].data_ptr<index_t>();
          offsets_ptr[2 * i] = unique_dst_ids[i].size(0);
          pointers_ptr[2 * i + 1] = src_ids[i].data_ptr<index_t>();
          offsets_ptr[2 * i + 1] = src_ids[i].size(0);
          pointers_ptr[2 * num_batches + i] = dst_ids[i].data_ptr<index_t>();
          offsets_ptr[2 * num_batches + i] = dst_ids[i].size(0);
        }
        std::exclusive_scan(
            offsets_ptr, offsets_ptr + 3 * num_batches + 1, offsets_ptr, 0ll);
        auto pointers_and_offsets_dev =
            pointers_and_offsets.to(stream.device());
        auto offsets_dev = pointers_and_offsets_dev.slice(0, 3 * num_batches);
        auto pointers_dev_ptr =
            reinterpret_cast<index_t**>(pointers_and_offsets_dev.data_ptr());
        auto offsets_dev_ptr = offsets_dev.data_ptr<int64_t>();
        auto indexes = ExpandIndptrImpl(
            offsets_dev, torch::kInt32, torch::nullopt,
            offsets_ptr[3 * num_batches]);
        auto map = cuco::static_map{
            offsets_ptr[2 * num_batches],
            0.5,  // load_factor
            cuco::empty_key{static_cast<int64_t>(-1)},
            cuco::empty_value{static_cast<int64_t>(-1)},
            {},
            cuco::linear_probing<1, cuco::default_hash_function<int64_t>>{},
            {},
            {},
            cuda::CUDAWorkspaceAllocator<cuco::pair<int64_t, int64_t>>{},
            cuco::cuda_stream_ref{stream},
        };
        C10_CUDA_KERNEL_LAUNCH_CHECK();  // Check the map constructor's success.
        const dim3 block(BLOCK_SIZE);
        const dim3 grid(
            (offsets_ptr[2 * num_batches] + BLOCK_SIZE - 1) / BLOCK_SIZE);
        CUDA_KERNEL_CALL(
            _InsertAndSetMinBatched, grid, block, 0,
            offsets_ptr[2 * num_batches], indexes.data_ptr<int32_t>(),
            pointers_dev_ptr, offsets_dev_ptr, map.ref(cuco::insert_and_find));
        auto valid = torch::empty(
            offsets_ptr[2 * num_batches] + 1,
            src_ids[0].options().dtype(torch::kInt64));
        CUDA_KERNEL_CALL(
            _IsInsertedBatched, grid, block, 0, offsets_ptr[2 * num_batches],
            indexes.data_ptr<int32_t>(), pointers_dev_ptr, offsets_dev_ptr,
            map.ref(cuco::find), valid.data_ptr<int64_t>());
        valid = ExclusiveCumSum(valid);
        auto unique_ids_offsets = torch::empty(
            num_batches + 1,
            c10::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        auto unique_ids_offsets_ptr = unique_ids_offsets.data_ptr<int64_t>();
        for (int64_t i = 0; i <= num_batches; i++) {
          unique_ids_offsets_ptr[i] = offsets_ptr[2 * i];
        }
        THRUST_CALL(
            gather, unique_ids_offsets_ptr,
            unique_ids_offsets_ptr + unique_ids_offsets.size(0),
            valid.data_ptr<int64_t>(), unique_ids_offsets_ptr);
        at::cuda::CUDAEvent unique_ids_offsets_event;
        unique_ids_offsets_event.record();
        auto unique_ids =
            torch::empty(offsets_ptr[2 * num_batches], src_ids[0].options());
        CUDA_KERNEL_CALL(
            _GetInsertedBatched, grid, block, 0, offsets_ptr[2 * num_batches],
            indexes.data_ptr<int32_t>(), pointers_dev_ptr, offsets_dev_ptr,
            map.ref(cuco::find), valid.data_ptr<int64_t>(),
            unique_ids.data_ptr<index_t>());
        auto mapped_ids =
            torch::empty(offsets_ptr[3 * num_batches], unique_ids.options());
        CUDA_KERNEL_CALL(
            _MapIdsBatched, grid, block, 0, num_batches,
            offsets_ptr[3 * num_batches], indexes.data_ptr<int32_t>(),
            pointers_dev_ptr, offsets_dev_ptr, map.ref(cuco::find),
            mapped_ids.data_ptr<index_t>());
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
            results;
        unique_ids_offsets_event.synchronize();
        for (int64_t i = 0; i < num_batches; i++) {
          results.emplace_back(
              unique_ids.slice(
                  0, unique_ids_offsets_ptr[i], unique_ids_offsets_ptr[i + 1]),
              mapped_ids.slice(
                  0, offsets_ptr[2 * i + 1], offsets_ptr[2 * i + 2]),
              mapped_ids.slice(
                  0, offsets_ptr[2 * num_batches + i],
                  offsets_ptr[2 * num_batches + i + 1]));
        }
        return results;
      }));
}

std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
UniqueAndCompactBatched(
    const std::vector<torch::Tensor>& src_ids,
    const std::vector<torch::Tensor>& dst_ids,
    const std::vector<torch::Tensor> unique_dst_ids, int num_bits) {
  auto dev_id = cuda::GetCurrentStream().device_index();
  static std::mutex mtx;
  static std::unordered_map<decltype(dev_id), int> compute_capability_cache;
  const auto compute_capability_major = [&] {
    std::lock_guard lock(mtx);
    auto it = compute_capability_cache.find(dev_id);
    if (it != compute_capability_cache.end()) {
      return it->second;
    } else {
      int major;
      CUDA_DRIVER_CHECK(cuDeviceGetAttribute(
          &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_id));
      return compute_capability_cache[dev_id] = major;
    }
  }();
  if (compute_capability_major >= 7) {
    return UniqueAndCompactBatchedMap(src_ids, dst_ids, unique_dst_ids);
  } else {
    return UniqueAndCompactBatchedSort(
        src_ids, dst_ids, unique_dst_ids, num_bits);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, int num_bits) {
  return UniqueAndCompactBatched(
      {src_ids}, {dst_ids}, {unique_dst_ids}, num_bits)[0];
}

}  // namespace ops
}  // namespace graphbolt
