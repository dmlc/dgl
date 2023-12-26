/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/sort_impl.cu
 * @brief Sort implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>

#include <cub/cub.cuh>

#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

template <bool return_original_positions>
std::conditional_t<
    return_original_positions, std::pair<torch::Tensor, torch::Tensor>,
    torch::Tensor>
Sort(torch::Tensor input, int num_bits) {
  int64_t num_items = input.size(0);
  auto sorted_array = torch::empty_like(input);
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  return AT_DISPATCH_INDEX_TYPES(
      input.scalar_type(), "SortImpl", ([&] {
        const auto input_keys = input.data_ptr<index_t>();
        index_t* sorted_keys = sorted_array.data_ptr<index_t>();
        if (num_bits == 0) {
          num_bits = sizeof(index_t) * 8;
        }

        if constexpr (return_original_positions) {
          // We utilize int64_t for the values array. (torch::kLong == int64_t)
          auto original_idx =
              torch::arange(num_items, input.options().dtype(torch::kLong));
          auto sorted_idx = torch::empty_like(original_idx);
          const int64_t* input_values = original_idx.data_ptr<int64_t>();
          int64_t* sorted_values = sorted_idx.data_ptr<int64_t>();
          size_t tmp_storage_size = 0;
          CUDA_CALL(cub::DeviceRadixSort::SortPairs(
              nullptr, tmp_storage_size, input_keys, sorted_keys, input_values,
              sorted_values, num_items, 0, num_bits, stream));
          auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
          CUDA_CALL(cub::DeviceRadixSort::SortPairs(
              tmp_storage.get(), tmp_storage_size, input_keys, sorted_keys,
              input_values, sorted_values, num_items, 0, num_bits, stream));
          return std::make_pair(sorted_array, sorted_idx);
        } else {
          size_t tmp_storage_size = 0;
          CUDA_CALL(cub::DeviceRadixSort::SortKeys(
              nullptr, tmp_storage_size, input_keys, sorted_keys, num_items, 0,
              num_bits, stream));
          auto tmp_storage = allocator.AllocateStorage<char>(tmp_storage_size);
          CUDA_CALL(cub::DeviceRadixSort::SortKeys(
              tmp_storage.get(), tmp_storage_size, input_keys, sorted_keys,
              num_items, 0, num_bits, stream));
          return sorted_array;
        }
      }));
}

}  //  namespace ops
}  //  namespace graphbolt
