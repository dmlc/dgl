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

template <bool return_original_positions, typename scalar_t>
std::conditional_t<
    return_original_positions, std::pair<torch::Tensor, torch::Tensor>,
    torch::Tensor>
Sort(const scalar_t* input_keys, int64_t num_items, int num_bits) {
  const auto options = torch::TensorOptions().device(c10::DeviceType::CUDA);
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  constexpr c10::ScalarType dtype = c10::CppTypeToScalarType<scalar_t>::value;
  auto sorted_array = torch::empty(num_items, options.dtype(dtype));
  auto sorted_keys = sorted_array.data_ptr<scalar_t>();
  if (num_bits == 0) {
    num_bits = sizeof(scalar_t) * 8;
  }

  if constexpr (return_original_positions) {
    // We utilize int64_t for the values array. (torch::kLong == int64_t)
    auto original_idx = torch::arange(num_items, options.dtype(torch::kLong));
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
        tmp_storage.get(), tmp_storage_size, input_keys, sorted_keys, num_items,
        0, num_bits, stream));
    return sorted_array;
  }
}

template <bool return_original_positions>
std::conditional_t<
    return_original_positions, std::pair<torch::Tensor, torch::Tensor>,
    torch::Tensor>
Sort(torch::Tensor input, int num_bits) {
  return AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "SortImpl", ([&] {
                                      return Sort<return_original_positions>(
                                          input.data_ptr<scalar_t>(),
                                          input.size(0), num_bits);
                                    }));
}

}  //  namespace ops
}  //  namespace graphbolt
